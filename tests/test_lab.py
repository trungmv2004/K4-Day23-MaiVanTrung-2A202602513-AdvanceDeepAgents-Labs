"""Offline regression tests for citation integrity, source APIs and failed-run handling."""
import io
import json
import os
import tempfile
import unittest
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import quote

import httpx
from langchain_core.messages import AIMessage, HumanMessage

import agents
import research
import tools
from check_citations import check, main as validator_main
from finalize_citations import finalize
from prepare_sources import normalize, normalize_headings


def sample_sources():
    return [
        {"n": 1, "id": "2501.00001", "url": "https://arxiv.org/abs/2501.00001", "title": "A", "date": "2025-01-01", "source": "arxiv"},
        {"n": 2, "id": "2501.00002", "url": "https://huggingface.co/papers/2501.00002", "title": "B", "date": "2025-01-02", "source": "hf-search"},
        {"n": 3, "id": "https://example.org/c", "url": "https://example.org/c", "title": "C", "date": "", "source": "web"},
    ]


def sample_report():
    body = """# Survey
## TL;DR
- Finding [1].
- Finding [2].
- Finding [3].
## Background
Definition [1].
## First theme
Claim [1, 2].
## Second theme
Comparison [3].
## Third theme
Open question [2].
## Trends and open problems
Trend [3].
"""
    return finalize(body, sample_sources())[0]


def sandbox_fixture_download(backend, paths):
    if paths == [research.SOURCES_PATH]:
        return {research.SOURCES_PATH: json.dumps(sample_sources()).encode()}
    return research._citation_scripts()


class CitationTests(unittest.TestCase):
    def test_accepts_finalizer_output_and_grouped_citations(self):
        for body in ("Claim [1][2][3].", "Claim [1] [2] [3].", "Claim [1] (a note) [2][3].", "Claim [1, 2, 3].", "Claim [1-3].", "Claim [1–3]."):
            references = sample_report().split("## References", 1)[1]
            with self.subTest(body=body):
                self.assertEqual(check(body + "\n## References" + references, sample_sources()), [])

    def test_references_cannot_count_as_body_citations(self):
        references = sample_report().split("## References", 1)[1]
        problems = check("No claims.\n## References" + references, sample_sources())
        self.assertEqual(sum("never cited" in p for p in problems), 3)

    def test_ignores_code_and_markdown_links(self):
        report = sample_report().replace("# Survey", "# Survey\n```python\nx = '[98]'\n```\n~~~text\n[97]\n~~~\n`[96]` [95](https://example.org) [94][label]")
        self.assertEqual(check(report, sample_sources()), [])

    def test_detects_missing_duplicate_and_mismatched_references(self):
        report = sample_report().replace("Claim [1][2]", "Claim [1][2][99]")
        report = report.replace("[2] B.", "[1] B.").replace("https://example.org/c", "https://example.org/wrong")
        problems = check(report, sample_sources())
        for fragment in ("[99] cited", "found 2", "found 0", "URL differs"):
            self.assertTrue(any(fragment in p for p in problems), problems)

    def test_rejects_bundled_urls(self):
        report = sample_report().replace("[1] A.", "[1] A; D https://example.org/d.")
        self.assertTrue(any("exactly one URL" in p for p in check(report, sample_sources())))

    def test_malformed_sources_do_not_crash(self):
        for sources in (None, {}, [], [None], [{"n": True, "url": "ftp://bad"}],
                        [{"n": 1, "url": "https://a"}, {"n": 1, "url": "https://a"}]):
            with self.subTest(sources=sources):
                self.assertTrue(check("body", sources))

    def test_heading_inside_code_is_not_references(self):
        self.assertTrue(any("missing ## References" in p for p in check("```\n## References\n```", sample_sources())))

    def test_parentheses_in_url_and_crlf_code_fences(self):
        sources = sample_sources()
        sources[2]["url"] = "https://example.org/model_(AI)?x=1"
        report = finalize("Claim [1][2][3].\n```text\n[98]\n````\n    [97]\n", sources)[0]
        self.assertEqual(check(report.replace("\n", "\r\n"), sources), [])

    def test_cli_returns_correct_exit_codes(self):
        with tempfile.TemporaryDirectory() as temporary, redirect_stdout(io.StringIO()) as output:
            root = Path(temporary)
            (root / "report.md").write_text(sample_report(), encoding="utf-8")
            (root / "sources.json").write_text(json.dumps(sample_sources()), encoding="utf-8")
            self.assertEqual(validator_main(["check", str(root / "report.md"), str(root / "sources.json")]), 0)
            self.assertIn("OK:", output.getvalue())
            self.assertEqual(validator_main(["check", str(root / "missing"), str(root / "sources.json")]), 1)


class SourcePreparationTests(unittest.TestCase):
    def test_unknown_dates_remain_unknown(self):
        sources = sample_sources()
        sources[0]["date"] = None
        sources[1].pop("date")
        sources[2]["date"] = "2025-07-27"
        prepared = normalize(sources)
        self.assertEqual([s["date"] for s in prepared], ["", "", "2025-07-27"])
        self.assertIsNone(sources[0]["date"])

    def test_heading_case_normalization_preserves_claims_and_theme_titles(self):
        report = "## Trends and Open Problems\nClaim [1].\n## A Named Theme\nComparison [2].\n"
        self.assertEqual(normalize_headings(report), "## Trends and open problems\nClaim [1].\n## A Named Theme\nComparison [2].\n")
    def test_duplicate_numbers_cannot_silently_change_citation_meaning(self):
        sources = sample_sources()
        sources[1]["n"] = 1
        with self.assertRaisesRegex(ValueError, "unique positive integer"):
            normalize(sources)
    def test_paper_ids_come_from_urls(self):
        sources = sample_sources()
        sources[0].update(id="wrong-transcription", url="https://arxiv.org/pdf/2501.00001v2.pdf")
        sources[1].update(id="wrong-transcription", url="https://huggingface.co/papers/2501.00002?utm_source=search")
        normalized = normalize(sources)
        self.assertEqual(normalized[0]["id"], "2501.00001")
        self.assertEqual(normalized[0]["url"], "https://arxiv.org/abs/2501.00001")
        self.assertEqual(normalized[1]["id"], "2501.00002")
        self.assertEqual(sources[0]["id"], "wrong-transcription")

    def test_wrong_family_cannot_be_silently_relabelled(self):
        sources = sample_sources()
        sources[0]["url"] = "https://huggingface.co/papers/2501.00001"
        with self.assertRaisesRegex(ValueError, "non-arXiv"):
            normalize(sources)

    def test_web_origin_and_old_arxiv_ids_are_preserved(self):
        sources = sample_sources()
        sources[0].update(url="http://arxiv.org/abs/cs/9901001v3", id="wrong")
        sources[2].update(url="https://arxiv.org/abs/2501.00003", id="")
        normalized = normalize(sources)
        self.assertEqual(normalized[0]["id"], "cs/9901001")
        self.assertEqual(normalized[2]["source"], "web")
        self.assertEqual(normalized[2]["id"], sources[2]["url"])


class SourceToolTests(unittest.TestCase):
    def test_schema_errors_return_error_strings_without_network_calls(self):
        arguments = [{"query": None}, {"keyword": None}, {"query": None}, {"query": None}, {"url": None}]
        with patch.object(tools, "_request") as request:
            for source_tool, params in zip(tools.SOURCE_TOOLS, arguments):
                with self.subTest(tool=source_tool.name):
                    self.assertTrue(source_tool.invoke(params).startswith("ERROR:"))
            request.assert_not_called()

    def response(self, *, data=None, text=None, status=200, headers=None):
        kwargs = {"json": data} if text is None else {"text": text}
        return httpx.Response(status, headers=headers, request=httpx.Request("GET", "https://example.org"), **kwargs)

    def test_backoff_and_final_attempt(self):
        fn = Mock(side_effect=[tools.RetryableError("busy"), tools.RetryableError("busy"), tools.RetryableError("busy")])
        with patch.object(tools.time, "sleep") as sleep, patch.object(tools.random, "uniform", return_value=0.5):
            with self.assertRaises(tools.RetryableError):
                tools.with_retry(fn, attempts=3, base=2, cap=4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2.5, 4])
            self.assertEqual(fn.call_count, 3)

    def test_retry_after_and_nonretryable_errors(self):
        with patch.object(tools.time, "sleep") as sleep:
            fn = Mock(side_effect=[tools.RetryableError("busy", retry_after=7), "ok"])
            self.assertEqual(tools.with_retry(fn), "ok")
            sleep.assert_called_once_with(7)
            fn = Mock(side_effect=ValueError("bug"))
            with self.assertRaises(ValueError):
                tools.with_retry(fn)
            self.assertEqual(fn.call_count, 1)

    def test_http_status_and_transport_retries(self):
        for status in (429, 500, 502, 503, 504):
            with self.subTest(status=status), patch.object(tools.httpx, "request", return_value=self.response(status=status, headers={"Retry-After": "5"})):
                with self.assertRaises(tools.RetryableError) as caught:
                    tools._request("GET", "https://example.org")
                self.assertEqual(caught.exception.retry_after, 5)
        with patch.object(tools.httpx, "request", side_effect=httpx.ConnectError("offline")):
            with self.assertRaises(tools.RetryableError):
                tools._request("GET", "https://example.org")
        with patch.object(tools.httpx, "request", return_value=self.response(status=403)):
            with self.assertRaises(httpx.HTTPStatusError):
                tools._request("GET", "https://example.org")

    def test_exa_multi_hour_free_quota_fails_without_retry_or_secret_body(self):
        response = self.response(status=429, headers={"Retry-After": "52135"},
                                 data={"error": {"message": "You've hit Exa's free MCP rate limit. "
                                       "URL?exaApiKey=unconfigured-secret"}})
        with patch.object(tools.httpx, "request", return_value=response) as request, \
                patch.object(tools.time, "sleep") as sleep:
            result = tools.web_fetch.invoke({"url": "https://example.org"})
            self.assertIn("Exa free MCP quota exhausted", result)
            self.assertIn("52135s", result)
            self.assertNotIn("unconfigured-secret", result)
            request.assert_called_once()
            sleep.assert_not_called()
        # A short cooldown still follows the ordinary retry path.
        response.headers["Retry-After"] = "5"
        with patch.object(tools.httpx, "request", return_value=response):
            with self.assertRaises(tools.RetryableError):
                tools._request("POST", tools.EXA_URL)

    def test_arxiv_canonicalization_query_and_throttle(self):
        atom = '<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>http://arxiv.org/abs/2501.00001v2</id><published>2025-01-01T00:00:00Z</published><title>A\n paper</title><summary>' + "word " * 200 + '</summary></entry></feed>'
        with patch.object(tools, "_request", return_value=self.response(text=atom)) as request, \
                patch.object(tools, "_ARXIV_LAST_CALL", None), patch.object(tools.time, "monotonic", side_effect=[10, 11, 13]), \
                patch.object(tools.time, "sleep") as sleep:
            records = json.loads(tools.arxiv_search.invoke({"query": 'all:"world" AND model OR', "max_results": 100}))
            tools.arxiv_search.invoke({"query": "world model"})
            self.assertEqual(records[0]["id"], "2501.00001")
            self.assertEqual(records[0]["url"], "https://arxiv.org/abs/2501.00001")
            self.assertEqual(records[0]["title"], "A paper")
            self.assertEqual(len(records[0]["summary"]), 600)
            self.assertEqual(request.call_args_list[0].kwargs["params"]["search_query"], "all:world AND all:model")
            self.assertEqual(request.call_args_list[0].kwargs["params"]["max_results"], 30)
            sleep.assert_called_once_with(2)
        with patch.object(tools, "_request") as request:
            self.assertEqual(tools.arxiv_search.invoke({"query": ':" AND OR'}), "NO RESULTS")
            request.assert_not_called()

    def test_hugging_face_normalization_filter_and_ai_summary(self):
        items = [{"paper": {"id": "a", "title": "World models", "summary": "abstract", "ai_summary": "AI summary", "upvotes": 2}},
                 {"paper": {"id": "b", "title": "WORLD model", "upvotes": 7}}, {"paper": {"title": "missing id"}}]
        with patch.object(tools, "_request", return_value=self.response(data=items)):
            daily = json.loads(tools.hf_daily_papers.invoke({"keyword": "world"}))
            self.assertEqual([r["id"] for r in daily], ["b", "a"])
            search = json.loads(tools.hf_search_papers.invoke({"query": "world"}))
            self.assertEqual(search[0]["summary"], "AI summary")
            self.assertEqual(search[0]["url"], "https://huggingface.co/papers/a")
            self.assertEqual(tools.hf_daily_papers.invoke({"keyword": "unrelated"}), "NO RESULTS")

    def test_exa_sse_metadata_limit_and_arguments(self):
        limited = {"jsonrpc": "2.0", "id": 1, "result": {"_meta": {"rateLimitExceeded": True}, "content": [{"type": "text", "text": "Try later"}]}}
        success = {"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "Paper https://example.org"}]}}
        responses = [self.response(text="event: message\ndata: " + json.dumps(p) + "\n\n") for p in (limited, success)]
        with patch.object(tools, "_request", side_effect=responses) as request, patch.object(tools.time, "sleep") as sleep:
            result = tools.web_search.invoke({"query": "world models"})
            self.assertIn("https://example.org", result)
            sleep.assert_called_once_with(20)
            params = request.call_args.kwargs["json"]["params"]
            self.assertEqual(params["name"], "web_search_exa")
            self.assertTrue(params["arguments"]["objective"])
        with patch.object(tools, "_request", return_value=self.response(data=success)) as request:
            tools.web_fetch.invoke({"url": "https://example.org"})
            self.assertEqual(request.call_args.kwargs["json"]["params"]["arguments"]["urls"], ["https://example.org"])

    def test_exa_json_rpc_error_and_secret_redaction(self):
        secret = "dummy/key+value"
        with patch.dict(os.environ, {"EXA_API_KEY": secret}), patch.object(tools, "_request", side_effect=RuntimeError("URL?exaApiKey=" + quote(secret, safe=""))):
            result = tools.web_search.invoke({"query": "test"})
            self.assertTrue(result.startswith("ERROR:"))
            self.assertNotIn(secret, result)
            self.assertNotIn(quote(secret, safe=""), result)
        with patch.object(tools, "_request", return_value=self.response(data={"error": {"code": -1, "message": "broken"}})):
            self.assertIn("ERROR:", tools.web_fetch.invoke({"url": "https://example.org"}))

    def test_article_about_rate_limits_is_valid_content(self):
        payload = {"result": {"_meta": {"rateLimit": {"remaining": 10}}, "content": [{"type": "text", "text": "Title: Designing API rate limits\nURL: https://example.org"}]}}
        with patch.object(tools, "_request", return_value=self.response(data=payload)), patch.object(tools.time, "sleep") as sleep:
            self.assertIn("Designing API", tools.web_search.invoke({"query": "API limits"}))
            sleep.assert_not_called()

    def test_all_tools_convert_exceptions_to_error_strings(self):
        invocations = [(tools.arxiv_search, {"query": "world"}), (tools.hf_daily_papers, {}),
                       (tools.hf_search_papers, {"query": "world"}), (tools.web_search, {"query": "world"}),
                       (tools.web_fetch, {"url": "https://example.org"})]
        with patch.object(tools, "_request", side_effect=ValueError("bad response")), patch.object(tools.time, "sleep"):
            for source_tool, arguments in invocations:
                with self.subTest(tool=source_tool.name):
                    self.assertTrue(source_tool.invoke(arguments).startswith("ERROR:"))


class ResearchTests(unittest.TestCase):
    def messages(self):
        return [AIMessage(content="", tool_calls=[{"name": "task", "args": {"subagent_type": "researcher"}, "id": str(i)} for i in range(3)],
                          usage_metadata={"input_tokens": 100, "output_tokens": 20, "total_tokens": 120})]

    def files(self):
        return {research.REPORT_PATH: sample_report().encode(), research.SOURCES_PATH: json.dumps(sample_sources()).encode()}

    def test_final_review_uses_downloaded_number_url_map_and_actual_report(self):
        old = {"messages": self.messages()}
        checker = AIMessage(content="", tool_calls=[{"name": "task", "args": {"subagent_type": "citation-checker"}, "id": "final-check"}])
        reviewed = {"messages": [*old["messages"], checker]}
        with patch.object(research, "download", return_value=self.files()), \
                patch.object(research, "_run_agent", return_value=reviewed) as run:
            self.assertIs(research._review_finalized(None, None, old, {}, model_name="model"), reviewed)
        messages = run.call_args.args[1]
        self.assertIs(messages[0], old["messages"][0])
        self.assertIn(sample_report(), messages[-1]["content"])
        self.assertIn('"n": 3, "title": "C", "url": "https://example.org/c"', messages[-1]["content"])
        self.assertIn("Never pass the URL you intended to cite", messages[-1]["content"])

    def test_final_review_rejects_claimed_success_without_a_new_checker_task(self):
        old = {"messages": self.messages()}
        with patch.object(research, "download", return_value=self.files()), \
                patch.object(research, "_run_agent", return_value=old):
            with self.assertRaisesRegex(RuntimeError, "new citation-checker task") as caught:
                research._review_finalized(None, None, old, {}, model_name="model")
        self.assertIs(caught.exception.result, old)

    def test_slug_and_lead_usage_metadata(self):
        for topic in ("../../x", "", "a" * 100, "survey about world model", "đề tài"):
            slug = research.slugify(topic)
            self.assertLessEqual(len(slug), 60)
            self.assertNotIn("/", slug)
            self.assertNotIn(".", slug)
            self.assertTrue(slug)
        metadata = research.summarize([HumanMessage(content="topic"), *self.messages()], 1.26, "model")
        self.assertEqual(metadata["subagent_calls"], 3)
        self.assertEqual(metadata["subagent_types"], {"researcher": 3})
        self.assertEqual(metadata["tokens"], {"input": 100, "output": 20})
        self.assertEqual(metadata["elapsed_s"], 1.3)

    def test_valid_outputs_preserve_sandbox_bytes(self):
        files = self.files()
        with tempfile.TemporaryDirectory() as temporary, patch.object(research, "download", return_value=files):
            path = research.save_outputs(None, "../../x", self.messages(), 1, "model", temporary)
            self.assertEqual(path.parent, Path(temporary))
            self.assertEqual(path.read_bytes(), files[research.REPORT_PATH])
            self.assertEqual(path.with_suffix(".sources.json").read_bytes(), files[research.SOURCES_PATH])
            metadata = json.loads(path.with_suffix(".meta.json").read_text())
            self.assertEqual(metadata["source_families"], ["arxiv", "hf-search", "web"])

    def test_invalid_outputs_never_create_files(self):
        valid = self.files()
        invalid = [dict(valid, **{research.REPORT_PATH: None}), dict(valid, **{research.REPORT_PATH: b"  "}),
                   dict(valid, **{research.SOURCES_PATH: b"{"}), dict(valid, **{research.SOURCES_PATH: b"[]"}),
                   dict(valid, **{research.SOURCES_PATH: b"{}"})]
        for files in invalid:
            with self.subTest(files=files), tempfile.TemporaryDirectory() as temporary, patch.object(research, "download", return_value=files):
                directory = Path(temporary) / "reports"
                with self.assertRaises(RuntimeError):
                    research.save_outputs(None, "topic", self.messages(), 1, "model", directory)
                self.assertFalse(directory.exists())

    def test_missing_delegation_and_source_families_are_failures(self):
        files = self.files()
        with tempfile.TemporaryDirectory() as temporary, patch.object(research, "download", return_value=files):
            with self.assertRaises(RuntimeError):
                research.save_outputs(None, "topic", [], 1, "model", temporary)
            self.assertEqual(list(Path(temporary).iterdir()), [])
        sources = sample_sources()
        sources[2]["source"] = "arxiv"
        sources[2]["url"] = "https://arxiv.org/abs/2501.00003"
        sources[2]["id"] = "2501.00003"
        report = finalize("Claim [1][2][3]", sources)[0]
        files = {research.REPORT_PATH: report.encode(), research.SOURCES_PATH: json.dumps(sources).encode()}
        with tempfile.TemporaryDirectory() as temporary, patch.object(research, "download", return_value=files):
            with self.assertRaisesRegex(RuntimeError, "three source families"):
                research.save_outputs(None, "topic", self.messages(), 1, "model", temporary)

    def test_bundle_restores_existing_outputs_on_write_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            payloads = {"a.md": b"new report", "a.sources.json": b"new sources", "a.meta.json": b"new metadata"}
            previous = {name: b"previous " + name.encode() for name in payloads}
            for name, data in previous.items():
                (root / name).write_bytes(data)
            replace = os.replace
            def fail_once(source, target):
                if Path(source).name == "a.sources.json" and Path(source).parent != root:
                    raise OSError("simulated disk failure")
                return replace(source, target)
            with patch.object(research.os, "replace", side_effect=fail_once), self.assertRaises(OSError):
                research._write_bundle(root, payloads)
            self.assertEqual({name: (root / name).read_bytes() for name in payloads}, previous)
            self.assertEqual(len(list(root.iterdir())), 3)

    def test_main_cleans_up_sandbox_on_agent_failure(self):
        cleaned = []
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                cleaned.append(True)
        agent = Mock()
        agent.stream.side_effect = RuntimeError("agent failed")
        with patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", return_value=research._citation_scripts()), \
                patch.object(research, "build_lead_agent", return_value=agent), patch.object(research, "save_outputs") as save, \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(research.main("topic"), 1)
            save.assert_not_called()
            self.assertEqual(agent.stream.call_args.kwargs["config"]["recursion_limit"], 1000)
        self.assertEqual(cleaned, [True])

    def test_gemini_timeout_reaches_sdk_request_and_failure_cleans_up(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key="offline-test", vertexai=False)
        cleaned = []
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="")
        @contextmanager
        def sandbox():
            try:
                yield backend
            finally:
                cleaned.append(True)
        agent = Mock()
        agent.stream.side_effect = httpx.ReadTimeout("stalled response")
        with patch.dict(os.environ, {"LAB_MODEL_TIMEOUT": "7.5", "LAB_MODEL_RPS": "0"}), \
                patch.object(research, "make_model", return_value=model), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent", return_value=agent), patch.object(research, "save_outputs") as save, \
                redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(research.main("topic"), 1)
            save.assert_not_called()
        request = model._prepare_request([HumanMessage(content="offline timeout regression")])
        self.assertEqual(request["config"].http_options.timeout, 7500)
        self.assertEqual(request["config"].http_options.retry_options.attempts, 3)
        self.assertEqual(cleaned, [True])

    def test_main_restores_trusted_scripts_before_final_validation(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        scripts = research._citation_scripts()
        @contextmanager
        def sandbox():
            yield backend
        with patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload") as upload, patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, patch.object(research, "save_outputs", return_value=Path("report.md")), \
                patch.object(research, "_check_notes"), \
                patch.object(research, "_review_finalized", side_effect=lambda backend, agent, result, config, **kw: result), \
                redirect_stdout(io.StringIO()):
            build.return_value.stream.return_value = [{"messages": self.messages()}]
            self.assertEqual(research.main("topic"), 0)
            self.assertEqual(upload.call_count, 3)
            self.assertEqual(upload.call_args_list[0].args[1], scripts)
            self.assertEqual(upload.call_args_list[1].args[1], scripts)

    def test_lead_model_is_separate_from_research_and_recorded_in_metadata(self):
        backend, base = Mock(), Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        writer = SimpleNamespace(model_name="writer-model")
        @contextmanager
        def sandbox():
            yield backend
        with patch.dict(os.environ, {"LAB_MODEL": "base-model", "LAB_MODEL_RPS": "0"}), \
                patch.object(research, "make_model", return_value=base), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, patch.object(research, "_check_notes"), \
                patch.object(research, "_review_finalized", side_effect=lambda backend, agent, result, config, **kw: result), \
                patch.object(research, "save_outputs", return_value=Path("report.md")) as save, redirect_stdout(io.StringIO()):
            build.return_value.stream.return_value = [{"messages": self.messages()}]
            self.assertEqual(research.main("topic", lead_model=writer), 0)
            self.assertIs(build.call_args_list[0].args[1], writer)
            self.assertIs(build.call_args.args[1], writer)
            self.assertIs(build.call_args.kwargs["subagent_model"], base)
            self.assertEqual(save.call_args.args[4], "writer-model")
            self.assertEqual(save.call_args.kwargs["model_names"],
                             {"collector": "writer-model", "lead": "writer-model", "subagents": "base-model"})

    def test_lead_daily_quota_preserves_actual_graph_history_for_fallback(self):
        from langchain_core.exceptions import ModelRateLimitError
        history = self.messages()
        agent, replacement, factory = Mock(), Mock(), Mock()
        def interrupted(*args, **kwargs):
            yield {"messages": history}
            raise ModelRateLimitError("strong-model GenerateRequestsPerDayPerProjectPerModel")
        agent.stream.side_effect = interrupted
        replacement.stream.return_value = [{"messages": history}]
        factory.return_value = replacement
        with redirect_stdout(io.StringIO()):
            result = research._run_agent(agent, [], {}, model_name="provider:strong-model", fallback_factory=factory)
        factory.assert_called_once_with()
        continued = replacement.stream.call_args.args[0]["messages"]
        self.assertIs(continued[0], history[0])
        self.assertIn("Do not restart research", continued[-1]["content"])
        self.assertEqual(result["messages"], history)

    def test_child_daily_quota_and_minute_quota_do_not_switch_the_lead(self):
        from langchain_core.exceptions import ModelRateLimitError
        for message in ("child-model PerDay", "strong-model-mini PerDay", "strong-model PerMinute"):
            agent, factory = Mock(), Mock()
            agent.stream.side_effect = ModelRateLimitError(message)
            with self.assertRaises(ModelRateLimitError):
                research._run_agent(agent, [], {}, model_name="strong-model", fallback_factory=factory)
            factory.assert_not_called()

    def test_temporary_model_server_failure_can_continue_but_invalid_requests_cannot(self):
        from langchain_core.exceptions import ModelAPIError
        for status in (503, 400):
            agent, replacement, factory = Mock(), Mock(), Mock()
            error = ModelAPIError("server response")
            error.code = status
            agent.stream.side_effect = error
            replacement.stream.return_value = [{"messages": self.messages()}]
            factory.return_value = replacement
            if status == 503:
                with redirect_stdout(io.StringIO()):
                    self.assertTrue(research._run_agent(agent, [], {}, model_name="strong-model", fallback_factory=factory))
                factory.assert_called_once_with()
            else:
                with self.assertRaises(ModelAPIError):
                    research._run_agent(agent, [], {}, model_name="strong-model", fallback_factory=factory)
                factory.assert_not_called()

    def test_verification_delegations_do_not_seed_unverified_titles_or_dates(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        @contextmanager
        def sandbox():
            yield backend
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, patch.object(research, "_check_notes"), \
                patch.object(research, "_review_finalized", side_effect=lambda backend, agent, result, config, **kw: result), \
                patch.object(research, "save_outputs", return_value=Path("report.md")), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "topic.md").write_bytes(self.files()[research.REPORT_PATH])
            (root / "topic.sources.json").write_bytes(self.files()[research.SOURCES_PATH])
            build.return_value.stream.return_value = [{"messages": self.messages()}]
            self.assertEqual(research.main("topic", repair_existing=True, repair_dir=root), 0)
            prompt = build.return_value.stream.call_args_list[0].args[0]["messages"][0]["content"]
            self.assertNotIn('"date":', prompt)
            self.assertNotIn('"title":', prompt)
            self.assertIn('"url": "https://arxiv.org/abs/2501.00001"', prompt)

    def test_focused_repair_selects_evidence_without_rewriting_uploaded_report(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        sources = [*sample_sources(), {"n": 4, "id": "https://example.org/unchanged", "url": "https://example.org/unchanged",
                                     "title": "Untouched", "date": "", "source": "web"}]
        original = finalize(sample_report().split("## References")[0] + "Other supported fact [4].", sources)[0].encode()
        @contextmanager
        def sandbox():
            yield backend
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload") as upload, patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, patch.object(research, "_check_notes"), \
                patch.object(research, "_review_finalized", side_effect=lambda backend, agent, result, config, **kw: result), \
                patch.object(research, "save_outputs", return_value=Path("report.md")), redirect_stdout(io.StringIO()):
            root = Path(temporary)
            (root / "topic.md").write_bytes(original)
            (root / "topic.sources.json").write_text(json.dumps(sources), encoding="utf-8")
            build.return_value.stream.return_value = [{"messages": self.messages()}]
            self.assertEqual(research.main("topic", repair_existing=True, repair_dir=root, repair_feedback="Verify the exact publication day.",
                                          verification_urls=[s["url"] for s in sources[:3]]), 0)
            collector_prompt = build.return_value.stream.call_args_list[0].args[0]["messages"][0]["content"]
            self.assertNotIn("https://example.org/unchanged", collector_prompt)
            self.assertIn("Verify the exact publication day.", collector_prompt)
            writer_prompt = build.return_value.stream.call_args_list[1].args[0]["messages"][-1]["content"]
            self.assertIn("FOCUSED correction", writer_prompt)
            self.assertEqual(upload.call_args_list[1].args[1][research.REPORT_PATH], original)

    def test_focused_repair_rejects_unknown_or_too_few_existing_sources(self):
        backend = Mock()
        backend.execute.return_value = SimpleNamespace(exit_code=0, output="OK: valid")
        @contextmanager
        def sandbox():
            yield backend
        with tempfile.TemporaryDirectory() as temporary, \
                patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            root = Path(temporary)
            (root / "topic.md").write_bytes(self.files()[research.REPORT_PATH])
            (root / "topic.sources.json").write_bytes(self.files()[research.SOURCES_PATH])
            for selected in ([], ["https://example.org/missing"], [s["url"] for s in sample_sources()[:2]]):
                self.assertEqual(research.main("topic", repair_existing=True, repair_dir=root, verification_urls=selected), 1)
            self.assertEqual(research.main("topic", verification_urls=["https://example.org/c"]), 2)
            build.assert_not_called()

    def test_validation_feedback_reuses_existing_sandbox_and_messages(self):
        backend = Mock()
        @contextmanager
        def sandbox():
            yield backend
        with patch.object(research, "make_model", return_value=Mock()), patch.object(research, "open_sandbox", sandbox), \
                patch.object(research, "upload"), patch.object(research, "download", side_effect=sandbox_fixture_download), \
                patch.object(research, "build_lead_agent") as build, patch.object(research, "save_outputs", return_value=Path("report.md")), \
                patch.object(research, "_check_notes"), \
                patch.object(research, "_review_finalized", side_effect=lambda backend, agent, result, config, **kw: result), \
                patch.object(research, "_execute_checked", side_effect=["", RuntimeError("bad source id"), "", "", "OK: valid", "", "", "OK: valid"]), \
                redirect_stdout(io.StringIO()):
            build.return_value.stream.return_value = [{"messages": self.messages()}]
            self.assertEqual(research.main("topic"), 0)
            self.assertEqual(build.return_value.stream.call_count, 3)
            corrected = build.return_value.stream.call_args.args[0]["messages"]
            self.assertEqual(len(corrected), 2)
            self.assertIn("bad source id", corrected[-1]["content"])

    def test_checker_tasks_cannot_replace_independent_research(self):
        messages = [AIMessage(content="", tool_calls=[
            {"name": "task", "args": {"subagent_type": "citation-checker"}, "id": str(i)} for i in range(3)])]
        with tempfile.TemporaryDirectory() as temporary, patch.object(research, "download", return_value=self.files()):
            target = Path(temporary) / "reports"
            with self.assertRaisesRegex(RuntimeError, "three researcher tasks"):
                research.save_outputs(None, "topic", messages, 1, "model", target)
            self.assertFalse(target.exists())

    def test_research_requires_actual_nonempty_sandbox_notes(self):
        backend = Mock()
        for notes in ([], ["one.md"], ["one.md", "two.md"]):
            backend.execute.return_value = SimpleNamespace(exit_code=0, output=json.dumps(notes))
            with self.assertRaisesRegex(RuntimeError, "three nonempty evidence notes"):
                research._check_notes(backend)
        backend.execute.return_value = SimpleNamespace(exit_code=0, output=json.dumps(["one.md", "two.md", "three.md"]))
        research._check_notes(backend)

    def test_complete_researcher_reply_is_persisted_without_changing_evidence(self):
        backend = Mock()
        backend.download_files.return_value = [SimpleNamespace(content=None)]
        backend.write.return_value = SimpleNamespace(error=None)
        path = agents.NOTES_DIR + "/01-verification.md"
        notes = "# Question\nTopic: topic\n## Source 1\ntitle: Retrieved title\nurl: https://example.org\nsource: web\nEvidence:\n- Retrieved fact.\n## Synthesis\nSupported comparison.\n## Gaps\nUnknown date.\n"
        middleware = agents.PersistResearchNotesMiddleware(backend)
        middleware.after_agent({"messages": [HumanMessage(content="Write notes to " + path), AIMessage(content=notes)]}, None)
        backend.write.assert_called_once_with(path, notes)
        backend.write.reset_mock()
        blocked_reply = AIMessage(content=[{"type": "thinking", "thinking": "internal reasoning"},
                                           {"type": "text", "text": notes}])
        middleware.after_agent({"messages": [HumanMessage(content="Write notes to " + path), blocked_reply]}, None)
        backend.write.assert_called_once_with(path, notes)

    def test_notes_fallback_does_not_write_summaries_or_unassigned_paths(self):
        backend = Mock()
        backend.download_files.return_value = [SimpleNamespace(content=None)]
        middleware = agents.PersistResearchNotesMiddleware(backend)
        for path in (agents.NOTES_DIR + "/01.md", agents.NOTES_DIR + "/../report.md", "/tmp/elsewhere.md"):
            middleware.after_agent({"messages": [HumanMessage(content="Write to " + path), AIMessage(content="Finished research, six sources.")]}, None)
        backend.write.assert_not_called()

    def test_notes_fallback_preserves_already_written_notes(self):
        backend = Mock()
        backend.download_files.return_value = [SimpleNamespace(content=b"existing evidence")]
        middleware = agents.PersistResearchNotesMiddleware(backend)
        middleware.after_agent({"messages": [HumanMessage(content=agents.NOTES_DIR + "/01.md"), AIMessage(content="Different summary")]}, None)
        backend.write.assert_not_called()

    def test_report_template_requires_three_themes(self):
        self.assertEqual(research._validate_structure(sample_report()), [])
        report = sample_report().replace("## Third theme\nOpen question [2].\n", "")
        self.assertTrue(any("3-6 theme" in p for p in research._validate_structure(report)))

    def test_subagents_have_independent_limits_and_tools(self):
        first, second = agents.build_subagents(), agents.build_subagents()
        self.assertEqual(first[0]["tools"], tools.SOURCE_TOOLS)
        self.assertEqual(first[1]["tools"], [tools.web_fetch])
        for spec, fresh in zip(first, second):
            self.assertEqual(len(spec["middleware"]), 4)
            self.assertIsNot(spec["middleware"][0], fresh["middleware"][0])

    def test_lead_also_bounds_default_general_purpose_helper(self):
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_lead_agent(None, None)
            specs = create.call_args.kwargs["subagents"]
            self.assertEqual({s["name"] for s in specs}, {"researcher", "citation-checker", "general-purpose"})
            self.assertTrue(all(len(s["middleware"]) == 4 for s in specs))

    def test_collection_phase_cannot_delegate_to_checker(self):
        with patch.object(agents, "create_deep_agent") as create:
            agents.build_lead_agent(None, None, research_only=True)
            specs = create.call_args.kwargs["subagents"]
            self.assertIn("researcher", {s["name"] for s in specs})
            self.assertNotIn("citation-checker", {s["name"] for s in specs})
            self.assertEqual(create.call_args.kwargs["system_prompt"], agents.COLLECTOR_PROMPT)
            agents.build_lead_agent(None, None, research_only=True, verify_existing=True)
            researcher = next(s for s in create.call_args.kwargs["subagents"] if s["name"] == "researcher")
            self.assertEqual(researcher["tools"], [tools.web_fetch])

    def test_model_transport_retry_is_bounded_and_does_not_retry_programming_errors(self):
        retry = agents._limits(3, 6)[-1]
        response = object()
        with patch("langchain.agents.middleware.model_retry.time.sleep") as sleep:
            handler = Mock(side_effect=[httpx.RemoteProtocolError("disconnected"), response])
            self.assertIs(retry.wrap_model_call(Mock(), handler), response)
            self.assertEqual(handler.call_count, 2)
            self.assertEqual(sleep.call_count, 1)
            sleep.reset_mock()
            handler = Mock(side_effect=httpx.ReadTimeout("stalled"))
            with self.assertRaises(httpx.ReadTimeout):
                retry.wrap_model_call(Mock(), handler)
            self.assertEqual(handler.call_count, 3)
            self.assertEqual(sleep.call_count, 2)
            sleep.reset_mock()
            handler = Mock(side_effect=ValueError("invalid configuration"))
            with self.assertRaises(ValueError):
                retry.wrap_model_call(Mock(), handler)
            self.assertEqual(handler.call_count, 1)
            sleep.assert_not_called()

    def test_model_minute_quota_waits_for_reset_and_daily_quota_fails_immediately(self):
        from langchain_core.exceptions import ModelRateLimitError
        retry = agents._limits(3, 6)[-2]
        response = object()
        with patch("langchain.agents.middleware.model_retry.time.sleep") as sleep:
            handler = Mock(side_effect=[ModelRateLimitError("GenerateContentInputTokensPerModelPerMinute-FreeTier; retry in 15s"), response])
            self.assertIs(retry.wrap_model_call(Mock(), handler), response)
            sleep.assert_called_once_with(60)
            sleep.reset_mock()
            handler = Mock(side_effect=ModelRateLimitError("GenerateRequestsPerDayPerProjectPerModel-FreeTier"))
            with self.assertRaises(ModelRateLimitError):
                retry.wrap_model_call(Mock(), handler)
            self.assertEqual(handler.call_count, 1)
            sleep.assert_not_called()
            handler = Mock(side_effect=ModelRateLimitError("PerMinute and PerDay quotas exceeded"))
            with self.assertRaises(ModelRateLimitError):
                retry.wrap_model_call(Mock(), handler)
            self.assertEqual(handler.call_count, 1)

    def test_lead_compiles_with_both_retry_policies(self):
        from langchain_google_genai import ChatGoogleGenerativeAI
        model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", api_key="offline-test", vertexai=False)
        for research_only in (True, False):
            graph = agents.build_lead_agent(None, model, research_only=research_only)
            self.assertTrue(callable(graph.invoke))
        middleware = agents._limits(3, 6)
        self.assertEqual(len({m.name for m in middleware}), len(middleware))


if __name__ == "__main__":
    unittest.main()
