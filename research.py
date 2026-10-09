"""Run a source-grounded survey and download its validated sandbox artifacts.

Usage: python research.py "survey about world model"
"""
import json
import os
import re
import shlex
import sys
import threading
import time
from collections import Counter
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import ModelAPIError, ModelRateLimitError

from agents import FINALIZER_PATH, NOTES_DIR, PREPARER_PATH, REPORT_PATH, SOURCES_PATH, VALIDATOR_PATH, WORKDIR, build_lead_agent
from check_citations import check
from model import make_model
from sandbox import download, open_sandbox, upload
from tools import _redact

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
VALIDATOR_SOURCE = ROOT / "check_citations.py"
FINALIZER_SOURCE = ROOT / "finalize_citations.py"
PREPARER_SOURCE = ROOT / "prepare_sources.py"
FAMILIES = {"arxiv", "hf-daily", "hf-search", "web"}


class _ProgressLogger(BaseCallbackHandler):
    """Print occasional progress without logging prompts, arguments or credentials."""

    def __init__(self):
        self._lock = threading.Lock()
        self._last_log = time.monotonic()
        self._calls = 0

    def on_tool_start(self, serialized, input_str, **kwargs):
        with self._lock:
            self._calls += 1
            if time.monotonic() - self._last_log >= 30:
                name = (serialized or {}).get("name", "tool")
                print(f"Progress: {self._calls} tool calls started; latest: {name}", flush=True)
                self._last_log = time.monotonic()


def slugify(topic):
    """Make a traversal-safe, lower-case file stem of at most 60 characters."""
    return re.sub(r"[\W_]+", "-", str(topic).lower(), flags=re.UNICODE).strip("-")[:60].rstrip("-") or "topic"


def build_prompt(topic):
    """Supply the requested topic and local date to the lead."""
    return (f"Produce a complete English research survey on this topic: {topic.strip()}\n"
            f"Current date: {datetime.now().astimezone().date().isoformat()}.\n"
            "Use at least three independent researcher tasks in parallel, real retrieved "
            "evidence and at least three source labels. Follow the required report structure. "
            f"Save notes in {WORKDIR}/research/notes, sources in {SOURCES_PATH}, "
            f"and the report in {REPORT_PATH}. Finalize citations inside the sandbox, "
            "validate them, spot-check claims, and leave the final validated artifacts there.")


def _field(message, name, default=None):
    return message.get(name, default) if isinstance(message, dict) else getattr(message, name, default)


def summarize(messages, elapsed, model_name):
    """Count lead tool calls and token usage; subagent tokens are not included."""
    calls = Counter()
    subagent_types = Counter()
    tokens = {"input": 0, "output": 0}
    for message in messages:
        for call in _field(message, "tool_calls", []) or []:
            name = _field(call, "name")
            if name:
                calls[name] += 1
            if name == "task":
                arguments = _field(call, "args", {}) or {}
                subagent_types[arguments.get("subagent_type", "unknown")] += 1
        usage = _field(message, "usage_metadata", {}) or {}
        tokens["input"] += usage.get("input_tokens", 0) or 0
        tokens["output"] += usage.get("output_tokens", 0) or 0
    return {"model": model_name, "elapsed_s": round(elapsed, 1),
            "subagent_calls": calls["task"], "subagent_types": dict(subagent_types),
            "tool_calls": dict(calls), "tokens": tokens}


def _validate_sources(sources):
    problems = []
    for source in sources:
        label = source.get("source")
        identifier = source.get("id")
        if label not in FAMILIES:
            problems.append(f"source [{source.get('n')}] has an invalid source family")
        if not isinstance(identifier, str) or not identifier.strip():
            problems.append(f"source [{source.get('n')}] has no id")
        if not isinstance(source.get("title"), str) or not source["title"].strip():
            problems.append(f"source [{source.get('n')}] has no title")
        if not isinstance(source.get("date"), str):
            problems.append(f"source [{source.get('n')}] date must be a string (empty if unknown)")
        if label in {"arxiv", "hf-daily", "hf-search"}:
            prefix = "https://arxiv.org/abs/" if label == "arxiv" else "https://huggingface.co/papers/"
            if source.get("url") != prefix + str(identifier):
                problems.append(f"source [{source.get('n')}] URL does not match its family/id")
            if label == "arxiv" and re.search(r"v\d+$", str(identifier)):
                problems.append(f"source [{source.get('n')}] arXiv ID contains a version suffix")
    if len({s.get("source") for s in sources} & FAMILIES) < 3:
        problems.append("report must retain at least three source families")
    return problems


def _validate_structure(report_text):
    headings = re.findall(r"(?m)^##[ \t]+(.+?)[ \t]*\r?$", report_text)
    required = {"TL;DR", "Background", "Trends and open problems", "References"}
    problems = [f"missing required heading: ## {heading}" for heading in sorted(required - set(headings))]
    themes = [h for h in headings if h.casefold() not in {r.casefold() for r in required}]
    if not 3 <= len(themes) <= 6:
        problems.append(f"report needs 3-6 theme sections in addition to Background/Trends (found {len(themes)})")
    return problems


def _write_bundle(reports_dir, payloads):
    """Stage every artifact before replacing outputs; restore previous files on I/O failure."""
    reports_dir.mkdir(parents=True, exist_ok=True)
    if any((reports_dir / name).exists() and not (reports_dir / name).is_file() for name in payloads):
        raise OSError("an output path exists and is not a regular file")
    with TemporaryDirectory(prefix=".research-", dir=reports_dir) as temporary:
        staging = Path(temporary)
        for name, data in payloads.items():
            (staging / name).write_bytes(data)
        changed = []
        try:
            for name in payloads:
                target = reports_dir / name
                backup = staging / (name + ".backup")
                existed = target.exists()
                if existed:
                    os.replace(target, backup)
                changed.append((target, backup, existed))
                os.replace(staging / name, target)
        except OSError:
            for target, backup, existed in reversed(changed):
                if existed:
                    os.replace(backup, target)
                else:
                    target.unlink(missing_ok=True)
            raise


def save_outputs(backend, topic, messages, elapsed, model_name, reports_dir=REPORTS, *, model_names=None):
    """Preserve downloaded bytes; invalid runs write no artifacts."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report_data, sources_data = files.get(REPORT_PATH), files.get(SOURCES_PATH)
    if not report_data or not report_data.strip():
        raise RuntimeError("sandbox did not produce a non-empty report")
    if not sources_data:
        raise RuntimeError("sandbox did not produce sources.json")
    try:
        report_text = report_data.decode("utf-8")
        sources = json.loads(sources_data.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise RuntimeError("sandbox outputs contain invalid UTF-8 or JSON") from exc
    problems = check(report_text, sources)
    if problems:
        raise RuntimeError("invalid citations: " + "; ".join(problems[:8]))
    problems = _validate_sources(sources)
    problems += _validate_structure(report_text)
    if problems:
        raise RuntimeError("incomplete research: " + "; ".join(problems))
    summary = summarize(messages, elapsed, model_name)
    if summary["subagent_calls"] < 3:
        raise RuntimeError("incomplete research: lead made fewer than three subagent task calls")
    if summary["subagent_types"].get("researcher", 0) < 3:
        raise RuntimeError("incomplete research: lead must delegate at least three researcher tasks, in addition to citation checking")
    metadata = {"topic": topic.strip(), **summary,
                "n_sources": len(sources), "source_families": sorted({s["source"] for s in sources})}
    if model_names:
        metadata["models"] = model_names
    stem = slugify(topic)
    payloads = {f"{stem}.sources.json": sources_data,
                f"{stem}.meta.json": (json.dumps(metadata, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
                f"{stem}.md": report_data}
    _write_bundle(Path(reports_dir), payloads)
    return Path(reports_dir) / f"{stem}.md"


def _execute_checked(backend, command):
    result = backend.execute(command)
    if result.exit_code != 0:
        raise RuntimeError(f"sandbox command failed: {command}: {result.output}")
    return result.output


def _citation_scripts():
    return {VALIDATOR_PATH: VALIDATOR_SOURCE.read_bytes(), FINALIZER_PATH: FINALIZER_SOURCE.read_bytes(),
            PREPARER_PATH: PREPARER_SOURCE.read_bytes()}


class _FinalReviewRequired(RuntimeError):
    """Keep real graph messages when a final review omits its required checker."""

    def __init__(self, result):
        super().__init__("finalized report needs a new citation-checker task using its actual source map")
        self.result = result


def _review_finalized(backend, agent, result, config, *, model_name, fallback_factory=None):
    """Check factual citations against the actual finalized numbering, not cached drafts."""
    files = download(backend, [REPORT_PATH, SOURCES_PATH])
    report = files[REPORT_PATH].decode("utf-8")
    sources = json.loads(files[SOURCES_PATH])
    numbering = [{key: source[key] for key in ("n", "title", "url")} for source in sources]
    before = summarize(result.get("messages", []), 0, model_name)["subagent_types"].get("citation-checker", 0)
    message = ("Host finalization is complete. The ACTUAL report and source map below are authoritative; "
               "earlier drafts, notes and task descriptions may use different citation numbers. "
               "Read this exact map and check ALL named-system citation associations in the report. "
               "You MUST call a NEW task(subagent_type='citation-checker') NOW, even if a checker "
               "already checked an earlier draft. Ask it to spot-check 3-5 factual claims, prioritizing "
               "the specific audit findings supplied earlier, then named systems "
               "in TL;DR and Trends. Give it each verbatim claim, its citation number, and the literal "
               "title/URL resolved from THIS map. A wrong paper title or unrelated URL is UNSUPPORTED, "
               "even if another paper supports that claim. Never pass the URL you intended to cite "
               "instead of the URL currently assigned to the number. Correct unsupported claim/citation "
               "pairs INSIDE the sandbox using the existing evidence notes. Preserve this n/URL map; "
               "edit body citations to the relevant existing numbers, do not rebuild or reorder sources. "
               "Unknown publication days remain empty. Run the protected scripts again and reread "
               "the final source map before stopping. Do not restart evidence collection.\n"
               f"Actual finalized map: {json.dumps(numbering, ensure_ascii=False)}\n"
               f"Actual finalized report:\n{report}")
    reviewed = _run_agent(agent, [*result.get("messages", []), {"role": "user", "content": message}],
                          config, model_name=model_name, fallback_factory=fallback_factory)
    after = summarize(reviewed.get("messages", []), 0, model_name)["subagent_types"].get("citation-checker", 0)
    if after <= before:
        raise _FinalReviewRequired(reviewed)
    return reviewed


def _check_notes(backend):
    """Require actual nonempty evidence files in the sandbox before accepting a run."""
    script = ("import json; from pathlib import Path; "
              f"print(json.dumps([str(p) for p in Path({NOTES_DIR!r}).glob('*.md') "
              "if p.is_file() and p.stat().st_size > 0]))")
    paths = json.loads(_execute_checked(backend, "python3 -c " + shlex.quote(script)))
    if not isinstance(paths, list) or len(paths) < 3:
        raise RuntimeError("incomplete research: researchers must write at least three nonempty evidence notes in the sandbox")


def _run_agent(agent, messages, config, *, model_name="", fallback_factory=None):
    """Keep actual completed graph states so a lead quota cannot discard its work."""
    result = {"messages": messages}
    try:
        for state in agent.stream({"messages": messages}, config=config, stream_mode="values"):
            result = state
    except (ModelRateLimitError, ModelAPIError) as exc:
        message = str(exc)
        scope = re.sub(r"[^a-z]", "", message.lower())
        model_id = model_name.rsplit(":", 1)[-1].removeprefix("models/")
        names_root = bool(model_id and re.search(r"(?<![A-Za-z0-9_.-])" + re.escape(model_id) + r"(?![A-Za-z0-9_.-])", message))
        daily_quota = isinstance(exc, ModelRateLimitError) and "perday" in scope and names_root
        unavailable = isinstance(exc, ModelAPIError) and getattr(exc, "code", None) in {500, 502, 503, 504}
        if not fallback_factory or not (daily_quota or unavailable):
            raise
        print("Lead unavailable or daily quota reached; continuing the saved graph state in the same sandbox.", flush=True)
        replacement = fallback_factory()
        continuation = {"role": "user", "content": "The previous lead exhausted its daily model quota. "
                        "Continue from the actual completed task history and existing sandbox notes/files. "
                        "Do not restart research or rewrite supported passages unnecessarily. Complete any "
                        "remaining citation checks, corrections and protected-script validation."}
        return _run_agent(replacement, [*result.get("messages", []), continuation], config)
    return result


def main(topic, *, repair_existing=False, repair_dir=REPORTS, repair_feedback="", lead_model=None,
         verification_urls=None):
    """Run one topic, always clean up the sandbox; return 0 success, 1 failure, 2 missing topic."""
    if not topic.strip():
        print('Usage: python research.py "<topic>"', file=sys.stderr)
        return 2
    if verification_urls is not None and not repair_existing:
        print("FAILED: verification_urls requires repair_existing=True", file=sys.stderr)
        return 2
    try:
        model = make_model()
        if type(model).__module__.startswith("langchain_google_genai"):
            # The Google client otherwise has no read timeout. A stalled model
            # response must fail within a bounded time so sandbox cleanup runs.
            model.timeout = float(os.getenv("LAB_MODEL_TIMEOUT", "120"))
            if model.timeout <= 0:
                raise ValueError("LAB_MODEL_TIMEOUT must be positive")
            model.max_retries = max(1, min(model.max_retries, 3))
        # Researchers share this model, so one host-side limiter also paces their
        # parallel calls. Gemini free-tier token limits can otherwise be exceeded.
        default_rps = "0.1" if type(model).__module__.startswith("langchain_google_genai") else "0"
        requests_per_second = float(os.getenv("LAB_MODEL_RPS", default_rps))
        if requests_per_second > 0:
            model.rate_limiter = InMemoryRateLimiter(requests_per_second=requests_per_second,
                                                    check_every_n_seconds=0.1, max_bucket_size=1)
        model_name = (os.getenv("LAB_MODEL") or os.getenv("OPENAI_DEPLOYMENT_MODEL")
                      or getattr(model, "model_name", None) or str(type(model).__name__))
        writer_name = model_name
        model_names = None
        if lead_model is not None:
            writer_name = getattr(lead_model, "model_name", None) or getattr(lead_model, "model", None)
            if not isinstance(writer_name, str) or not writer_name:
                writer_name = type(lead_model).__name__
            if type(lead_model).__module__.startswith("langchain_google_genai"):
                lead_model.timeout = float(os.getenv("LAB_MODEL_TIMEOUT", "120"))
                if lead_model.timeout <= 0:
                    raise ValueError("LAB_MODEL_TIMEOUT must be positive")
                lead_model.max_retries = 3
                writer_name = "google_genai:" + writer_name.removeprefix("models/")
            if requests_per_second > 0:
                lead_model.rate_limiter = model.rate_limiter
            model_names = {"collector": writer_name, "lead": writer_name, "subagents": model_name}
        started = time.monotonic()
        print(f"Researching: {topic.strip()}", flush=True)
        with open_sandbox() as backend:
            _execute_checked(backend, f"mkdir -p {WORKDIR}/research/notes {WORKDIR}/report")
            scripts = _citation_scripts()
            upload(backend, scripts)
            if download(backend, list(scripts)) != scripts:
                raise RuntimeError("failed to upload citation scripts to the sandbox")
            prompt = build_prompt(topic)
            collection_prompt = (f"Topic: {topic.strip()}\n"
                                 f"Current date: {datetime.now().astimezone().date().isoformat()}\n"
                                 "Collect evidence only. Delegate three independent researcher tasks "
                                 "in one response, giving each the full context, complete note format "
                                 "and distinct absolute notes path. Do not write a report or check citations.")
            if repair_existing:
                stem = slugify(topic)
                existing_report = (Path(repair_dir) / f"{stem}.md").read_bytes()
                existing_sources = (Path(repair_dir) / f"{stem}.sources.json").read_bytes()
                upload(backend, {REPORT_PATH: existing_report, SOURCES_PATH: existing_sources})
                sources_for_verification = json.loads(existing_sources)
                if verification_urls is not None:
                    requested = set(verification_urls)
                    known = {source["url"] for source in sources_for_verification}
                    if not requested or requested - known:
                        raise ValueError("verification_urls must select existing source URLs")
                    sources_for_verification = [source for source in sources_for_verification
                                                if source["url"] in requested]
                    if len(sources_for_verification) < 3:
                        raise ValueError("focused verification requires at least three existing sources")
                verification_groups = [
                    {"notes_path": f"{WORKDIR}/research/notes/{index + 1:02d}-verification.md",
                     "sources": [{key: source.get(key) for key in ("n", "id", "url", "source")}
                                 for source in sources_for_verification[index::3]]}
                    for index in range(3)
                ]
                prompt += ("\nAn existing report and numbered sources have been uploaded to the paths above. "
                           "Repair them using the complete team workflow. Treat existing claims and citation "
                           "associations as unverified: named systems may point to the wrong paper. "
                           "Use the collector's at least three independent researcher tasks, which fetch "
                           "the EXISTING source URLs and write fresh evidence notes. Preserve the original "
                           "retrieval labels for these existing sources when fetching them for verification. "
                           "Keep existing source URL/n assignments, fix ALL incorrect claim-to-source links, "
                           "remove unsupported claims, and write 3-6 themes plus exact required headings. "
                           "Never invent facts to fill a theme. Verify final named-system citations after "
                           "finalizing and spot-check claims across the entire corrected report.\n"
                           "The completed collector tasks correspond to the THREE URL-verification "
                           "groups below. Reuse those tasks and notes: do not repeat all verification "
                           "groups after evidence collection. Delegate only specific missing evidence, "
                           "with full topic, date, note format and a new notes path. The collector's "
                           "researchers fetch assigned URLs and preserve original source labels. Use this plan instead "
                           "of the default topic-search allocation; do not run arxiv_search or other topic "
                           "searches merely to re-find URLs already provided. Check original dates against "
                           "the fetched evidence; unknown dates stay empty, never invent January 1 dates. "
                           "Existing titles may also be incorrect: replace them with exact retrieved paper "
                           "titles while preserving each source's URL/n and original retrieval label. "
                           "Rewrite unverified passages from fresh notes instead of merely rearranging "
                           "old paragraphs. Give each named system its own supporting citation immediately "
                           "after its claim. After finalizing, have citation-checker inspect EVERY TL;DR "
                           "bullet and EVERY Trends/open-problems claim with the actual final URL mapping. "
                           "Remove unrelated extra citations and unsupported details, then re-finalize. "
                           f"Verification groups: {json.dumps(verification_groups, ensure_ascii=False)}")
                if repair_feedback:
                    prompt += "\nAdditional audit findings to verify against fetched evidence:\n" + repair_feedback
                    collection_prompt += ("\nEvidence audit focus: include these findings in each relevant "
                                          "researcher task; verify them using primary evidence, not inferred "
                                          "search metadata.\n" + repair_feedback)
                if verification_urls is not None:
                    prompt += ("\nThis is a FOCUSED correction of a previously generated report. "
                               "Use edit_file or a narrow Python patch inside the sandbox to correct ONLY "
                               "the specified audit findings. Preserve other supported paragraphs and the "
                               "existing source number/URL map; do not regenerate the whole report or source "
                               "list. Retain previously retrieved source labels. The three fresh evidence "
                               "groups cover the selected existing URLs; reuse their completed tasks.")
                collection_prompt += ("\nUse exactly the three URL-verification groups below instead of topic searches. "
                                      "Each researcher must fetch every assigned URL using web_fetch, record "
                                      "only retrieved facts and exact titles/dates, preserve the original source "
                                      "label, and write its assigned notes file. Unknown dates stay empty. "
                                      "Return evidence and gaps, not citation verdicts or report text.\n"
                                      f"Groups: {json.dumps(verification_groups, ensure_ascii=False)}")
                if verification_urls is not None:
                    collection_prompt += ("\nCOLLECTION ONLY: delegate these three groups immediately, "
                                          "including the audit questions in their task descriptions. "
                                          "Instructions above about editing report text or source metadata "
                                          "are for the later lead, not for you or your researchers. "
                                          "Do not inspect or edit the existing report or sources.json; "
                                          "the groups already supply the source context. Verify evidence "
                                          "and write only the three assigned notes files, then stop.")
            config = {"recursion_limit": 1000, "callbacks": [_ProgressLogger()]}
            collector = (build_lead_agent(backend, lead_model, research_only=True,
                                         verify_existing=repair_existing, subagent_model=model)
                         if lead_model is not None else
                         build_lead_agent(backend, model, research_only=True, verify_existing=repair_existing))
            def collector_fallback():
                nonlocal collector
                collector = build_lead_agent(backend, model, research_only=True, verify_existing=repair_existing)
                model_names["collector"] = [writer_name, model_name]
                return collector
            can_fallback = lead_model is not None and writer_name != model_name
            result = _run_agent(collector, [{"role": "user", "content": collection_prompt}], config,
                                model_name=writer_name, fallback_factory=collector_fallback if can_fallback else None)
            for collection_attempt in range(3):
                try:
                    _check_notes(backend)
                    counts = summarize(result.get("messages", []), 0, model_name)
                    if counts["subagent_types"].get("researcher", 0) < 3:
                        raise RuntimeError("collection requires three researcher task calls")
                    break
                except RuntimeError as exc:
                    if collection_attempt == 2:
                        raise
                    print(f"Completing evidence notes: {_redact(exc)}", flush=True)
                    result = _run_agent(collector, [*result.get("messages", []),
                        {"role": "user", "content": f"Host evidence validation: {_redact(exc)}. "
                         "Delegate missing researcher tasks with distinct notes paths and the complete "
                         "note schema. Each must fetch its assigned evidence and write its own nonempty "
                         "notes file. Complete at least three researcher tasks and three notes before stopping."}],
                        config, model_name=writer_name, fallback_factory=collector_fallback if can_fallback else None)
            print("Evidence ready: three researcher tasks and notes verified.", flush=True)
            agent = (build_lead_agent(backend, lead_model, subagent_model=model)
                     if lead_model is not None else build_lead_agent(backend, model))
            def writer_fallback():
                nonlocal agent
                agent = build_lead_agent(backend, model)
                model_names["lead"] = [writer_name, model_name]
                return agent
            result = _run_agent(agent, [*result.get("messages", []),
                {"role": "user", "content": prompt + "\nEvidence collection is complete and host-verified. "
                 "Read all three researcher notes and reuse the completed research tasks. Follow the "
                 "original topic and audit instructions to write sources.json and the report, finalize "
                 "citations, verify named-system associations and have citation-checker check every "
                 "TL;DR and Trends claim. Do not repeat collection unless a specific evidence gap requires it."}],
                config, model_name=writer_name, fallback_factory=writer_fallback if can_fallback else None)
            final_review_done = False
            for correction in range(4):
                # Restore host-owned scripts: generated code cannot weaken validation.
                upload(backend, scripts)
                if download(backend, list(scripts)) != scripts:
                    raise RuntimeError("failed to restore trusted citation scripts")
                try:
                    _check_notes(backend)
                    _execute_checked(backend, f"python3 {PREPARER_PATH}")
                    _execute_checked(backend, f"python3 {FINALIZER_PATH}")
                    output = _execute_checked(backend, f"python3 {VALIDATOR_PATH}")
                    if not output.strip().startswith("OK:"):
                        raise RuntimeError(f"sandbox validator did not confirm OK; output: {output!r}")
                    if not final_review_done:
                        print("Reviewing factual citations against the finalized source map.", flush=True)
                        result = _review_finalized(backend, agent, result, config, model_name=writer_name,
                                                  fallback_factory=writer_fallback if can_fallback else None)
                        final_review_done = True
                        upload(backend, scripts)
                        if download(backend, list(scripts)) != scripts:
                            raise RuntimeError("failed to restore trusted citation scripts after review")
                        _check_notes(backend)
                        _execute_checked(backend, f"python3 {PREPARER_PATH}")
                        _execute_checked(backend, f"python3 {FINALIZER_PATH}")
                        output = _execute_checked(backend, f"python3 {VALIDATOR_PATH}")
                        if not output.strip().startswith("OK:"):
                            raise RuntimeError(f"sandbox validator did not confirm OK after review: {output!r}")
                    path = save_outputs(backend, topic, result.get("messages", []),
                                        time.monotonic() - started, writer_name, model_names=model_names)
                    break
                except RuntimeError as exc:
                    if isinstance(exc, _FinalReviewRequired):
                        result = exc.result
                    if correction == 3:
                        raise
                    feedback = _redact(exc)
                    print(f"Repairing outputs ({correction + 1}/3): {feedback}", flush=True)
                    try:
                        current_sources = json.loads(download(backend, [SOURCES_PATH])[SOURCES_PATH])
                        numbering = [{"n": s.get("n"), "url": s.get("url"), "title": s.get("title", "")}
                                     for s in current_sources if isinstance(s, dict)] if isinstance(current_sources, list) else []
                    except (ValueError, TypeError, KeyError):
                        numbering = []
                    message = (f"Host validation found: {feedback}\n"
                               "Repair the existing report/sources INSIDE the sandbox using the existing notes. "
                               "Do not restart all research or edit the protected scripts. If a family is missing, "
                               "cite relevant existing notes or delegate targeted additional research. "
                               "If evidence notes are missing, delegate researcher tasks to fetch the "
                               "provided URLs and write at least three distinct notes files before proceeding. "
                               f"Preserve these source number/URL assignments: {json.dumps(numbering, ensure_ascii=False)}. "
                               "Append new sources with unused numbers; do not reorder or renumber existing sources. "
                               "Run prepare_sources.py, finalize_citations.py and check_citations.py, then "
                               "spot-check any changed claims. Leave the final corrected files in the sandbox.")
                    result = _run_agent(agent, [*result.get("messages", []),
                                              {"role": "user", "content": message}], config,
                                        model_name=writer_name, fallback_factory=writer_fallback if can_fallback else None)
        print(f"Saved: {path}")
        return 0
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {_redact(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    arguments = sys.argv[1:]
    repair = bool(arguments and arguments[0] == "--repair")
    sys.exit(main(" ".join(arguments[1:] if repair else arguments), repair_existing=repair))
