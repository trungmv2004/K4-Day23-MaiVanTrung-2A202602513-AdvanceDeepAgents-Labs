"""Normalize paper identifiers and required heading capitalization inside the sandbox."""
import json
import re
import sys
from urllib.parse import urlsplit

SOURCES = "/tmp/work/research/sources.json"
REPORT = "/tmp/work/report/report.md"


def normalize_headings(report):
    canonical = {name.casefold(): name for name in ("TL;DR", "Background", "Trends and open problems", "References")}
    def replace(match):
        heading = match[1].strip()
        return "## " + canonical.get(heading.casefold(), heading)
    return re.sub(r"(?m)^##[ \t]+([^\r\n]+)[ \t]*\r?$", replace, report)


def normalize(sources):
    """Derive paper IDs from their URLs; never invent or relabel a source family."""
    if not isinstance(sources, list) or not all(isinstance(s, dict) for s in sources):
        raise ValueError("sources.json must be a list of objects")
    normalized, numbers = [], set()
    for source in sources:
        entry = dict(source)
        if entry.get("date") is None:
            entry["date"] = ""
        number = entry.get("n")
        if type(number) is not int or number < 1 or number in numbers:
            raise ValueError(f"source number {number!r} must be a unique positive integer; do not overwrite existing citation numbers")
        numbers.add(number)
        label = entry.get("source")
        parsed = urlsplit(entry.get("url") or "")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"source [{entry.get('n')}] has an invalid HTTP(S) URL")
        if label == "arxiv":
            if parsed.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
                raise ValueError(f"source [{entry.get('n')}] arxiv label has a non-arXiv URL")
            match = re.fullmatch(r"/(?:abs|pdf|html)/(.+?)(?:\.pdf)?/?", parsed.path)
            identifier = re.sub(r"v\d+$", "", match[1]) if match else ""
            if not re.fullmatch(r"(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]*/\d{7})", identifier):
                raise ValueError(f"source [{entry.get('n')}] has an invalid arXiv paper URL")
            entry.update(id=identifier, url=f"https://arxiv.org/abs/{identifier}")
        elif label in {"hf-daily", "hf-search"}:
            if parsed.hostname not in {"huggingface.co", "www.huggingface.co"}:
                raise ValueError(f"source [{entry.get('n')}] Hugging Face label has a different domain")
            match = re.fullmatch(r"/papers/(\d{4}\.\d{4,5})(?:v\d+)?/?", parsed.path)
            if not match:
                raise ValueError(f"source [{entry.get('n')}] has an invalid Hugging Face paper URL")
            entry.update(id=match[1], url=f"https://huggingface.co/papers/{match[1]}")
        elif label == "web":
            if not isinstance(entry.get("id"), str) or not entry["id"].strip():
                entry["id"] = entry["url"]
        else:
            raise ValueError(f"source [{entry.get('n')}] has an unknown source family")
        normalized.append(entry)
    return normalized


def main(argv):
    path = argv[1] if len(argv) > 1 else SOURCES
    report_path = argv[2] if len(argv) > 2 else REPORT
    try:
        with open(path, encoding="utf-8") as handle:
            sources = normalize(json.load(handle))
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(sources, handle, ensure_ascii=False, indent=2)
        try:
            with open(report_path, encoding="utf-8") as handle:
                report = handle.read()
        except FileNotFoundError:
            report = None
        if report is not None:
            prepared_report = normalize_headings(report)
            if prepared_report != report:
                with open(report_path, "w", encoding="utf-8") as handle:
                    handle.write(prepared_report)
        print(f"PREPARED: normalized {len(sources)} source identifiers")
        return 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"cannot prepare sources: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
