"""Check citations INSIDE the sandbox using only the Python standard library.

research.py uploads this file to the sandbox and the lead agent runs it with the `execute` tool:
    python3 /tmp/work/research/check_citations.py [report.md] [sources.json]
It must exit 0 and print "OK: ..." when the report is consistent, else print each problem and exit 1.
"""
import json
import re
import sys
from collections import Counter

REPORT = "/tmp/work/report/report.md"
SOURCES = "/tmp/work/research/sources.json"


def _without_code(text):
    lines, fence = [], None
    for line in text.splitlines(keepends=True):
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
            lines.append("\n" if line.endswith("\n") else "")
        elif marker:
            fence = marker[1]
            lines.append("\n" if line.endswith("\n") else "")
        elif line.startswith(("    ", "\t")):
            lines.append("\n" if line.endswith("\n") else "")
        else:
            lines.append(line)
    return re.sub(r"(`+).*?\1", "", "".join(lines), flags=re.DOTALL)


def check(report_text, sources):
    """Validate source identities, body citations and one-URL-per-source References."""
    problems = []
    if not isinstance(sources, list) or not sources:
        return ["sources.json must be a non-empty list of objects"]
    by_n, seen_urls = {}, set()
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            problems.append(f"source entry {index + 1} must be an object")
            continue
        n, url = source.get("n"), source.get("url")
        if type(n) is not int or n < 1:
            problems.append(f"source entry {index + 1}: n must be a positive integer")
        elif n in by_n:
            problems.append(f"duplicate source number [{n}]")
        else:
            by_n[n] = source
        if not isinstance(url, str) or not re.match(r"^https?://\S+$", url):
            problems.append(f"source entry {index + 1}: invalid http(s) URL")
        elif url in seen_urls:
            problems.append(f"duplicate source URL: {url}")
        else:
            seen_urls.add(url)

    # Mask code while preserving newlines, so headings inside code do not count.
    visible = _without_code(report_text)
    headings = list(re.finditer(r"(?m)^##[ \t]+References[ \t]*\r?$", visible))
    if not headings:
        problems.append("missing ## References heading")
        body, references = visible, ""
    else:
        if len(headings) != 1:
            problems.append("expected exactly one ## References heading")
        body, references = visible[:headings[0].start()], visible[headings[0].end():]

    cited = set()
    # Adjacent citations [1][2] count, unlike Markdown reference links [1][label].
    body = re.sub(r"!?\[\d+\](?:\([^\n]*?\)|\s*\[[^\]\n]*\])",
                  lambda m: m.group() if re.fullmatch(r"\[\d+\]\s*\[\d+\]", m.group()) else "", body)
    pattern = r"(?<!!)\[(\d+(?:\s*[,–-]\s*\d+)*)\](?!\()"
    for match in re.finditer(pattern, body):
        for part in re.split(r"\s*,\s*", match.group(1)):
            span = re.fullmatch(r"(\d+)\s*[–-]\s*(\d+)", part)
            if span:
                first, last = map(int, span.groups())
                if not 0 <= last - first <= 200:
                    problems.append(f"invalid citation range [{part}]")
                    continue
                cited.update(range(first, last + 1))
            else:
                cited.add(int(part))
    problems.extend(f"[{n}] cited but missing from sources.json" for n in sorted(cited - by_n.keys()))
    problems.extend(f"source [{n}] never cited" for n in sorted(by_n.keys() - cited))

    counts = Counter()
    for line in references.splitlines():
        match = re.match(r"^\[(\d+)\](?:\s|$)", line)
        if not match:
            if line.strip():
                problems.append("unexpected content in References: " + line[:100])
            continue
        n = int(match.group(1))
        counts[n] += 1
        if n not in by_n:
            problems.append(f"reference [{n}] missing from sources.json")
        urls = re.findall(r"https?://[^\s<>]+", line)
        if len(urls) != 1:
            problems.append(f"reference [{n}] must contain exactly one URL")
        elif n in by_n:
            expected = by_n[n].get("url")
            # Preserve parentheses/query punctuation in URLs, allowing Markdown wrappers.
            if not isinstance(expected, str) or not urls[0].startswith(expected) or urls[0][len(expected):].strip(").,;]"):
                problems.append(f"reference [{n}] URL differs from sources.json")
    for n in sorted(by_n):
        if counts[n] != 1:
            problems.append(f"source [{n}] needs exactly one reference line (found {counts[n]})")
    return problems


def main(argv):
    report_path = argv[1] if len(argv) > 1 else REPORT
    sources_path = argv[2] if len(argv) > 2 else SOURCES
    try:
        with open(report_path, encoding="utf-8") as f:
            report = f.read()
        with open(sources_path, encoding="utf-8") as f:
            sources = json.load(f)
    except (OSError, ValueError) as exc:
        print(f"cannot read inputs: {exc}")
        return 1
    problems = check(report, sources)
    if problems:
        print("\n".join(problems))
        return 1
    print(f"OK: {len(sources)} sources, all citations resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
