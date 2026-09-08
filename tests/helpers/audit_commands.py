"""Inventory every shell/Python documentation block and check syntax, without executing it.

Syntax validation is NOT runtime validation; the output deliberately distinguishes
them. Full workflow records are produced by validate_examples.py/test_ld_tutorial.py.
"""

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path


def blocks(path):
    lines = path.read_text().splitlines()
    i = 0
    while i < len(lines):
        fence = re.match(r"^(`{3,})(bash|shell|python)\s*$", lines[i])
        rst = re.match(r"^\.\. code-block:: (bash|shell|python)\s*$", lines[i])
        if fence:
            start, language, marker = i + 1, fence[2], fence[1]
            i += 1
            content = []
            while i < len(lines) and lines[i].strip() != marker:
                content.append(lines[i])
                i += 1
            yield start, language, "\n".join(content)
        elif rst:
            start, language = i + 1, rst[1]
            i += 1
            content = []
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("   ")):
                content.append(lines[i][3:] if lines[i].startswith("   ") else "")
                i += 1
            yield start, language, "\n".join(content).strip()
            continue
        i += 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("docs/source"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for path in sorted(args.source.rglob("*")):
        if path.suffix not in (".md", ".rst"):
            continue
        for line, language, code in blocks(path):
            error = ""
            if language == "python":
                try:
                    ast.parse(code)
                except SyntaxError as exc:
                    error = str(exc)
            else:
                result = subprocess.run(["bash", "-n"], input=code, text=True, capture_output=True)
                error = result.stderr if result.returncode else ""
            records.append(
                dict(
                    file=str(path),
                    line=line,
                    language=language,
                    code=code,
                    sha256=hashlib.sha256(code.encode()).hexdigest(),
                    syntax="failed" if error else "passed",
                    error=error,
                    runtime="not established by this syntax-only check",
                )
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(records, indent=2) + "\n")
    failures = [r for r in records if r["error"]]
    print(f"Inventoried {len(records)} executable blocks; {len(failures)} syntax errors")
    if failures:
        raise SystemExit(json.dumps(failures, indent=2))


if __name__ == "__main__":
    main()
