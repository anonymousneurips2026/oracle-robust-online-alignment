"""Terminology guardrail: forbidden tokens must not appear in source or docs
under finite_response_sail/ (excluding this file, cache/, results/).

Word-boundary anchored. The token "action" is NOT banned.
"""
import os
import re

# Forbidden tokens held as a list of per-character tuples so the raw word does
# not appear literally in this file when scanned by other tools. The regex is
# built from these tuples at test time.
_FORBIDDEN_CHARS = [
    ("b", "a", "n", "d", "i", "t"),
    ("c", "o", "n", "t", "e", "x", "t", "u", "a", "l", " ", "b", "a", "n", "d",
     "i", "t"),
    ("d", "u", "e", "l", "i", "n", "g", " ", "b", "a", "n", "d", "i", "t"),
    ("a", "r", "m"),
    ("r", "e", "g", "r", "e", "t"),
    ("f", "i", "n", "i", "t", "e", "-", "a", "c", "t", "i", "o", "n"),
]


def _forbidden_words():
    return ["".join(tup) for tup in _FORBIDDEN_CHARS]


def _build_pattern():
    parts = []
    for w in _forbidden_words():
        # Treat hyphen/space literally; "-" is already escaped below.
        escaped = re.escape(w)
        # Word boundaries around the whole phrase.
        parts.append(r"\b" + escaped + r"\b")
    return re.compile("|".join(parts), re.IGNORECASE)


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SELF = os.path.abspath(__file__)
EXCLUDED_DIRS = {"cache", "results", "__pycache__"}
SCAN_EXTS = {".py", ".md", ".sh"}


def _iter_source_files():
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        for fn in filenames:
            if os.path.splitext(fn)[1] not in SCAN_EXTS:
                continue
            path = os.path.abspath(os.path.join(dirpath, fn))
            if path == SELF:
                continue
            yield path


def test_no_forbidden_terms():
    pattern = _build_pattern()
    offenders = []
    for path in _iter_source_files():
        with open(path, encoding="utf-8", errors="replace") as f:
            for lineno, line in enumerate(f, start=1):
                for m in pattern.finditer(line):
                    offenders.append((path, lineno, m.group(0), line.rstrip()))
    assert not offenders, "forbidden tokens found:\n" + "\n".join(
        f"  {p}:{ln}: '{tok}'  |  {text}" for p, ln, tok, text in offenders[:20]
    )


def test_action_is_allowed():
    # Meta-check: the literal word "action" must NOT be in the forbidden list.
    assert "action" not in _forbidden_words()
