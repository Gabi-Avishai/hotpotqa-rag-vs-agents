"""Text normalization ``norm_v1`` and mention tokenization ``tok_v1`` (PREREGISTRATION.md §2.2, §2.3).

``norm_v1`` is applied to corpus text and queries of the normalized BM25 candidates (C1,
C2) and to every alias and sentence of the title-mention graph. Displayed text and
citations always use the original text.
"""
from __future__ import annotations

import re
import unicodedata
from typing import List

_NON_DECOMPOSING = str.maketrans({
    "ø": "o", "Ø": "O", "ß": "ss", "æ": "ae", "Æ": "AE", "œ": "oe", "Œ": "OE",
    "đ": "d", "Đ": "D", "ł": "l", "Ł": "L", "ı": "i", "þ": "th", "Þ": "Th", "ð": "d", "Ð": "D",
})
_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'"})
_DASHES = str.maketrans({c: "-" for c in "–—‐‑‒−"})
_POSSESSIVE = re.compile(r"'s\b")
_ACRONYM = re.compile(r"\b(?:[A-Za-z]\.){2,}")
_WORD = re.compile(r"\w+")


def norm_v1(text: str) -> str:
    """The six preregistered steps, in order."""
    text = unicodedata.normalize("NFKD", text)                                      # 1
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = text.translate(_NON_DECOMPOSING)                                         # 2
    text = text.translate(_APOSTROPHES)                                             # 3
    text = _POSSESSIVE.sub("", text).replace("'", "")
    text = text.translate(_DASHES)                                                  # 4
    text = _ACRONYM.sub(lambda m: m.group(0).replace(".", ""), text)                # 5
    return " ".join(text.split())                                                   # 6


def tok_v1(text: str) -> List[str]:
    """Case-folded ``\\w+`` tokens of ``norm_v1(text)``."""
    return _WORD.findall(norm_v1(text).casefold())
