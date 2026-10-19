"""A cheap language hint (Uzbek / Russian / English) for the prompt. Claude makes the final call."""

from __future__ import annotations

import re
from typing import Optional

APOSTROPHES = "'ʻ’‘`ʼ"
_APOS_RE = re.compile(f"[{re.escape(APOSTROPHES)}]")

UZ_CYRILLIC_LETTERS = set("ўқғҳЎҚҒҲ")
UZ_CYRILLIC_WORDS = {
    "салом", "ассалому", "ассалом", "алайкум", "алейкум", "рахмат", "раҳмат", "яхши", "яхшими",
    "яхшимисиз", "яхшимисан", "нима", "нимага", "қалесан", "калесан", "қалайсан", "қалайсиз",
    "йўқ", "йук", "бугун", "эртага", "керак", "кейин", "хозир", "ҳозир", "хоп", "бўпти", "бопти",
    "мен", "сен", "сиз", "биз", "қаерда", "каерда", "қаердасан", "қачон", "качон", "нега", "ака", "опа",
}

UZ_LATIN_WORDS = {
    "salom", "assalomu", "assalom", "alaykum", "rahmat", "raxmat", "yaxshi", "yaxshimi", "nima",
    "nimaga", "qalesan", "qalaysan", "qalaysiz", "qayerda", "qayerdasan", "qayerdasiz", "bugun",
    "ertaga", "kerak", "xop", "xo'p", "bopti", "boldi", "zor", "hozir", "keyin", "men", "sen", "siz",
    "biz", "yoq", "qachon", "nega", "qanday", "aka", "opa", "uka", "kechqurun", "ertalab", "endi",
    "yana", "bilan", "uchun", "lekin",
}

UZ_LATIN_SUFFIXES = (
    "misan", "misiz", "mizmi", "yapsan", "yapman", "yapti", "yapmiz", "asizmi", "ganman", "dingmi",
)

_LATIN_WORD_RE = re.compile(r"[a-z']+")
_POSSESSIVE_RE = re.compile(r"[og]'s$")
_ENGLISH_APOSTROPHE_WORDS = {"o'clock"}


def _cyrillic_language(lowered: str) -> str:
    if any(ch in UZ_CYRILLIC_LETTERS for ch in lowered):
        return "uz"
    words = set(re.findall(r"[Ѐ-ӿ]+", lowered))
    return "uz" if words & UZ_CYRILLIC_WORDS else "ru"


def _latin_language(lowered: str) -> str:
    normalized = _APOS_RE.sub("'", lowered)
    words = _LATIN_WORD_RE.findall(normalized)
    for word in words:
        if word in _ENGLISH_APOSTROPHE_WORDS or _POSSESSIVE_RE.search(word):
            continue
        if "o'" in word or "g'" in word:
            return "uz"
    for word in words:
        bare = word.replace("'", "")
        if not bare:
            continue
        if bare in UZ_LATIN_WORDS or word in UZ_LATIN_WORDS:
            return "uz"
        if any(bare.endswith(suffix) for suffix in UZ_LATIN_SUFFIXES):
            return "uz"
    return "en"


def detect_language(text: Optional[str]) -> Optional[str]:
    """Return ``"uz"``, ``"ru"``, ``"en"`` or ``None`` when the text has no letters."""
    if not text:
        return None
    lowered = text.lower()
    cyrillic = sum(1 for ch in lowered if "Ѐ" <= ch <= "ӿ")
    latin = sum(1 for ch in lowered if "a" <= ch <= "z")
    if cyrillic == 0 and latin == 0:
        return None
    if cyrillic > latin:
        return _cyrillic_language(lowered)
    return _latin_language(lowered)
