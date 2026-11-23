"""Text helpers shared by the catalog and the reports."""
from __future__ import annotations

import re
import unicodedata

ELLIPSIS = "…"


def normalize_query(text: str) -> str:
    """Case- and space-insensitive form used for searching ('STRASSE' matches 'Straße')."""
    return " ".join(str(text).casefold().split())


def slugify(text: str) -> str:
    """'Crème Brûlée 2-pack' -> 'creme-brulee-2-pack'. Accents and compatibility characters
    (ligatures such as 'ﬁ', full-width digits) are folded to ASCII; anything else is dropped."""
    ascii_text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", ascii_text.lower()).strip("-")


def truncate(text: str, limit: int) -> str:
    """Cut `text` to at most `limit` characters. A cut is marked with an ellipsis, which counts
    toward the limit. Text that already fits is returned unchanged."""
    if limit < 1:
        raise ValueError("limit must be at least 1")
    text = str(text)
    if len(text) <= limit:
        return text
    return text[: limit - 1] + ELLIPSIS
