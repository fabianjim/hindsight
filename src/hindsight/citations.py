import re
from collections.abc import Iterable, Mapping

from pydantic import BaseModel

MIN_QUOTE_CHARS = 3
MAX_QUOTE_CHARS = 300


class Citation(BaseModel):
    entry_id: int
    quote: str


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", text.casefold()).split())


# keeps citations whose quote appears in the cited entry, ignoring case and punctuation
def verified(citations: Iterable[Citation], bodies: Mapping[int, str]) -> list[Citation]:
    padded = {entry_id: f" {normalize(body)} " for entry_id, body in bodies.items()}
    kept: list[Citation] = []
    for citation in citations:
        quote = normalize(citation.quote)
        body = padded.get(citation.entry_id)
        if (
            body is not None
            and MIN_QUOTE_CHARS <= len(quote)
            and len(citation.quote) <= MAX_QUOTE_CHARS
            # padded so a quote can't match half a word
            and f" {quote} " in body
            and citation not in kept
        ):
            kept.append(citation)
    return kept
