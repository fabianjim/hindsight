from hindsight.citations import Citation, verified

BODIES = {
    1: "Bought NVDA after the earnings dip. Data-center demand looks durable; I'll hold for years.",
    2: "Sold AMD — felt like FOMO.",
}


def cite(entry_id: int, quote: str) -> Citation:
    return Citation(entry_id=entry_id, quote=quote)


def test_keeps_quotes_that_appear_in_the_cited_entry() -> None:
    citations = [
        cite(1, "Data-center demand looks durable"),
        cite(1, "data center demand looks durable, I'll hold"),
        cite(2, "Sold AMD - felt like FOMO"),
    ]

    assert verified(citations, BODIES) == citations


def test_drops_invented_misattributed_and_partial_word_quotes() -> None:
    citations = [
        cite(1, "margins will expand"),
        cite(2, "Data-center demand looks durable"),
        cite(3, "Bought NVDA"),
        cite(1, "ought NVD"),
        cite(1, "a"),
    ]

    assert verified(citations, BODIES) == []


def test_drops_duplicates_and_overlong_quotes() -> None:
    body = "x " * 400
    citations = [cite(1, "Bought NVDA"), cite(1, "Bought NVDA"), cite(9, body.strip())]

    assert verified(citations, {**BODIES, 9: body}) == [cite(1, "Bought NVDA")]
