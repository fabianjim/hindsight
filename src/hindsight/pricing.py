from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_micros_per_million: int
    output_micros_per_million: int

    def cost(self, input_tokens: int, output_tokens: int) -> int:
        total = input_tokens * self.input_micros_per_million
        total += output_tokens * self.output_micros_per_million
        return -(-total // 1_000_000)


# pinned snapshots only, so a price never changes under us; no cache or batch discounts
PRICES = {
    "openai:gpt-4.1-mini-2025-04-14": Price(400_000, 1_600_000),
}


def price_of(model: str) -> Price:
    try:
        return PRICES[model]
    except KeyError:
        raise ValueError(f"no price for {model}; add its pinned snapshot to PRICES") from None
