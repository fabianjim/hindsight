from scripts.export_openapi import CONTRACT, render


def test_contract_is_up_to_date() -> None:
    assert CONTRACT.read_text() == render(), "run: uv run python -m scripts.export_openapi"
