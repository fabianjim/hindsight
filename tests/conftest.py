from collections.abc import Iterator

import pytest

from hindsight.config import get_settings

SERVICE_TOKEN = "test-token-" + "x" * 32


@pytest.fixture(autouse=True)
def settings_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("HINDSIGHT_SERVICE_TOKEN", SERVICE_TOKEN)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
