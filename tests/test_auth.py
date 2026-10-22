from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hindsight.auth import require_service_token
from hindsight.config import get_settings
from tests.conftest import SERVICE_TOKEN

app = FastAPI()


@app.get("/protected")
def protected(_: Annotated[None, Depends(require_service_token)]) -> dict[str, str]:
    return {"status": "ok"}


client = TestClient(app)


def test_accepts_the_service_token() -> None:
    response = client.get("/protected", headers={"Authorization": f"Bearer {SERVICE_TOKEN}"})

    assert response.status_code == 200


@pytest.mark.parametrize(
    "headers",
    [{}, {"Authorization": "Bearer wrong-token"}, {"Authorization": f"Basic {SERVICE_TOKEN}"}],
)
def test_rejects_missing_or_wrong_token(headers: dict[str, str]) -> None:
    response = client.get("/protected", headers=headers)

    assert response.status_code == 401


def test_rejects_short_service_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HINDSIGHT_SERVICE_TOKEN", "too-short")
    get_settings.cache_clear()

    with pytest.raises(ValidationError):
        get_settings()
