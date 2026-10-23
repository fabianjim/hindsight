import uuid
from dataclasses import dataclass

import openai
from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from hindsight.config import get_settings

# no sdk retries: each one would be an extra provider charge inside a single reservation
MAX_RETRIES = 0
TIMEOUT = openai.Timeout(30.0, connect=5.0)


@dataclass(frozen=True)
class UserContext:
    user_id: int
    generation_id: uuid.UUID


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    # never fall back to a stray OPENAI_API_KEY in the environment
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise RuntimeError("HINDSIGHT_OPENAI_API_KEY is not set")
    model = init_chat_model(
        settings.chat_model,
        api_key=settings.openai_api_key.get_secret_value(),
        max_retries=MAX_RETRIES,
        timeout=TIMEOUT,
        store=False,
    )
    assert isinstance(model, BaseChatModel)
    return model
