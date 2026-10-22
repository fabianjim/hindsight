from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from hindsight.config import get_settings


@dataclass(frozen=True)
class UserContext:
    user_id: int


def get_chat_model() -> BaseChatModel:
    settings = get_settings()
    key = settings.openai_api_key
    model = init_chat_model(settings.chat_model, api_key=key.get_secret_value() if key else None)
    assert isinstance(model, BaseChatModel)
    return model
