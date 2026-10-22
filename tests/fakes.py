from collections.abc import Callable, Sequence
from typing import Any, Self

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.tools import BaseTool


class FakeToolModel(GenericFakeChatModel):
    def bind_tools(
        self, tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], **_: Any
    ) -> Self:
        return self
