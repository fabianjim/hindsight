from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from pydantic import Field


class FakeToolModel(GenericFakeChatModel):
    # kwargs each call received, e.g. max_tokens bound by middleware
    calls: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    prompts: list[list[BaseMessage]] = Field(default_factory=list[list[BaseMessage]])

    def bind_tools(
        self, tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool], **kwargs: Any
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self.bind(**kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self.calls.append(kwargs)
        self.prompts.append(messages)
        return super()._generate(messages, stop, run_manager, **kwargs)
