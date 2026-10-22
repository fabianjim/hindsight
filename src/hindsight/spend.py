import json
from collections.abc import Awaitable, Callable
from typing import Any

from langchain.agents.middleware import AgentMiddleware, AgentState, ModelRequest, ModelResponse
from langchain.agents.structured_output import AutoStrategy, ProviderStrategy, ToolStrategy
from langchain_core.messages import AIMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from hindsight.budget import BudgetError, BudgetLedger
from hindsight.llm import UserContext
from hindsight.pricing import price_of

# headroom for provider-side formatting the request bytes don't show
REQUEST_OVERHEAD_TOKENS = 2048


# reserves each model call's worst case, then settles the real cost.
# keep it last in the middleware list so it sits closest to the model and every retry pays
class BudgetMiddleware(AgentMiddleware[AgentState[Any], UserContext, Any]):
    def __init__(self, ledger: BudgetLedger, model: str, step: str, max_output_tokens: int) -> None:
        super().__init__()
        if max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        self._ledger = ledger
        self._price = price_of(model)
        self._step = step
        self._max_output_tokens = max_output_tokens

    async def awrap_model_call(
        self,
        request: ModelRequest[UserContext],
        handler: Callable[[ModelRequest[UserContext]], Awaitable[ModelResponse[Any]]],
    ) -> ModelResponse[Any]:
        allowance = self._price.cost(input_token_bound(request), self._max_output_tokens)
        n = 1 + sum(isinstance(m, AIMessage) for m in request.messages)
        call = await self._ledger.reserve_call(
            request.runtime.context.generation_id, f"{self._step}-{n}", allowance
        )
        settings = {**request.model_settings, "max_tokens": self._max_output_tokens}
        # a provider error leaves the call unsettled, so the job keeps the full reservation
        response = await handler(request.override(model_settings=settings))

        usage = [m.usage_metadata for m in response.result if isinstance(m, AIMessage)]
        if not usage or None in usage:
            raise BudgetError("model response has no usage")
        actual = self._price.cost(
            sum(u["input_tokens"] for u in usage if u),
            sum(u["output_tokens"] for u in usage if u),
        )
        await self._ledger.settle_call(call, actual)
        if actual > allowance:
            raise BudgetError("model call cost more than its reservation")
        return response


def input_token_bound(request: ModelRequest[UserContext]) -> int:
    # a token is at least one utf-8 byte, so bytes never undercount
    messages = (
        [request.system_message, *request.messages] if request.system_message else request.messages
    )
    parts: list[Any] = [m.model_dump(mode="json") for m in messages]
    parts += [convert_to_openai_tool(t) for t in request.tools]
    parts += response_schemas(request.response_format)
    return len(json.dumps(parts, ensure_ascii=False).encode()) + REQUEST_OVERHEAD_TOKENS


def response_schemas(response_format: object) -> list[dict[str, Any]]:
    match response_format:
        case None:
            return []
        case ToolStrategy():
            return [spec.json_schema for spec in response_format.schema_specs]
        case ProviderStrategy():
            return [response_format.schema_spec.json_schema]
        case AutoStrategy():
            return [spec.json_schema for spec in ToolStrategy(response_format.schema).schema_specs]
        case _:
            raise ValueError(f"can't bound the cost of response_format {response_format!r}")
