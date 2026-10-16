import inspect
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Type

from pydantic import BaseModel, Field


class RetrieveKnowledgeInput(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=4, ge=1, le=20)
    document_ids: list[str] | None = Field(default=None, min_length=1)


class SearchHistoryInput(BaseModel):
    session_id: str = Field(min_length=1)
    limit: int = Field(default=10, ge=1, le=50)


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: Type[BaseModel]
    handler: Callable[[BaseModel], Any] | Callable[[BaseModel], Awaitable[Any]]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool already registered: {tool.name}")
        self._tools[tool.name] = tool

    def describe(self) -> list[dict[str, str]]:
        return [{"name": tool.name, "description": tool.description} for tool in self._tools.values()]

    async def execute(self, name: str, payload: dict[str, Any]) -> Any:
        tool = self._tools.get(name)
        if not tool:
            raise KeyError(f"unknown tool: {name}")
        arguments = tool.input_model.model_validate(payload)
        result = tool.handler(arguments)
        if inspect.isawaitable(result):
            return await result
        return result
