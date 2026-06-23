from typing import Any, Generic, Literal, TypeVar

from pydantic import BaseModel


OutputT = TypeVar("OutputT")


class DatagenClientConfig(BaseModel, Generic[OutputT]):
    model: Literal[
        "google/gemini-3.1-flash-lite",
        "google/gemini-3.5-flash",
        "google/gemini-3.5-pro",
        "google/gemini-3.5-pro-exp-06-25",
        "openai/gpt-5.4",
        "openai/gpt-5.4-mini",
        "openai/gpt-5.4-nano",
        "moonshotai/kimi-2.6",
    ]
    instructions: str
    output_format: Any | None = None
    prompt: str


class ChronologyItem(BaseModel):
    event: str
    details: str
    day: int



class Chronology(BaseModel):
    items: list[ChronologyItem]