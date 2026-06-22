from dataclasses import dataclass
from typing import Literal

@dataclass
class GDPQAQuestion:
    id: str
    category: str
    subcategory: str
    question: str
    correct_answer: str
    incorrect_answers: list[str]
    correct_letter: Literal['A', 'B', 'C', 'D']
    prompt: str


@dataclass
class ResponseChunk:
    content: str
    token_count: int
    received_at: float

@dataclass
class ModelUsageMetrics:
    input_tokens: int
    output_tokens: int
    cost: float


@dataclass
class ModelResponse:
    content: str
    ttft: float
    output_speed: float
    input_tokens_client: int
    output_tokens_client: int
    usage: ModelUsageMetrics | None


@dataclass
class EvaluationResult:
    question_id: str
    question: str
    ttft: float
    output_speed: float
    input_tokens_client: int
    input_tokens_model: int
    output_tokens_client: int
    output_tokens_model: int
    valid_answer: bool
    supplied_answer: str | None
    correct_answer: str
    correctly_answered: bool
    cost: float


@dataclass
class ModelMessage:
    role: Literal["user", "assistant", "system"]
    content: str
