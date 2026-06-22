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
    supplied_answer: str
    correct_answer: str
    correctly_answered: bool
    cost: float
