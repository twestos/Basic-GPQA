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


@dataclass
class LegalChronoTask:
    task_id: str
    sector: str
    occupation: str
    domain: str
    matter_id: str
    matter_title: str
    category: str
    prompt: str
    reference_files: list[str]
    reference_file_urls: list[str]
    reference_file_hf_uris: list[str]
    deliverable_text: str
    deliverable_files: list[str]
    scoring_rubric: str
    source_event_ids: list[str]
    gold_answer: str


@dataclass
class ReferenceFileContent:
    path: str
    content: str
    truncated: bool


@dataclass
class LegalChronoScore:
    valid_score: bool
    score: int | None
    normalized_score: float
    passed: bool
    rationale: str
    grader_response: str


@dataclass
class LegalChronoEvaluationResult:
    task_id: str
    domain: str
    matter_id: str
    category: str
    occupation: str
    prompt: str
    reference_files: list[str]
    reference_file_hf_uris: list[str]
    reference_files_truncated: list[str]
    gold_answer: str
    scoring_rubric: str
    model: str
    grader_model: str
    model_response: str
    grader_response: str
    valid_score: bool
    score: int | None
    normalized_score: float
    passed: bool
    grader_rationale: str
    answer_ttft: float
    answer_output_speed: float
    answer_input_tokens_client: int
    answer_input_tokens_model: int
    answer_output_tokens_client: int
    answer_output_tokens_model: int
    answer_cost: float
    grader_ttft: float
    grader_output_speed: float
    grader_input_tokens_client: int
    grader_input_tokens_model: int
    grader_output_tokens_client: int
    grader_output_tokens_model: int
    grader_cost: float
    total_cost: float
