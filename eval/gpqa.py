import asyncio
from eval.client import Client
from eval.types import EvaluationResult, GDPQAQuestion, ModelMessage
from eval.utils import extract_answer, load_gdpqa_dataset


class GPQA:
    def __init__(self, model: str, max_concurrency: int = 8):
        self.client = Client(model)
        self.dataset: list[GDPQAQuestion] = []
        self.max_concurrency = max_concurrency

    def load_dataset(self, variant: str):
        self.dataset = load_gdpqa_dataset(variant)

    async def run(self) -> list[EvaluationResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_with_limit(question: GDPQAQuestion) -> EvaluationResult:
            async with semaphore:
                return await self._evaluate_question(question)

        return await asyncio.gather(*[
            evaluate_with_limit(question)
            for question in self.dataset
        ])

    async def _evaluate_question(self, question: GDPQAQuestion) -> EvaluationResult:
        response = await self.client.ask([
            ModelMessage(role="user", content=question.prompt)
        ])

        supplied_answer = extract_answer(response.content)
        correctly_answered = supplied_answer == question.correct_letter if supplied_answer else False
        usage = response.usage

        return EvaluationResult(
            question_id=question.id,
            question=question.question,
            ttft=response.ttft,
            output_speed=response.output_speed,
            input_tokens_client=response.input_tokens_client,
            input_tokens_model=usage.input_tokens if usage else 0,
            output_tokens_client=response.output_tokens_client,
            output_tokens_model=usage.output_tokens if usage else 0,
            valid_answer=supplied_answer is not None,
            supplied_answer=supplied_answer,
            correct_answer=question.correct_answer,
            correctly_answered=correctly_answered,
            cost=usage.cost if usage else 0.0,
        )