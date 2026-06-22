import os
import httpx
import time
import json
import asyncio
from dotenv import load_dotenv
from eval.types import GDPQAQuestion, ResponseChunk, ModelUsageMetrics, EvaluationResult
from eval.utils import load_gdpqa_dataset, count_tokens, extract_answer

load_dotenv()


class GPQA:
    def __init__(self, model: str):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or export it before running.")

        self.model = model
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        self.dataset: list[GDPQAQuestion] = []

    def load_dataset(self, variant: str):
        self.dataset = load_gdpqa_dataset(variant)

    async def run(self):
        async with httpx.AsyncClient(timeout=None) as client:
            return await self._run(client)

    async def _run(self, client: httpx.AsyncClient):
        results = await asyncio.gather(*[self._evaluate_question(client, question) for question in self.dataset])
        return results

    async def _evaluate_question(self, client: httpx.AsyncClient, question: GDPQAQuestion):
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": question.prompt
                }
            ],
            "stream": True
        }
        result: EvaluationResult | None = None
        async with client.stream("POST", self.base_url, headers=self.headers, json=payload) as r:
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as error:
                if error.response.status_code == 401:
                    raise RuntimeError("OpenRouter rejected the request. Check that OPENROUTER_API_KEY is valid.") from None
                raise
            buffer = ""
            client_token_count = 0
            start_time = time.perf_counter()
            first_chunk_time = None
            chunks: list[ResponseChunk] = []
            model_usage_metrics: ModelUsageMetrics | None = None
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                line = line.removeprefix("data: ")
                if line == "[DONE]":
                    continue
                data = json.loads(line)

                content = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    buffer += content
                    if not first_chunk_time:
                        first_chunk_time = time.perf_counter()

                    chunk_token_count = count_tokens(content)
                    client_token_count += chunk_token_count
                    chunks.append(ResponseChunk(
                        content=content,
                        token_count=chunk_token_count,
                        received_at=time.perf_counter()
                    ))

                usage = data.get("usage", {})
                if usage:
                    model_usage_metrics = ModelUsageMetrics(
                        input_tokens=usage.get("prompt_tokens", 0),
                        output_tokens=usage.get("completion_tokens", 0),
                        cost=usage.get("cost", 0.0)
                    )
            ttft = first_chunk_time - start_time
            output_speed = (client_token_count - chunks[0].token_count) / (chunks[-1].received_at - chunks[0].received_at)
            supplied_answer = extract_answer(buffer)
            correctly_answered = supplied_answer == question.correct_letter if supplied_answer else False
            valid_answer = supplied_answer is not None
            cost = model_usage_metrics.cost if model_usage_metrics else 0.0
            result = EvaluationResult(
                question_id=question.id,
                question=question.question,
                ttft=ttft,
                output_speed=output_speed,
                input_tokens_client=client_token_count,
                input_tokens_model=model_usage_metrics.input_tokens if model_usage_metrics else 0,
                output_tokens_client=client_token_count,
                output_tokens_model=model_usage_metrics.output_tokens if model_usage_metrics else 0,
                valid_answer=valid_answer,
                supplied_answer=supplied_answer,
                correct_answer=question.correct_answer,
                correctly_answered=correctly_answered,
                cost=cost
            )
        return result








        