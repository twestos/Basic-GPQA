import asyncio
import os
import httpx
import json
import time
from dataclasses import asdict
from dotenv import load_dotenv


from eval.types import ModelMessage, ModelResponse, ModelUsageMetrics, ResponseChunk
from eval.utils import count_tokens


load_dotenv()


class Client:
    def __init__(self, model: str, max_retries: int = 3):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set. Add it to .env or export it before running.")

        self.model = model
        self.max_retries = max_retries
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    async def ask(self, messages: list[ModelMessage]) -> ModelResponse:
        for attempt in range(self.max_retries + 1):
            try:
                return await self._ask_once(messages)
            except (httpx.HTTPStatusError, httpx.TransportError, RuntimeError) as error:
                if attempt >= self.max_retries or not self._should_retry(error):
                    raise
                await asyncio.sleep(2 ** attempt)

        raise RuntimeError("OpenRouter request failed after retries.")

    def _should_retry(self, error: Exception) -> bool:
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            return status_code in {408, 409, 425, 429} or status_code >= 500
        if isinstance(error, httpx.TransportError):
            return True

        message = str(error).lower()
        transient_error_markers = (
            "timeout",
            "temporarily unavailable",
            "upstream",
            "overloaded",
            "rate limit",
            "sse stream",
            "json error",
        )
        return any(marker in message for marker in transient_error_markers)

    async def _ask_once(self, messages: list[ModelMessage]) -> ModelResponse:
        payload = {
            "model": self.model,
            "messages": [asdict(message) for message in messages],
            "stream": True
        }
        buffer = ""
        start_time = time.perf_counter()
        first_chunk_time = None
        chunks: list[ResponseChunk] = []
        client_token_input = sum(count_tokens(message.content) for message in messages)
        client_token_output = 0
        model_usage_metrics: ModelUsageMetrics | None = None
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream("POST", self.base_url, headers=self.headers, json=payload) as r:
                try:
                    r.raise_for_status()
                except httpx.HTTPStatusError as error:
                    if error.response.status_code == 401:
                        raise RuntimeError("OpenRouter rejected the request. Check that OPENROUTER_API_KEY is valid.") from None
                    raise
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line = line.removeprefix("data: ")
                    if line == "[DONE]":
                        break
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise RuntimeError("OpenRouter stream error: malformed JSON in SSE stream") from error
                    error = data.get("error")
                    if error:
                        message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                        raise RuntimeError(f"OpenRouter stream error: {message}")

                    usage = data.get("usage", {})
                    if usage:
                        model_usage_metrics = ModelUsageMetrics(
                            input_tokens=usage.get("prompt_tokens", 0),
                            output_tokens=usage.get("completion_tokens", 0),
                            cost=usage.get("cost", 0.0)
                        )
                    choices = data.get("choices") or []
                    if not choices:
                        continue
                    choice = choices[0]
                    content = (
                        choice.get("delta", {}).get("content")
                        or choice.get("message", {}).get("content", "")
                    )
                    if content:
                        token_count = count_tokens(content)
                        client_token_output += token_count
                        buffer += content
                        if not first_chunk_time:
                            first_chunk_time = time.perf_counter()
                        chunks.append(ResponseChunk(
                            content=content,
                            token_count=token_count,
                            received_at=time.perf_counter()
                        ))
                ttft = first_chunk_time - start_time if first_chunk_time else 0.0
                output_speed = 0.0
                if len(chunks) > 1:
                    output_duration = chunks[-1].received_at - chunks[0].received_at
                    if output_duration > 0:
                        output_speed = (client_token_output - chunks[0].token_count) / output_duration
        return ModelResponse(
            content=buffer,
            ttft=ttft,
            output_speed=output_speed,
            input_tokens_client=client_token_input,
            output_tokens_client=client_token_output,
            usage=model_usage_metrics,
        )