import os
from typing import Generic, TypeVar, cast

from dotenv import load_dotenv
from pydantic_ai import Agent
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider

from datagen.types import DatagenClientConfig

load_dotenv()


OutputT = TypeVar("OutputT")


class DatasetGeneratorBase(Generic[OutputT]):
    def __init__(self, config: DatagenClientConfig[OutputT]):
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set. Add it to .env or export it before running."
            )

        openrouter_provider = OpenRouterProvider(
            api_key=api_key,
        )
        openrouter_model = OpenRouterModel(
            config.model,
            provider=openrouter_provider,
        )
        self.output_format = config.output_format or str

        self.agent = Agent(
            model=openrouter_model,
            output_type=self.output_format,
            instructions=config.instructions,
        )

    def generate(self, prompt: str) -> OutputT:
        response = self.agent.run_sync(prompt)

        return cast(OutputT, response.output)
