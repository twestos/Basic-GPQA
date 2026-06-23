import json
import sys
from argparse import ArgumentParser
from pathlib import Path

if __name__ == "__main__" and not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datagen.client import DatasetGeneratorBase
from datagen.types import Chronology, DatagenClientConfig
from datagen.prompts import build_chronology_prompt


MODEL = "openai/gpt-5.4"
BUSINESS_CONTEXT = """
A highly adversarial divorce legal matter between two parties, Alice and Bob.
"""


def generate_structured_chronology() -> Chronology:
    config: DatagenClientConfig[Chronology] = DatagenClientConfig(
        model=MODEL,
        instructions="",
        output_format=Chronology,
        prompt=build_chronology_prompt(business_context=BUSINESS_CONTEXT),
    )
    agent = DatasetGeneratorBase[Chronology](config)
    return agent.generate(prompt=config.prompt)


def generate_text_chronology() -> str:
    config: DatagenClientConfig[str] = DatagenClientConfig(
        model=MODEL,
        instructions="Return the chronology as plain text.",
        prompt=f"""
Create a fictional but realistic chronology of events for this business context:

{BUSINESS_CONTEXT.strip()}

Return the chronology as plain text with dated entries. Do not return JSON.
""".strip(),
    )
    agent = DatasetGeneratorBase[str](config)
    return agent.generate(prompt=config.prompt)


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument(
        "--output",
        choices=("structured", "text"),
        default="structured",
        help="Generate a typed Chronology object or plain text.",
    )
    args = parser.parse_args()

    if args.output == "text":
        print(generate_text_chronology())
        return

    chronology = generate_structured_chronology()

    print(f"Generated {len(chronology.items)} chronology items.")
    print(json.dumps(chronology.model_dump(), indent=4))


if __name__ == "__main__":
    main()