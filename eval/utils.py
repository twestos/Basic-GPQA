import tiktoken
import pandas
import random
import hashlib
import re
from typing import Literal
from eval.types import GDPQAQuestion, ModelMessage


ENCODER = tiktoken.get_encoding("o200k_base")


MULTI_CHOICE_QUESTION_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A').

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def load_gdpqa_dataset(variant: str):

    ## Load the dataset from the URL
    df = pandas.read_csv(
        f"https://openaipublic.blob.core.windows.net/simple-evals/gpqa_{variant}.csv"
    )
    rows = [row.to_dict() for _, row in df.iterrows()]

    ## Process the dataset into a list of dicts
    questions: list[GDPQAQuestion] = []
    for row in rows:
        letters = ['A', 'B', 'C', 'D']
        random.shuffle(letters)
        choices = [
            row["Correct Answer"],
            row["Incorrect Answer 1"],
            row["Incorrect Answer 2"],
            row["Incorrect Answer 3"],
        ]
        randomised_choices = zip(letters, choices)
        prompt = MULTI_CHOICE_QUESTION_TEMPLATE.format(
            Question=row["Question"],
            **{letter: choice for letter, choice in randomised_choices},
        )

        questions.append(GDPQAQuestion(
            id=hashlib.sha256(row["Question"].encode("utf-8")).hexdigest(),
            category=row["High-level domain"],
            subcategory=row["Subdomain"],
            question=row["Question"],
            correct_answer=row["Correct Answer"],
            incorrect_answers=(
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ),
            correct_letter=letters[0],
            prompt=prompt,
        ))
    return questions



MULTI_CHOICE_REGEX_PATTERNS = [
    # Primary Pattern
    r"(?i)[\*_]{0,2}Answer[\*_]{0,2}\s*:[\s\*_]{0,2}\s*([A-Z])(?![a-zA-Z0-9])",

    # Fallback Patterns
    r"\\boxed\{[^}]*([A-Z])[^}]*\}",
    r"answer is ([a-zA-Z])",
    r"answer is \(([a-zA-Z])",
    r"([A-Z])\)\s*[^A-Z]*",
    r"([A-Z])\s+is\s+the\s+correct\s+answer",
    r"([A-Z])\s*$",
    r"([A-Z])\s*\.",
    r"([A-Z])\s*[^\w]"
]



def extract_answer(text: str) -> Literal['A', 'B', 'C', 'D', None]:
    for pattern in MULTI_CHOICE_REGEX_PATTERNS:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return None

