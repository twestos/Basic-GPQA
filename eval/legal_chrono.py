from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Iterable

import numpy as np
from huggingface_hub import hf_hub_download

from eval.client import Client
from eval.types import (
    LegalChronoEvaluationResult,
    LegalChronoScore,
    LegalChronoTask,
    ModelMessage,
    ModelResponse,
    ReferenceFileContent,
)


DEFAULT_REPO_ID = "twestoss/legal-matter-chrono-bench"
DEFAULT_MODEL = "qwen/qwen3.7-plus"
DEFAULT_GRADER_MODEL = "openai/gpt-5.4"


ANSWER_SYSTEM_PROMPT = """
You are evaluating a synthetic legal matter file. Answer the user's question
using only the provided reference files. Your response should be a plain English
answer, not JSON. Be precise about chronology, contradictions, current matter
state, and source authority. Cite filenames when useful.
""".strip()


GRADER_SYSTEM_PROMPT = """
You are a careful legal chronology benchmark grader. Score the candidate answer
against the gold answer and task rubric. You are grading semantic correctness,
not prose style. Reward answers that correctly track chronology, contradictions,
source authority, live issues, and current matter state. Penalize unsupported
claims, stale facts presented as current, missed material caveats, and wrong
dates or parties.

Return plain text in this exact style:
Score: <integer 0-5>
Rationale: <brief explanation>
""".strip()


def load_legal_chrono_tasks(
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    split_path: str = "data/train.jsonl",
    limit: int | None = None,
    offset: int = 0,
    domain: str | None = None,
    category: str | None = None,
    matter_id: str | None = None,
) -> list[LegalChronoTask]:
    dataset_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=split_path,
        revision=revision,
    )
    tasks: list[LegalChronoTask] = []
    with Path(dataset_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            task = LegalChronoTask(
                task_id=row["task_id"],
                sector=row.get("sector", ""),
                occupation=row.get("occupation", ""),
                domain=row.get("domain", ""),
                matter_id=row.get("matter_id", ""),
                matter_title=row.get("matter_title", ""),
                category=row.get("category", ""),
                prompt=row["prompt"],
                reference_files=list(row.get("reference_files", [])),
                reference_file_urls=list(row.get("reference_file_urls", [])),
                reference_file_hf_uris=list(row.get("reference_file_hf_uris", [])),
                deliverable_text=row.get("deliverable_text", ""),
                deliverable_files=list(row.get("deliverable_files", [])),
                scoring_rubric=row.get("scoring_rubric", ""),
                source_event_ids=list(row.get("source_event_ids", [])),
                gold_answer=row.get("gold_answer") or row.get("deliverable_text", ""),
            )
            if domain and task.domain != domain:
                continue
            if category and task.category != category:
                continue
            if matter_id and task.matter_id != matter_id:
                continue
            tasks.append(task)

    sliced = tasks[offset:]
    if limit is not None:
        sliced = sliced[:limit]
    return sliced


def load_reference_files(
    task: LegalChronoTask,
    repo_id: str = DEFAULT_REPO_ID,
    revision: str = "main",
    max_reference_chars: int = 120_000,
    max_file_chars: int = 30_000,
) -> list[ReferenceFileContent]:
    loaded: list[ReferenceFileContent] = []
    remaining_chars = max_reference_chars
    for reference_file in task.reference_files:
        if remaining_chars <= 0:
            loaded.append(
                ReferenceFileContent(
                    path=reference_file,
                    content="[Reference omitted: max total reference characters reached.]",
                    truncated=True,
                )
            )
            continue

        file_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=reference_file,
            revision=revision,
        )
        raw_content = Path(file_path).read_text(encoding="utf-8", errors="replace")
        char_budget = min(max_file_chars, remaining_chars)
        truncated = len(raw_content) > char_budget
        content = raw_content[:char_budget]
        if truncated:
            content += "\n\n[Truncated for evaluation prompt length.]"
        loaded.append(
            ReferenceFileContent(
                path=reference_file,
                content=content,
                truncated=truncated,
            )
        )
        remaining_chars -= len(content)
    return loaded


def build_answer_prompt(
    task: LegalChronoTask,
    references: list[ReferenceFileContent],
) -> str:
    reference_block = _format_reference_block(references)
    return f"""
Task ID: {task.task_id}
Domain: {task.domain}
Matter: {task.matter_title} ({task.matter_id})
Question category: {task.category}

Question:
{task.prompt}

Reference files:
{reference_block}

Answer the question in plain English. Use only the provided reference files.
""".strip()


def build_grader_prompt(
    task: LegalChronoTask,
    model_response: str,
    references: list[ReferenceFileContent],
    include_references: bool = False,
) -> str:
    reference_section = ""
    if include_references:
        reference_section = f"""

Reference files:
{_format_reference_block(references)}
""".rstrip()

    return f"""
Task ID: {task.task_id}
Domain: {task.domain}
Matter: {task.matter_title} ({task.matter_id})
Question category: {task.category}

Question:
{task.prompt}

Gold answer:
{task.gold_answer}

Task rubric:
{task.scoring_rubric}

Candidate answer:
{model_response}
{reference_section}

Score the candidate answer from 0 to 5:
- 5: fully correct; captures all material points, chronology, caveats, and current-state implications.
- 4: mostly correct; minor omissions or imprecision, no material contradiction.
- 3: partially correct; captures the broad issue but misses important evidence, timing, or caveats.
- 2: weak; some relevant facts, but substantial omissions or confused chronology.
- 1: minimally relevant; mostly incorrect or unsupported.
- 0: irrelevant, empty, or fundamentally wrong.

Return only:
Score: <integer 0-5>
Rationale: <brief explanation>
""".strip()


def parse_grader_score(grader_response: str) -> LegalChronoScore:
    score = _extract_score(grader_response)
    valid_score = score is not None
    normalized_score = (score / 5) if score is not None else 0.0
    return LegalChronoScore(
        valid_score=valid_score,
        score=score,
        normalized_score=normalized_score,
        passed=bool(score is not None and score >= 4),
        rationale=_extract_rationale(grader_response),
        grader_response=grader_response,
    )


class LegalChronoEval:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        grader_model: str = DEFAULT_GRADER_MODEL,
        repo_id: str = DEFAULT_REPO_ID,
        revision: str = "main",
        max_concurrency: int = 4,
        max_reference_chars: int = 120_000,
        max_file_chars: int = 30_000,
        include_references_in_grader: bool = False,
    ):
        self.model = model
        self.grader_model = grader_model
        self.repo_id = repo_id
        self.revision = revision
        self.max_concurrency = max_concurrency
        self.max_reference_chars = max_reference_chars
        self.max_file_chars = max_file_chars
        self.include_references_in_grader = include_references_in_grader
        self.answer_client = Client(model)
        self.grader_client = Client(grader_model)
        self.dataset: list[LegalChronoTask] = []

    def load_dataset(
        self,
        limit: int | None = None,
        offset: int = 0,
        domain: str | None = None,
        category: str | None = None,
        matter_id: str | None = None,
    ) -> None:
        self.dataset = load_legal_chrono_tasks(
            repo_id=self.repo_id,
            revision=self.revision,
            limit=limit,
            offset=offset,
            domain=domain,
            category=category,
            matter_id=matter_id,
        )

    async def run(self) -> list[LegalChronoEvaluationResult]:
        semaphore = asyncio.Semaphore(self.max_concurrency)

        async def evaluate_with_limit(task: LegalChronoTask) -> LegalChronoEvaluationResult:
            async with semaphore:
                return await self._evaluate_task(task)

        return await asyncio.gather(*[evaluate_with_limit(task) for task in self.dataset])

    async def _evaluate_task(self, task: LegalChronoTask) -> LegalChronoEvaluationResult:
        references = await asyncio.to_thread(
            load_reference_files,
            task,
            self.repo_id,
            self.revision,
            self.max_reference_chars,
            self.max_file_chars,
        )
        answer_prompt = build_answer_prompt(task, references)
        answer_response = await self.answer_client.ask(
            [
                ModelMessage(role="system", content=ANSWER_SYSTEM_PROMPT),
                ModelMessage(role="user", content=answer_prompt),
            ]
        )
        grader_prompt = build_grader_prompt(
            task,
            answer_response.content,
            references,
            include_references=self.include_references_in_grader,
        )
        grader_response = await self.grader_client.ask(
            [
                ModelMessage(role="system", content=GRADER_SYSTEM_PROMPT),
                ModelMessage(role="user", content=grader_prompt),
            ]
        )
        score = parse_grader_score(grader_response.content)
        return _build_result(
            task=task,
            references=references,
            model=self.model,
            grader_model=self.grader_model,
            answer_response=answer_response,
            grader_response=grader_response,
            score=score,
        )


def write_results(
    results: list[LegalChronoEvaluationResult],
    output_dir: Path,
    run_name: str,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / f"{run_name}.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(asdict(result), sort_keys=True) + "\n")

    summary = summarize_results(results)
    summary_path = output_dir / f"{run_name}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with (output_dir / "legal_chrono_summary.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(summary, sort_keys=True) + "\n")
    return results_path, summary_path


def summarize_results(results: list[LegalChronoEvaluationResult]) -> dict:
    scores = [result.normalized_score for result in results]
    raw_scores = [result.score for result in results if result.score is not None]
    answer_ttfts = [result.answer_ttft for result in results]
    grader_ttfts = [result.grader_ttft for result in results]
    answer_speeds = [result.answer_output_speed for result in results]
    grader_speeds = [result.grader_output_speed for result in results]
    total_cost = sum(result.total_cost for result in results)
    return {
        "timestamp": datetime.now().isoformat(),
        "task_count": len(results),
        "valid_score_count": sum(result.valid_score for result in results),
        "pass_at_4": mean([result.passed for result in results]) if results else 0.0,
        "average_score": mean(raw_scores) if raw_scores else 0.0,
        "average_normalized_score": mean(scores) if scores else 0.0,
        "total_cost": total_cost,
        "cost_per_task": total_cost / len(results) if results else 0.0,
        "answer_ttft": _percentiles(answer_ttfts),
        "grader_ttft": _percentiles(grader_ttfts),
        "answer_output_speed": _percentiles(answer_speeds),
        "grader_output_speed": _percentiles(grader_speeds),
        "answer_input_tokens_client": sum(result.answer_input_tokens_client for result in results),
        "answer_input_tokens_model": sum(result.answer_input_tokens_model for result in results),
        "answer_output_tokens_client": sum(result.answer_output_tokens_client for result in results),
        "answer_output_tokens_model": sum(result.answer_output_tokens_model for result in results),
        "grader_input_tokens_client": sum(result.grader_input_tokens_client for result in results),
        "grader_input_tokens_model": sum(result.grader_input_tokens_model for result in results),
        "grader_output_tokens_client": sum(result.grader_output_tokens_client for result in results),
        "grader_output_tokens_model": sum(result.grader_output_tokens_model for result in results),
        "by_domain": _group_summary(results, "domain"),
        "by_category": _group_summary(results, "category"),
        "model": results[0].model if results else "",
        "grader_model": results[0].grader_model if results else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the legal matter chronology benchmark.")
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--revision", default="main")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--grader-model", default=DEFAULT_GRADER_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--domain", default=None)
    parser.add_argument("--category", default=None)
    parser.add_argument("--matter-id", default=None)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-reference-chars", type=int, default=120_000)
    parser.add_argument("--max-file-chars", type=int, default=30_000)
    parser.add_argument("--include-references-in-grader", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load dataset and print the first prompt without making model calls.",
    )
    args = parser.parse_args()

    benchmark = LegalChronoEval(
        model=args.model,
        grader_model=args.grader_model,
        repo_id=args.repo_id,
        revision=args.revision,
        max_concurrency=args.max_concurrency,
        max_reference_chars=args.max_reference_chars,
        max_file_chars=args.max_file_chars,
        include_references_in_grader=args.include_references_in_grader,
    )
    benchmark.load_dataset(
        limit=args.limit,
        offset=args.offset,
        domain=args.domain,
        category=args.category,
        matter_id=args.matter_id,
    )
    print(f"Loaded {len(benchmark.dataset)} tasks from {args.repo_id}@{args.revision}")
    if args.dry_run:
        if benchmark.dataset:
            references = load_reference_files(
                benchmark.dataset[0],
                repo_id=args.repo_id,
                revision=args.revision,
                max_reference_chars=args.max_reference_chars,
                max_file_chars=args.max_file_chars,
            )
            print(build_answer_prompt(benchmark.dataset[0], references)[:8000])
        return

    results = asyncio.run(benchmark.run())
    run_name = args.run_name or (
        "legal_chrono_"
        f"{args.model.replace('/', '_')}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    results_path, summary_path = write_results(results, args.output_dir, run_name)
    summary = summarize_results(results)
    print(f"Wrote results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Average score: {summary['average_score']:.2f}/5")
    print(f"Pass@4: {summary['pass_at_4']:.2%}")
    print(f"Total cost: ${summary['total_cost']:.6f}")


def _build_result(
    task: LegalChronoTask,
    references: list[ReferenceFileContent],
    model: str,
    grader_model: str,
    answer_response: ModelResponse,
    grader_response: ModelResponse,
    score: LegalChronoScore,
) -> LegalChronoEvaluationResult:
    answer_usage = answer_response.usage
    grader_usage = grader_response.usage
    answer_cost = answer_usage.cost if answer_usage else 0.0
    grader_cost = grader_usage.cost if grader_usage else 0.0
    return LegalChronoEvaluationResult(
        task_id=task.task_id,
        domain=task.domain,
        matter_id=task.matter_id,
        category=task.category,
        occupation=task.occupation,
        prompt=task.prompt,
        reference_files=task.reference_files,
        reference_file_hf_uris=task.reference_file_hf_uris,
        reference_files_truncated=[
            reference.path for reference in references if reference.truncated
        ],
        gold_answer=task.gold_answer,
        scoring_rubric=task.scoring_rubric,
        model=model,
        grader_model=grader_model,
        model_response=answer_response.content,
        grader_response=grader_response.content,
        valid_score=score.valid_score,
        score=score.score,
        normalized_score=score.normalized_score,
        passed=score.passed,
        grader_rationale=score.rationale,
        answer_ttft=answer_response.ttft,
        answer_output_speed=answer_response.output_speed,
        answer_input_tokens_client=answer_response.input_tokens_client,
        answer_input_tokens_model=answer_usage.input_tokens if answer_usage else 0,
        answer_output_tokens_client=answer_response.output_tokens_client,
        answer_output_tokens_model=answer_usage.output_tokens if answer_usage else 0,
        answer_cost=answer_cost,
        grader_ttft=grader_response.ttft,
        grader_output_speed=grader_response.output_speed,
        grader_input_tokens_client=grader_response.input_tokens_client,
        grader_input_tokens_model=grader_usage.input_tokens if grader_usage else 0,
        grader_output_tokens_client=grader_response.output_tokens_client,
        grader_output_tokens_model=grader_usage.output_tokens if grader_usage else 0,
        grader_cost=grader_cost,
        total_cost=answer_cost + grader_cost,
    )


def _format_reference_block(references: Iterable[ReferenceFileContent]) -> str:
    parts = []
    for reference in references:
        parts.append(
            f"<reference_file path=\"{reference.path}\">\n"
            f"{reference.content}\n"
            f"</reference_file>"
        )
    return "\n\n".join(parts) or "[No reference files provided.]"


def _extract_score(text: str) -> int | None:
    patterns = [
        r"(?im)^\s*score\s*:\s*([0-5])\b",
        r"(?i)\bscore\s*(?:is|=)\s*([0-5])\b",
        r"\b([0-5])\s*/\s*5\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1))
    return None


def _extract_rationale(text: str) -> str:
    match = re.search(r"(?is)rationale\s*:\s*(.+)$", text)
    if match:
        return match.group(1).strip()
    return text.strip()


def _percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    p50, p75, p90, p95, p99 = np.percentile(values, [50, 75, 90, 95, 99])
    return {
        "p50": float(p50),
        "p75": float(p75),
        "p90": float(p90),
        "p95": float(p95),
        "p99": float(p99),
    }


def _group_summary(
    results: list[LegalChronoEvaluationResult],
    attribute: str,
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[LegalChronoEvaluationResult]] = {}
    for result in results:
        grouped.setdefault(str(getattr(result, attribute)), []).append(result)

    return {
        key: {
            "task_count": len(group),
            "average_score": mean([item.score for item in group if item.score is not None])
            if any(item.score is not None for item in group)
            else 0.0,
            "average_normalized_score": mean([item.normalized_score for item in group]),
            "pass_at_4": mean([item.passed for item in group]),
        }
        for key, group in grouped.items()
    }


if __name__ == "__main__":
    main()
