from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from datagen.legal_matter.io import read_jsonl, write_jsonl
from datagen.legal_matter.types import (
    GDPValTask,
    GeneratedArtifact,
    GoldQuestion,
    Matter,
)


DOMAIN_OCCUPATIONS = {
    "commercial_lease": "Commercial Property Lawyers",
    "employment_dispute": "Employment Lawyers",
    "family_property_settlement": "Family Lawyers",
}


@dataclass(frozen=True)
class GDPValExportResult:
    dataset_dir: str
    repo_id: str | None
    task_count: int
    reference_file_count: int
    tasks_path: str
    train_path: str
    reference_files_dir: str


def export_gdpval_style_dataset(
    dataset_dir: Path | str,
    repo_id: str | None = None,
    revision: str = "main",
) -> GDPValExportResult:
    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    matters = read_jsonl(root / "matters.jsonl", Matter)
    artifacts = read_jsonl(root / "artifacts.jsonl", GeneratedArtifact)
    questions = read_jsonl(root / "questions.jsonl", GoldQuestion)

    matters_by_id = {matter.matter_id: matter for matter in matters}
    artifacts_by_key = {
        (artifact.matter_id, artifact.artifact_id): artifact
        for artifact in artifacts
    }

    reference_root = root / "reference_files"
    if reference_root.exists():
        shutil.rmtree(reference_root)
    reference_root.mkdir(parents=True, exist_ok=True)

    copied_reference_files: set[str] = set()
    tasks: list[GDPValTask] = []
    for question in questions:
        matter = matters_by_id[question.matter_id]
        reference_files = _copy_question_reference_files(
            root=root,
            reference_root=reference_root,
            question=question,
            artifacts_by_key=artifacts_by_key,
            copied_reference_files=copied_reference_files,
        )
        tasks.append(
            GDPValTask(
                task_id=question.question_id,
                sector="Professional, Scientific, and Technical Services",
                occupation=DOMAIN_OCCUPATIONS.get(matter.matter_type, "Legal Professionals"),
                domain=matter.matter_type,
                matter_id=matter.matter_id,
                matter_title=matter.title,
                category=question.category,
                prompt=question.prompt,
                reference_files=reference_files,
                reference_file_urls=[
                    _reference_file_url(repo_id, path, revision)
                    for path in reference_files
                    if repo_id
                ],
                reference_file_hf_uris=[
                    _reference_file_hf_uri(repo_id, path, revision)
                    for path in reference_files
                    if repo_id
                ],
                deliverable_text=question.expected_answer,
                deliverable_files=[],
                scoring_rubric=question.scoring_rubric,
                source_event_ids=question.source_event_ids,
                gold_answer=question.expected_answer,
            )
        )

    tasks_path = root / "tasks.jsonl"
    train_path = root / "data" / "train.jsonl"
    write_jsonl(tasks_path, tasks)
    write_jsonl(train_path, tasks)
    _update_dataset_card_config(root / "README.md")
    return GDPValExportResult(
        dataset_dir=str(root),
        repo_id=repo_id,
        task_count=len(tasks),
        reference_file_count=len(copied_reference_files),
        tasks_path=str(tasks_path),
        train_path=str(train_path),
        reference_files_dir=str(reference_root),
    )


def _copy_question_reference_files(
    root: Path,
    reference_root: Path,
    question: GoldQuestion,
    artifacts_by_key: dict[tuple[str, str], GeneratedArtifact],
    copied_reference_files: set[str],
) -> list[str]:
    reference_files: list[str] = []
    for artifact_id in question.required_evidence_artifact_ids:
        artifact = artifacts_by_key.get((question.matter_id, artifact_id))
        if not artifact:
            continue
        source_path = root / artifact.file_path
        if not source_path.exists():
            continue
        destination_relative = (
            Path("reference_files")
            / question.matter_id
            / source_path.name
        ).as_posix()
        destination_path = root / destination_relative
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        if destination_relative not in copied_reference_files:
            shutil.copy2(source_path, destination_path)
            copied_reference_files.add(destination_relative)
        reference_files.append(destination_relative)
    return _dedupe(reference_files)


def _reference_file_url(repo_id: str | None, path: str, revision: str) -> str:
    return f"https://huggingface.co/datasets/{repo_id}/resolve/{revision}/{path}"


def _reference_file_hf_uri(repo_id: str | None, path: str, revision: str) -> str:
    return f"hf://datasets/{repo_id}@{revision}/{path}"


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _update_dataset_card_config(path: Path) -> None:
    front_matter = """---
language:
- en
license: other
task_categories:
- question-answering
- text-generation
pretty_name: Legal Matter Chronology v1
tags:
- synthetic
- legal
- chronology
- evaluation
- multimodal
configs:
- config_name: default
  data_files:
  - split: train
    path: data/train.jsonl
---
"""
    if not path.exists():
        path.write_text(
            front_matter
            + "\n# Legal Matter Chronology v1\n\nSynthetic legal chronology evaluation dataset.\n",
            encoding="utf-8",
        )
        return

    content = path.read_text(encoding="utf-8")
    if not content.startswith("---\n"):
        path.write_text(front_matter + "\n" + content, encoding="utf-8")
        return

    end_index = content.find("\n---\n", 4)
    if end_index == -1:
        path.write_text(front_matter + "\n" + content, encoding="utf-8")
        return

    body = content[end_index + len("\n---\n") :]
    path.write_text(front_matter + body, encoding="utf-8")
