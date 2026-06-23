from __future__ import annotations

import asyncio
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from datagen.legal_matter.backends import (
    FixtureLegalMatterBackend,
    LegalMatterBackend,
    OpenRouterLegalMatterBackend,
)
from datagen.legal_matter.io import sha256_file, write_json, write_jsonl
from datagen.legal_matter.types import (
    DEFAULT_MATTER_DOMAIN,
    DEFAULT_JURISDICTION_STYLE,
    DOMAIN_CONFIGS,
    DEFAULT_MODEL,
    ArtifactContent,
    ArtifactSpec,
    CanonicalEvent,
    Contradiction,
    GeneratedArtifact,
    GoldQuestion,
    Manifest,
    Matter,
    MatterDomain,
    MatterStateSnapshot,
    StructuredMatter,
    ValidationReport,
)


def build_dataset(
    output_dir: Path | str,
    count: int = 1,
    model: str = DEFAULT_MODEL,
    domains: Sequence[str] | None = None,
    max_workers: int = 4,
    matter_concurrency: int = 2,
    artifact_concurrency: int | None = None,
    all_concurrent: bool = False,
    repair_attempts: int = 2,
    use_mock: bool = False,
    force: bool = False,
    backend: LegalMatterBackend | None = None,
) -> ValidationReport:
    return asyncio.run(
        build_dataset_async(
            output_dir=output_dir,
            count=count,
            model=model,
            domains=domains,
            matter_concurrency=matter_concurrency,
            artifact_concurrency=artifact_concurrency or max_workers,
            all_concurrent=all_concurrent,
            repair_attempts=repair_attempts,
            use_mock=use_mock,
            force=force,
            backend=backend,
        )
    )


async def build_dataset_async(
    output_dir: Path | str,
    count: int = 1,
    model: str = DEFAULT_MODEL,
    domains: Sequence[str] | None = None,
    matter_concurrency: int = 2,
    artifact_concurrency: int = 4,
    all_concurrent: bool = False,
    repair_attempts: int = 2,
    use_mock: bool = False,
    force: bool = False,
    backend: LegalMatterBackend | None = None,
) -> ValidationReport:
    output_path = Path(output_dir)
    if count < 1:
        raise ValueError("count must be at least 1")
    if not all_concurrent and matter_concurrency < 1:
        raise ValueError("matter_concurrency must be at least 1")
    if not all_concurrent and artifact_concurrency < 1:
        raise ValueError("artifact_concurrency must be at least 1")
    if repair_attempts < 0:
        raise ValueError("repair_attempts must be 0 or greater")
    selected_domains = _normalise_domains(domains)
    if output_path.exists() and any(output_path.iterdir()) and not force:
        raise FileExistsError(
            f"{output_path} already exists and is not empty. Pass --force to rebuild it."
        )
    if force and output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if all_concurrent:
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            ThreadPoolExecutor(max_workers=max(32, count * 32))
        )

    generation_backend = backend or (
        FixtureLegalMatterBackend() if use_mock else OpenRouterLegalMatterBackend(model)
    )

    matter_semaphore = None if all_concurrent else asyncio.Semaphore(matter_concurrency)
    artifact_semaphore = None if all_concurrent else asyncio.Semaphore(artifact_concurrency)
    matter_results = await asyncio.gather(
        *[
            _build_one_matter(
                index=index,
                domain=_domain_for_index(index, selected_domains),
                backend=generation_backend,
                output_path=output_path,
                matter_semaphore=matter_semaphore,
                artifact_semaphore=artifact_semaphore,
            )
            for index in range(1, count + 1)
        ]
    )
    matter_results = sorted(matter_results, key=lambda result: result.index)

    manifest = Manifest(
        generator_model="fixture" if use_mock else model,
        matter_type=(
            selected_domains[0]
            if len(selected_domains) == 1
            else "mixed"
        ),
        count=count,
        defaults={
            "domains": ",".join(selected_domains),
            "jurisdiction_style": DEFAULT_JURISDICTION_STYLE,
            "artifact_shape": "hybrid folder with JSONL indexes",
            "upload_visibility": "private",
            "repair_attempts": str(repair_attempts),
            "all_concurrent": str(all_concurrent),
        },
    )

    matters = [result.matter for result in matter_results]
    events = [
        event
        for result in matter_results
        for event in result.structured.canonical_events
    ]
    contradictions = [
        contradiction
        for result in matter_results
        for contradiction in result.structured.contradictions
    ]
    state_snapshots = [
        snapshot
        for result in matter_results
        for snapshot in result.structured.state_snapshots
    ]
    generated_artifacts = [
        artifact
        for result in matter_results
        for artifact in result.generated_artifacts
    ]
    questions = [
        question
        for result in matter_results
        for question in result.questions
    ]

    write_json(output_path / "manifest.json", manifest)
    write_jsonl(output_path / "matters.jsonl", matters)
    write_jsonl(output_path / "canonical_events.jsonl", events)
    write_jsonl(output_path / "contradictions.jsonl", contradictions)
    write_jsonl(output_path / "gold_states.jsonl", state_snapshots)
    write_jsonl(output_path / "artifacts.jsonl", generated_artifacts)
    write_jsonl(output_path / "questions.jsonl", questions)

    from datagen.legal_matter.validator import validate_dataset_dir

    report = validate_dataset_dir(output_path, write_report=False)
    repair_log: list[dict[str, object]] = []
    if repair_attempts:
        from datagen.legal_matter.repair import repair_dataset_dir

        for attempt in range(1, repair_attempts + 1):
            if report.passed:
                break
            repair_result = repair_dataset_dir(output_path, report)
            repair_log.append(
                {
                    "attempt": attempt,
                    "changed": repair_result.changed,
                    "fixes": repair_result.fixes,
                }
            )
            if not repair_result.changed:
                break
            report = validate_dataset_dir(output_path, write_report=False)

    if repair_log:
        write_json(output_path / "repair_report.json", {"attempts": repair_log})
    write_json(output_path / "validation_report.json", report)
    _write_dataset_card(output_path / "README.md", manifest, report)
    return report


@dataclass(frozen=True)
class _MatterBuildResult:
    index: int
    matter: Matter
    structured: StructuredMatter
    generated_artifacts: list[GeneratedArtifact]
    questions: list[GoldQuestion]


def _normalise_domains(domains: Sequence[str] | None) -> list[MatterDomain]:
    raw_domains = list(domains or [DEFAULT_MATTER_DOMAIN])
    if not raw_domains:
        raise ValueError("At least one domain must be provided")

    selected: list[MatterDomain] = []
    allowed_domains = set(DOMAIN_CONFIGS)
    for domain in raw_domains:
        if domain not in allowed_domains:
            allowed = ", ".join(sorted(allowed_domains))
            raise ValueError(f"Unknown domain {domain!r}. Choose from: {allowed}")
        selected.append(domain)  # type: ignore[arg-type]
    return selected


def _domain_for_index(index: int, domains: list[MatterDomain]) -> MatterDomain:
    return domains[(index - 1) % len(domains)]


def _matter_id_for(index: int, domain: MatterDomain) -> str:
    prefix = DOMAIN_CONFIGS[domain]["matter_id_prefix"]
    return f"{prefix}_{index:03d}"


async def _build_one_matter(
    index: int,
    domain: MatterDomain,
    backend: LegalMatterBackend,
    output_path: Path,
    matter_semaphore: asyncio.Semaphore | None,
    artifact_semaphore: asyncio.Semaphore | None,
) -> _MatterBuildResult:
    if matter_semaphore is None:
        return await _build_one_matter_unlocked(
            index=index,
            domain=domain,
            backend=backend,
            output_path=output_path,
            artifact_semaphore=artifact_semaphore,
        )

    async with matter_semaphore:
        return await _build_one_matter_unlocked(
            index=index,
            domain=domain,
            backend=backend,
            output_path=output_path,
            artifact_semaphore=artifact_semaphore,
        )


async def _build_one_matter_unlocked(
    index: int,
    domain: MatterDomain,
    backend: LegalMatterBackend,
    output_path: Path,
    artifact_semaphore: asyncio.Semaphore | None,
) -> _MatterBuildResult:
    matter_id = _matter_id_for(index, domain)
    chronology = await asyncio.to_thread(
        backend.generate_chronology,
        index,
        domain,
    )
    structured = await asyncio.to_thread(
        backend.structure_matter,
        chronology,
        matter_id,
        domain,
    )
    structured = _normalise_structured_matter(structured, matter_id, domain)
    blueprint = await asyncio.to_thread(
        backend.create_artifact_blueprint,
        structured,
    )
    artifact_specs = _normalise_artifact_specs(blueprint.artifacts)
    structured = _attach_artifact_support(structured, artifact_specs)

    artifact_task = asyncio.create_task(
        _generate_and_write_artifacts(
            backend=backend,
            structured=structured,
            artifact_specs=artifact_specs,
            output_dir=output_path,
            matter_id=matter_id,
            artifact_semaphore=artifact_semaphore,
        )
    )
    question_task = asyncio.to_thread(
        backend.generate_questions,
        structured,
        artifact_specs,
        matter_id,
    )
    artifact_records, question_set = await asyncio.gather(
        artifact_task,
        question_task,
    )
    matter = _build_matter_record(structured, artifact_specs)
    return _MatterBuildResult(
        index=index,
        matter=matter,
        structured=structured,
        generated_artifacts=artifact_records,
        questions=question_set.questions,
    )


def _normalise_structured_matter(
    structured: StructuredMatter,
    matter_id: str,
    domain: MatterDomain,
) -> StructuredMatter:
    event_ids = [event.event_id for event in structured.canonical_events]
    checkpoint_ids = [snapshot.snapshot_id for snapshot in structured.state_snapshots]

    matter = structured.matter.model_copy(
        update={
            "matter_id": matter_id,
            "jurisdiction_style": DEFAULT_JURISDICTION_STYLE,
            "matter_type": domain,
            "event_ids": event_ids,
            "checkpoint_ids": checkpoint_ids,
        }
    )
    events = [
        event.model_copy(update={"matter_id": matter_id})
        for event in sorted(structured.canonical_events, key=lambda item: item.sequence_index)
    ]
    contradictions = [
        contradiction.model_copy(update={"matter_id": matter_id})
        for contradiction in structured.contradictions
    ]
    snapshots = [
        snapshot.model_copy(update={"matter_id": matter_id})
        for snapshot in sorted(structured.state_snapshots, key=lambda item: item.relative_day)
    ]
    return structured.model_copy(
        update={
            "matter": matter,
            "canonical_events": events,
            "contradictions": contradictions,
            "state_snapshots": snapshots,
        }
    )


def _normalise_artifact_specs(artifact_specs: list[ArtifactSpec]) -> list[ArtifactSpec]:
    normalised: list[ArtifactSpec] = []
    seen: set[str] = set()
    for position, spec in enumerate(artifact_specs, start=1):
        artifact_id = spec.artifact_id or f"ART-{position:03d}"
        if artifact_id in seen:
            raise ValueError(f"Duplicate artifact_id in blueprint: {artifact_id}")
        seen.add(artifact_id)
        normalised.append(spec.model_copy(update={"artifact_id": artifact_id}))
    return normalised


def _attach_artifact_support(
    structured: StructuredMatter,
    artifact_specs: list[ArtifactSpec],
) -> StructuredMatter:
    artifacts_by_event: dict[str, set[str]] = {}
    for spec in artifact_specs:
        for event_id in spec.source_event_ids:
            artifacts_by_event.setdefault(event_id, set()).add(spec.artifact_id)

    snapshots: list[MatterStateSnapshot] = []
    for snapshot in structured.state_snapshots:
        supporting_artifact_ids = set(snapshot.supporting_artifact_ids)
        for event_id in snapshot.supporting_event_ids:
            supporting_artifact_ids.update(artifacts_by_event.get(event_id, set()))
        snapshots.append(
            snapshot.model_copy(
                update={"supporting_artifact_ids": sorted(supporting_artifact_ids)}
            )
        )

    contradictions: list[Contradiction] = []
    for contradiction in structured.contradictions:
        artifact_ids = set(contradiction.artifact_ids)
        for event_id in contradiction.event_ids:
            artifact_ids.update(artifacts_by_event.get(event_id, set()))
        contradictions.append(
            contradiction.model_copy(update={"artifact_ids": sorted(artifact_ids)})
        )

    return structured.model_copy(
        update={
            "state_snapshots": snapshots,
            "contradictions": contradictions,
        }
    )


async def _generate_and_write_artifacts(
    backend: LegalMatterBackend,
    structured: StructuredMatter,
    artifact_specs: list[ArtifactSpec],
    output_dir: Path,
    matter_id: str,
    artifact_semaphore: asyncio.Semaphore | None,
) -> list[GeneratedArtifact]:
    artifact_root = output_dir / "matters" / matter_id / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    async def generate(spec: ArtifactSpec) -> tuple[ArtifactSpec, ArtifactContent]:
        if artifact_semaphore is None:
            content = await asyncio.to_thread(
                backend.generate_artifact,
                structured,
                spec,
            )
        else:
            async with artifact_semaphore:
                content = await asyncio.to_thread(
                    backend.generate_artifact,
                    structured,
                    spec,
                )
        if content.artifact_id != spec.artifact_id:
            raise ValueError(
                f"Artifact content ID {content.artifact_id} does not match spec {spec.artifact_id}"
            )
        return spec, content

    generated = await asyncio.gather(*[generate(spec) for spec in artifact_specs])

    records: list[GeneratedArtifact] = []
    for spec, content in generated:
        artifact_path = artifact_root / spec.file_name
        resolved_root = artifact_root.resolve()
        resolved_path = artifact_path.resolve()
        if not resolved_path.is_relative_to(resolved_root):
            raise ValueError(f"Artifact file path escapes artifact directory: {spec.file_name}")
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(content.content.rstrip() + "\n", encoding="utf-8")
        relative_path = artifact_path.relative_to(output_dir).as_posix()
        records.append(
            GeneratedArtifact(
                artifact_id=spec.artifact_id,
                matter_id=matter_id,
                artifact_type=spec.artifact_type,
                title=spec.title,
                file_path=relative_path,
                text_summary=content.text_summary,
                referenced_event_ids=spec.source_event_ids,
                visible_dates=content.visible_dates,
                reliability_level=spec.reliability,
                sha256=sha256_file(artifact_path),
            )
        )
    return records


def _build_matter_record(
    structured: StructuredMatter,
    artifact_specs: list[ArtifactSpec],
) -> Matter:
    return structured.matter.model_copy(
        update={
            "event_ids": [event.event_id for event in structured.canonical_events],
            "checkpoint_ids": [
                snapshot.snapshot_id for snapshot in structured.state_snapshots
            ],
            "artifact_ids": [artifact.artifact_id for artifact in artifact_specs],
        }
    )


def _write_dataset_card(
    path: Path,
    manifest: Manifest,
    report: ValidationReport,
) -> None:
    card = f"""---
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
---

# Legal Matter Chronology v1

This dataset contains fictional, synthetic commercial lease dispute matters for
evaluating whether language models can reason over a changing legal matter file.
It is not legal advice and does not describe real disputes.

## Contents

- `manifest.json`: dataset version, generation defaults, prompt version, and model.
- `matters.jsonl`: one row per synthetic matter.
- `canonical_events.jsonl`: hidden event-level source of truth.
- `contradictions.jsonl`: intentional contradictions and resolutions.
- `gold_states.jsonl`: current-position snapshots at matter checkpoints.
- `artifacts.jsonl`: artifact metadata and file paths.
- `questions.jsonl`: gold benchmark questions and evidence requirements.
- `matters/<matter_id>/artifacts/`: generated markdown and CSV artifacts.

## Defaults

- Dataset version: `{manifest.dataset_version}`
- Prompt version: `{manifest.prompt_version}`
- Matter type: `{manifest.matter_type}`
- Jurisdiction style: `{manifest.jurisdiction_style}`
- Generator model: `{manifest.generator_model}`
- Validation passed: `{report.passed}`
- Validation issues: `{report.issue_count}`

V1 uses markdown stand-ins for legal documents, email chains, Slack-style
threads, inspection notes, and image placeholders. CSV artifacts are generated
as real CSV files. Future versions can add rendered PDFs, DOCX files, and image
artifacts while preserving the same index structure.
"""
    path.write_text(card, encoding="utf-8")
