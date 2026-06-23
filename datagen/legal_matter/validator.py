from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from datagen.legal_matter.io import read_jsonl, sha256_file, write_json
from datagen.legal_matter.types import (
    CanonicalEvent,
    Contradiction,
    GeneratedArtifact,
    GoldQuestion,
    Matter,
    MatterStateSnapshot,
    ValidationIssue,
    ValidationReport,
)


ModelT = TypeVar("ModelT", bound=BaseModel)


def validate_dataset_dir(
    dataset_dir: Path | str,
    write_report: bool = True,
) -> ValidationReport:
    root = Path(dataset_dir)
    issues: list[ValidationIssue] = []

    matters = _load_jsonl(root / "matters.jsonl", Matter, issues)
    events = _load_jsonl(root / "canonical_events.jsonl", CanonicalEvent, issues)
    contradictions = _load_jsonl(root / "contradictions.jsonl", Contradiction, issues)
    snapshots = _load_jsonl(root / "gold_states.jsonl", MatterStateSnapshot, issues)
    artifacts = _load_jsonl(root / "artifacts.jsonl", GeneratedArtifact, issues)
    questions = _load_jsonl(root / "questions.jsonl", GoldQuestion, issues)

    matter_ids = {matter.matter_id for matter in matters}
    events_by_matter = _group_event_ids(events)
    material_events_by_matter = _group_material_event_ids(events)
    artifacts_by_matter = _group_artifact_ids(artifacts)
    snapshots_by_matter = _group_snapshot_ids(snapshots)
    artifact_refs_by_matter = _group_artifact_event_refs(artifacts)

    _validate_unique_ids(matters, artifacts, events, snapshots, questions, issues)
    _validate_matter_links(
        matters,
        events_by_matter,
        snapshots_by_matter,
        artifacts_by_matter,
        issues,
    )
    _validate_artifacts(root, artifacts, events_by_matter, issues)
    _validate_event_coverage(
        material_events_by_matter,
        artifact_refs_by_matter,
        issues,
    )
    _validate_contradictions(
        contradictions,
        events_by_matter,
        artifacts_by_matter,
        issues,
    )
    _validate_snapshots(
        snapshots,
        events_by_matter,
        artifacts_by_matter,
        issues,
    )
    _validate_questions(
        questions,
        matter_ids,
        events_by_matter,
        artifacts_by_matter,
        issues,
    )

    passed = not any(issue.severity == "error" for issue in issues)
    report = ValidationReport(
        dataset_dir=str(root),
        passed=passed,
        issue_count=len(issues),
        issues=issues,
    )
    if write_report:
        write_json(root / "validation_report.json", report)
    return report


def _load_jsonl(
    path: Path,
    model_type: type[ModelT],
    issues: list[ValidationIssue],
) -> list[ModelT]:
    if not path.exists():
        issues.append(
            ValidationIssue(
                severity="error",
                code="missing_index_file",
                message=f"Missing required index file: {path.name}",
            )
        )
        return []
    try:
        return read_jsonl(path, model_type)
    except Exception as error:
        issues.append(
            ValidationIssue(
                severity="error",
                code="invalid_index_file",
                message=f"Could not parse {path.name}: {error}",
            )
        )
        return []


def _group_event_ids(events: list[CanonicalEvent]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for event in events:
        grouped[event.matter_id].add(event.event_id)
    return grouped


def _group_material_event_ids(events: list[CanonicalEvent]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for event in events:
        if event.material:
            grouped[event.matter_id].add(event.event_id)
    return grouped


def _group_artifact_ids(artifacts: list[GeneratedArtifact]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts:
        grouped[artifact.matter_id].add(artifact.artifact_id)
    return grouped


def _group_snapshot_ids(snapshots: list[MatterStateSnapshot]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for snapshot in snapshots:
        grouped[snapshot.matter_id].add(snapshot.snapshot_id)
    return grouped


def _group_artifact_event_refs(
    artifacts: list[GeneratedArtifact],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for artifact in artifacts:
        grouped[artifact.matter_id].update(artifact.referenced_event_ids)
    return grouped


def _validate_unique_ids(
    matters: list[Matter],
    artifacts: list[GeneratedArtifact],
    events: list[CanonicalEvent],
    snapshots: list[MatterStateSnapshot],
    questions: list[GoldQuestion],
    issues: list[ValidationIssue],
) -> None:
    _add_duplicate_issues(
        "matter",
        [(matter.matter_id, matter.matter_id) for matter in matters],
        issues,
    )
    _add_duplicate_issues(
        "artifact",
        [
            (f"{artifact.matter_id}:{artifact.artifact_id}", artifact.artifact_id)
            for artifact in artifacts
        ],
        issues,
    )
    _add_duplicate_issues(
        "event",
        [(f"{event.matter_id}:{event.event_id}", event.event_id) for event in events],
        issues,
    )
    _add_duplicate_issues(
        "snapshot",
        [
            (f"{snapshot.matter_id}:{snapshot.snapshot_id}", snapshot.snapshot_id)
            for snapshot in snapshots
        ],
        issues,
    )
    _add_duplicate_issues(
        "question",
        [
            (f"{question.matter_id}:{question.question_id}", question.question_id)
            for question in questions
        ],
        issues,
    )


def _add_duplicate_issues(
    label: str,
    keys: list[tuple[str, str]],
    issues: list[ValidationIssue],
) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for key, display_id in keys:
        if key in seen:
            duplicates.add(display_id)
        seen.add(key)
    for duplicate_id in sorted(duplicates):
        issues.append(
            ValidationIssue(
                severity="error",
                code=f"duplicate_{label}_id",
                message=f"Duplicate {label} ID: {duplicate_id}",
            )
        )


def _validate_matter_links(
    matters: list[Matter],
    events_by_matter: dict[str, set[str]],
    snapshots_by_matter: dict[str, set[str]],
    artifacts_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for matter in matters:
        event_ids = events_by_matter.get(matter.matter_id, set())
        snapshot_ids = snapshots_by_matter.get(matter.matter_id, set())
        artifact_ids = artifacts_by_matter.get(matter.matter_id, set())
        for event_id in matter.event_ids:
            if event_id not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="matter_missing_event",
                        message=f"Matter references missing event {event_id}",
                        matter_id=matter.matter_id,
                        event_id=event_id,
                    )
                )
        for snapshot_id in matter.checkpoint_ids:
            if snapshot_id not in snapshot_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="matter_missing_snapshot",
                        message=f"Matter references missing snapshot {snapshot_id}",
                        matter_id=matter.matter_id,
                        snapshot_id=snapshot_id,
                    )
                )
        for artifact_id in matter.artifact_ids:
            if artifact_id not in artifact_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="matter_missing_artifact",
                        message=f"Matter references missing artifact {artifact_id}",
                        matter_id=matter.matter_id,
                        artifact_id=artifact_id,
                    )
                )


def _validate_artifacts(
    root: Path,
    artifacts: list[GeneratedArtifact],
    events_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for artifact in artifacts:
        event_ids = events_by_matter.get(artifact.matter_id, set())
        for event_id in artifact.referenced_event_ids:
            if event_id not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="artifact_missing_event",
                        message=f"Artifact references missing event {event_id}",
                        matter_id=artifact.matter_id,
                        artifact_id=artifact.artifact_id,
                        event_id=event_id,
                    )
                )

        file_path = root / artifact.file_path
        if not file_path.exists():
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="artifact_file_missing",
                    message=f"Artifact file does not exist: {artifact.file_path}",
                    matter_id=artifact.matter_id,
                    artifact_id=artifact.artifact_id,
                )
            )
            continue
        actual_sha = sha256_file(file_path)
        if actual_sha != artifact.sha256:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="artifact_sha_mismatch",
                    message=f"Artifact SHA mismatch for {artifact.file_path}",
                    matter_id=artifact.matter_id,
                    artifact_id=artifact.artifact_id,
                )
            )


def _validate_event_coverage(
    material_events_by_matter: dict[str, set[str]],
    artifact_refs_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for matter_id, event_ids in material_events_by_matter.items():
        covered_event_ids = artifact_refs_by_matter.get(matter_id, set())
        for event_id in sorted(event_ids - covered_event_ids):
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="material_event_uncovered",
                    message=f"Material event {event_id} is not represented by any artifact",
                    matter_id=matter_id,
                    event_id=event_id,
                )
            )


def _validate_contradictions(
    contradictions: list[Contradiction],
    events_by_matter: dict[str, set[str]],
    artifacts_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for contradiction in contradictions:
        event_ids = events_by_matter.get(contradiction.matter_id, set())
        artifact_ids = artifacts_by_matter.get(contradiction.matter_id, set())
        if not contradiction.intentional and not contradiction.resolution:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="unmarked_contradiction",
                    message=(
                        "Contradiction is not marked intentional and has no resolution"
                    ),
                    matter_id=contradiction.matter_id,
                )
            )
        for event_id in contradiction.event_ids:
            if event_id not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="contradiction_missing_event",
                        message=f"Contradiction references missing event {event_id}",
                        matter_id=contradiction.matter_id,
                        event_id=event_id,
                    )
                )
        for artifact_id in contradiction.artifact_ids:
            if artifact_id not in artifact_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="contradiction_missing_artifact",
                        message=f"Contradiction references missing artifact {artifact_id}",
                        matter_id=contradiction.matter_id,
                        artifact_id=artifact_id,
                    )
                )


def _validate_snapshots(
    snapshots: list[MatterStateSnapshot],
    events_by_matter: dict[str, set[str]],
    artifacts_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for snapshot in snapshots:
        event_ids = events_by_matter.get(snapshot.matter_id, set())
        artifact_ids = artifacts_by_matter.get(snapshot.matter_id, set())
        if not snapshot.supporting_artifact_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="checkpoint_without_artifact_support",
                    message="Checkpoint snapshot has no supporting artifacts",
                    matter_id=snapshot.matter_id,
                    snapshot_id=snapshot.snapshot_id,
                )
            )
        for event_id in snapshot.supporting_event_ids:
            if event_id not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="snapshot_missing_event",
                        message=f"Snapshot references missing event {event_id}",
                        matter_id=snapshot.matter_id,
                        event_id=event_id,
                        snapshot_id=snapshot.snapshot_id,
                    )
                )
        for artifact_id in snapshot.supporting_artifact_ids:
            if artifact_id not in artifact_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="snapshot_missing_artifact",
                        message=f"Snapshot references missing artifact {artifact_id}",
                        matter_id=snapshot.matter_id,
                        artifact_id=artifact_id,
                        snapshot_id=snapshot.snapshot_id,
                    )
                )


def _validate_questions(
    questions: list[GoldQuestion],
    matter_ids: set[str],
    events_by_matter: dict[str, set[str]],
    artifacts_by_matter: dict[str, set[str]],
    issues: list[ValidationIssue],
) -> None:
    for question in questions:
        if question.matter_id not in matter_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="question_missing_matter",
                    message=f"Question references missing matter {question.matter_id}",
                    matter_id=question.matter_id,
                )
            )
        if not question.required_evidence_artifact_ids:
            issues.append(
                ValidationIssue(
                    severity="error",
                    code="question_without_evidence",
                    message="Question has no required evidence artifacts",
                    matter_id=question.matter_id,
                )
            )
        event_ids = events_by_matter.get(question.matter_id, set())
        artifact_ids = artifacts_by_matter.get(question.matter_id, set())
        for event_id in question.source_event_ids:
            if event_id not in event_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="question_missing_event",
                        message=f"Question references missing event {event_id}",
                        matter_id=question.matter_id,
                        event_id=event_id,
                    )
                )
        for artifact_id in question.required_evidence_artifact_ids:
            if artifact_id not in artifact_ids:
                issues.append(
                    ValidationIssue(
                        severity="error",
                        code="question_missing_artifact",
                        message=f"Question references missing artifact {artifact_id}",
                        matter_id=question.matter_id,
                        artifact_id=artifact_id,
                    )
                )
