from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from datagen.legal_matter.io import read_jsonl, sha256_file, write_jsonl
from datagen.legal_matter.types import (
    CanonicalEvent,
    Contradiction,
    GeneratedArtifact,
    GoldQuestion,
    Matter,
    MatterStateSnapshot,
    ValidationReport,
)


@dataclass(frozen=True)
class RepairResult:
    changed: bool
    fixes: list[str] = field(default_factory=list)


def repair_dataset_dir(
    dataset_dir: Path | str,
    report: ValidationReport | None = None,
) -> RepairResult:
    root = Path(dataset_dir)
    matters = read_jsonl(root / "matters.jsonl", Matter)
    events = read_jsonl(root / "canonical_events.jsonl", CanonicalEvent)
    contradictions = read_jsonl(root / "contradictions.jsonl", Contradiction)
    snapshots = read_jsonl(root / "gold_states.jsonl", MatterStateSnapshot)
    artifacts = read_jsonl(root / "artifacts.jsonl", GeneratedArtifact)
    questions = read_jsonl(root / "questions.jsonl", GoldQuestion)

    fixes: list[str] = []
    events_by_key = {(event.matter_id, event.event_id): event for event in events}
    artifacts_by_key = {
        (artifact.matter_id, artifact.artifact_id): artifact for artifact in artifacts
    }

    fixes.extend(_repair_missing_or_changed_files(root, artifacts))
    fixes.extend(_repair_uncovered_material_events(root, events, artifacts, artifacts_by_key))

    artifacts_by_key = {
        (artifact.matter_id, artifact.artifact_id): artifact for artifact in artifacts
    }
    artifact_ids_by_matter = _artifact_ids_by_matter(artifacts)
    event_ids_by_matter = _event_ids_by_matter(events)
    artifacts_by_event = _artifacts_by_event(artifacts)

    fixes.extend(
        _repair_contradictions(
            contradictions,
            artifact_ids_by_matter,
            event_ids_by_matter,
            artifacts_by_event,
        )
    )
    fixes.extend(
        _repair_snapshots(
            snapshots,
            artifact_ids_by_matter,
            event_ids_by_matter,
            artifacts_by_event,
        )
    )
    fixes.extend(
        _repair_questions(
            questions,
            artifact_ids_by_matter,
            event_ids_by_matter,
            artifacts_by_event,
            events_by_key,
        )
    )
    fixes.extend(_repair_matter_links(matters, events, snapshots, artifacts))

    if fixes:
        write_jsonl(root / "matters.jsonl", matters)
        write_jsonl(root / "contradictions.jsonl", contradictions)
        write_jsonl(root / "gold_states.jsonl", snapshots)
        write_jsonl(root / "artifacts.jsonl", artifacts)
        write_jsonl(root / "questions.jsonl", questions)

    return RepairResult(changed=bool(fixes), fixes=fixes)


def _repair_missing_or_changed_files(
    root: Path,
    artifacts: list[GeneratedArtifact],
) -> list[str]:
    fixes: list[str] = []
    for index, artifact in enumerate(artifacts):
        path = root / artifact.file_path
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                _build_recovered_artifact_content(artifact),
                encoding="utf-8",
            )
            fixes.append(f"created missing artifact file {artifact.artifact_id}")
        actual_sha = sha256_file(path)
        if actual_sha != artifact.sha256:
            artifacts[index] = artifact.model_copy(update={"sha256": actual_sha})
            fixes.append(f"refreshed sha256 for {artifact.artifact_id}")
    return fixes


def _repair_uncovered_material_events(
    root: Path,
    events: list[CanonicalEvent],
    artifacts: list[GeneratedArtifact],
    artifacts_by_key: dict[tuple[str, str], GeneratedArtifact],
) -> list[str]:
    fixes: list[str] = []
    covered = {
        (artifact.matter_id, event_id)
        for artifact in artifacts
        for event_id in artifact.referenced_event_ids
    }
    for event in events:
        if not event.material or (event.matter_id, event.event_id) in covered:
            continue

        artifact_id = _supplemental_artifact_id(event.event_id)
        key = (event.matter_id, artifact_id)
        if key in artifacts_by_key:
            covered.add((event.matter_id, event.event_id))
            continue

        artifact_path = (
            root
            / "matters"
            / event.matter_id
            / "artifacts"
            / f"repair_{event.event_id.lower()}.md"
        )
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            _build_supplemental_artifact_content(event),
            encoding="utf-8",
        )
        artifact = GeneratedArtifact(
            artifact_id=artifact_id,
            matter_id=event.matter_id,
            artifact_type="client_note",
            title=f"Supplemental file note: {event.title}",
            file_path=artifact_path.relative_to(root).as_posix(),
            text_summary=(
                "Validator repair note covering canonical event "
                f"{event.event_id}: {event.title}"
            ),
            referenced_event_ids=[event.event_id],
            visible_dates=[event.date or f"relative day {event.relative_day}"],
            reliability_level="medium",
            sha256=sha256_file(artifact_path),
        )
        artifacts.append(artifact)
        artifacts_by_key[key] = artifact
        covered.add((event.matter_id, event.event_id))
        fixes.append(f"created supplemental artifact {artifact_id} for {event.event_id}")
    return fixes


def _repair_contradictions(
    contradictions: list[Contradiction],
    artifact_ids_by_matter: dict[str, set[str]],
    event_ids_by_matter: dict[str, set[str]],
    artifacts_by_event: dict[tuple[str, str], list[str]],
) -> list[str]:
    fixes: list[str] = []
    for index, contradiction in enumerate(contradictions):
        valid_event_ids = [
            event_id
            for event_id in contradiction.event_ids
            if event_id in event_ids_by_matter.get(contradiction.matter_id, set())
        ]
        artifact_ids = {
            artifact_id
            for artifact_id in contradiction.artifact_ids
            if artifact_id in artifact_ids_by_matter.get(contradiction.matter_id, set())
        }
        for event_id in valid_event_ids:
            artifact_ids.update(artifacts_by_event.get((contradiction.matter_id, event_id), []))

        updates: dict[str, object] = {}
        if valid_event_ids != contradiction.event_ids:
            updates["event_ids"] = valid_event_ids
        if sorted(artifact_ids) != contradiction.artifact_ids:
            updates["artifact_ids"] = sorted(artifact_ids)
        if not contradiction.intentional:
            updates["intentional"] = True
        if not contradiction.resolution:
            updates["resolution"] = (
                "Intentional benchmark contradiction. Resolve by comparing the "
                "listed event and artifact evidence, giving more weight to "
                "current, signed, or technically specific records."
            )
        if updates:
            contradictions[index] = contradiction.model_copy(update=updates)
            fixes.append(f"normalised contradiction {contradiction.contradiction_id}")
    return fixes


def _repair_snapshots(
    snapshots: list[MatterStateSnapshot],
    artifact_ids_by_matter: dict[str, set[str]],
    event_ids_by_matter: dict[str, set[str]],
    artifacts_by_event: dict[tuple[str, str], list[str]],
) -> list[str]:
    fixes: list[str] = []
    for index, snapshot in enumerate(snapshots):
        valid_event_ids = [
            event_id
            for event_id in snapshot.supporting_event_ids
            if event_id in event_ids_by_matter.get(snapshot.matter_id, set())
        ]
        artifact_ids = {
            artifact_id
            for artifact_id in snapshot.supporting_artifact_ids
            if artifact_id in artifact_ids_by_matter.get(snapshot.matter_id, set())
        }
        for event_id in valid_event_ids:
            artifact_ids.update(artifacts_by_event.get((snapshot.matter_id, event_id), []))

        updates: dict[str, object] = {}
        if valid_event_ids != snapshot.supporting_event_ids:
            updates["supporting_event_ids"] = valid_event_ids
        if sorted(artifact_ids) != snapshot.supporting_artifact_ids:
            updates["supporting_artifact_ids"] = sorted(artifact_ids)
        if updates:
            snapshots[index] = snapshot.model_copy(update=updates)
            fixes.append(f"normalised snapshot {snapshot.snapshot_id}")
    return fixes


def _repair_questions(
    questions: list[GoldQuestion],
    artifact_ids_by_matter: dict[str, set[str]],
    event_ids_by_matter: dict[str, set[str]],
    artifacts_by_event: dict[tuple[str, str], list[str]],
    events_by_key: dict[tuple[str, str], CanonicalEvent],
) -> list[str]:
    fixes: list[str] = []
    for index, question in enumerate(questions):
        valid_source_event_ids = [
            event_id
            for event_id in question.source_event_ids
            if event_id in event_ids_by_matter.get(question.matter_id, set())
        ]
        evidence_ids = [
            artifact_id
            for artifact_id in question.required_evidence_artifact_ids
            if artifact_id in artifact_ids_by_matter.get(question.matter_id, set())
        ]
        replacement_ids: list[str] = []
        for event_id in valid_source_event_ids:
            replacement_ids.extend(artifacts_by_event.get((question.matter_id, event_id), []))
        evidence_ids = _dedupe([*evidence_ids, *replacement_ids])

        updates: dict[str, object] = {}
        if valid_source_event_ids != question.source_event_ids:
            updates["source_event_ids"] = valid_source_event_ids
        if evidence_ids != question.required_evidence_artifact_ids:
            updates["required_evidence_artifact_ids"] = evidence_ids
        if not evidence_ids:
            fallback = _fallback_question_evidence(
                question,
                events_by_key,
                artifacts_by_event,
            )
            if fallback:
                updates["required_evidence_artifact_ids"] = fallback
        if updates:
            questions[index] = question.model_copy(update=updates)
            fixes.append(f"normalised question {question.question_id}")
    return fixes


def _repair_matter_links(
    matters: list[Matter],
    events: list[CanonicalEvent],
    snapshots: list[MatterStateSnapshot],
    artifacts: list[GeneratedArtifact],
) -> list[str]:
    fixes: list[str] = []
    events_by_matter = _event_ids_by_matter(events)
    snapshots_by_matter: dict[str, set[str]] = {}
    artifacts_by_matter = _artifact_ids_by_matter(artifacts)
    for snapshot in snapshots:
        snapshots_by_matter.setdefault(snapshot.matter_id, set()).add(snapshot.snapshot_id)

    for index, matter in enumerate(matters):
        updates = {
            "event_ids": sorted(events_by_matter.get(matter.matter_id, set())),
            "checkpoint_ids": sorted(snapshots_by_matter.get(matter.matter_id, set())),
            "artifact_ids": sorted(artifacts_by_matter.get(matter.matter_id, set())),
        }
        if (
            updates["event_ids"] != matter.event_ids
            or updates["checkpoint_ids"] != matter.checkpoint_ids
            or updates["artifact_ids"] != matter.artifact_ids
        ):
            matters[index] = matter.model_copy(update=updates)
            fixes.append(f"normalised matter links for {matter.matter_id}")
    return fixes


def _artifact_ids_by_matter(
    artifacts: list[GeneratedArtifact],
) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for artifact in artifacts:
        grouped.setdefault(artifact.matter_id, set()).add(artifact.artifact_id)
    return grouped


def _event_ids_by_matter(events: list[CanonicalEvent]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = {}
    for event in events:
        grouped.setdefault(event.matter_id, set()).add(event.event_id)
    return grouped


def _artifacts_by_event(
    artifacts: list[GeneratedArtifact],
) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = {}
    for artifact in artifacts:
        for event_id in artifact.referenced_event_ids:
            grouped.setdefault((artifact.matter_id, event_id), []).append(artifact.artifact_id)
    return grouped


def _fallback_question_evidence(
    question: GoldQuestion,
    events_by_key: dict[tuple[str, str], CanonicalEvent],
    artifacts_by_event: dict[tuple[str, str], list[str]],
) -> list[str]:
    for event_id in question.source_event_ids:
        event = events_by_key.get((question.matter_id, event_id))
        if event:
            artifact_ids = artifacts_by_event.get((event.matter_id, event.event_id), [])
            if artifact_ids:
                return artifact_ids
    return []


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def _supplemental_artifact_id(event_id: str) -> str:
    return f"ART-FIX-{event_id}"


def _build_supplemental_artifact_content(event: CanonicalEvent) -> str:
    facts = "\n".join(f"- {fact}" for fact in event.factual_assertions)
    disputes = "\n".join(f"- {fact}" for fact in event.disputed_facts)
    parties = ", ".join(event.parties) or "Not specified"
    return f"""# File note - {event.title}

Date recorded: {event.date or f"relative day {event.relative_day}"}
Parties/entities referenced: {parties}

## Matter Note

This file note was added to preserve a discrete artifact trail for a material
event in the synthetic matter file.

## Facts Recorded

{facts or "- No specific factual assertions recorded."}

## Disputed Or Contested Facts

{disputes or "- None specifically recorded."}

## Significance

{event.legal_or_commercial_significance}

## Later Impact

{event.later_impact}
"""


def _build_recovered_artifact_content(artifact: GeneratedArtifact) -> str:
    return f"""# Recovered Artifact Placeholder - {artifact.title}

Artifact ID: {artifact.artifact_id}
Visible dates: {", ".join(artifact.visible_dates) or "Not specified"}
Reliability level: {artifact.reliability_level}

{artifact.text_summary}
"""
