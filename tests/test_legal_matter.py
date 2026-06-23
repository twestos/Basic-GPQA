from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from datagen.legal_matter.backends import FixtureLegalMatterBackend
from datagen.legal_matter.generator import build_dataset
from datagen.legal_matter.gdpval_export import export_gdpval_style_dataset
from datagen.legal_matter.io import read_json, read_jsonl, write_jsonl
from datagen.legal_matter.repair import repair_dataset_dir
from datagen.legal_matter.types import (
    ArtifactBlueprint,
    ArtifactSpec,
    Contradiction,
    GDPValTask,
    GeneratedArtifact,
    GoldQuestion,
    Matter,
    MatterStateSnapshot,
)
from datagen.legal_matter.upload import upload_dataset_folder
from datagen.legal_matter.validator import validate_dataset_dir


class LegalMatterPipelineTests(unittest.TestCase):
    def test_schema_rejects_unsafe_artifact_file_name(self) -> None:
        with self.assertRaises(ValidationError):
            ArtifactSpec(
                artifact_id="ART-X",
                artifact_type="email_thread",
                title="Unsafe path",
                author="Test",
                recipients=["Test"],
                displayed_date="today",
                source_event_ids=["EVT-001"],
                facts_revealed=["Something happened."],
                file_name="../escape.md",
            )

    def test_jsonl_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "matters.jsonl"
            rows = [
                Matter(
                    matter_id="matter-1",
                    title="Matter 1",
                    summary="Synthetic matter.",
                )
            ]
            write_jsonl(path, rows)

            loaded = read_jsonl(path, Matter)

        self.assertEqual(loaded[0].matter_id, "matter-1")

    def test_mock_build_creates_manifest_and_validates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            report = build_dataset(
                output,
                count=1,
                use_mock=True,
                max_workers=1,
                matter_concurrency=1,
            )

            manifest = read_json(output / "manifest.json")

            self.assertTrue(report.passed)
            self.assertEqual(manifest["generator_model"], "fixture")
            self.assertTrue((output / "README.md").exists())
            self.assertTrue(
                (
                    output
                    / "matters"
                    / "lease_matter_001"
                    / "artifacts"
                    / "ledger_004_arrears.csv"
                ).exists()
            )

    def test_mock_build_supports_multiple_concurrent_matters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            report = build_dataset(
                output,
                count=3,
                use_mock=True,
                matter_concurrency=3,
                artifact_concurrency=6,
            )
            matters = read_jsonl(output / "matters.jsonl", Matter)

        self.assertTrue(report.passed)
        self.assertEqual([matter.matter_id for matter in matters], [
            "lease_matter_001",
            "lease_matter_002",
            "lease_matter_003",
        ])

    def test_mock_build_cycles_requested_domains(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            report = build_dataset(
                output,
                count=5,
                use_mock=True,
                domains=[
                    "commercial_lease",
                    "employment_dispute",
                    "family_property_settlement",
                ],
                matter_concurrency=2,
                artifact_concurrency=4,
            )
            matters = read_jsonl(output / "matters.jsonl", Matter)

        self.assertTrue(report.passed)
        self.assertEqual(
            [matter.matter_type for matter in matters],
            [
                "commercial_lease",
                "employment_dispute",
                "family_property_settlement",
                "commercial_lease",
                "employment_dispute",
            ],
        )
        self.assertEqual(
            [matter.matter_id for matter in matters],
            [
                "lease_matter_001",
                "employment_matter_002",
                "family_property_matter_003",
                "lease_matter_004",
                "employment_matter_005",
            ],
        )

    def test_mock_build_all_concurrent_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            report = build_dataset(
                output,
                count=4,
                use_mock=True,
                domains=[
                    "commercial_lease",
                    "employment_dispute",
                    "family_property_settlement",
                ],
                all_concurrent=True,
            )
            manifest = read_json(output / "manifest.json")
            questions = read_jsonl(output / "questions.jsonl", GoldQuestion)

        self.assertTrue(report.passed)
        self.assertEqual(manifest["defaults"]["all_concurrent"], "True")
        self.assertGreaterEqual(len(questions), 24)

    def test_build_auto_repair_fixes_undercovered_backend_output(self) -> None:
        class UndercoveredBackend(FixtureLegalMatterBackend):
            def create_artifact_blueprint(self, structured_matter):
                blueprint = super().create_artifact_blueprint(structured_matter)
                return ArtifactBlueprint(
                    artifacts=[
                        artifact
                        for artifact in blueprint.artifacts
                        if "EVT-008" not in artifact.source_event_ids
                    ]
                )

        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            report = build_dataset(
                output,
                count=1,
                backend=UndercoveredBackend(),
                repair_attempts=2,
                matter_concurrency=1,
                artifact_concurrency=2,
            )
            artifacts = read_jsonl(output / "artifacts.jsonl", GeneratedArtifact)

        self.assertTrue(report.passed)
        self.assertIn(
            "ART-FIX-EVT-008",
            {artifact.artifact_id for artifact in artifacts},
        )

    def test_validator_fails_missing_event_ref(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(output, count=1, use_mock=True, max_workers=1)
            artifacts_path = output / "artifacts.jsonl"
            artifacts = read_jsonl(artifacts_path, GeneratedArtifact)
            artifacts[0] = artifacts[0].model_copy(
                update={"referenced_event_ids": ["EVT-999"]}
            )
            write_jsonl(artifacts_path, artifacts)

            report = validate_dataset_dir(output, write_report=False)

        self.assertFalse(report.passed)
        self.assertIn("artifact_missing_event", {issue.code for issue in report.issues})

    def test_validator_fails_unmarked_contradiction(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(output, count=1, use_mock=True, max_workers=1)
            contradictions_path = output / "contradictions.jsonl"
            contradictions = read_jsonl(contradictions_path, Contradiction)
            contradictions[0] = contradictions[0].model_copy(
                update={"intentional": False, "resolution": None}
            )
            write_jsonl(contradictions_path, contradictions)

            report = validate_dataset_dir(output, write_report=False)

        self.assertFalse(report.passed)
        self.assertIn("unmarked_contradiction", {issue.code for issue in report.issues})

    def test_validator_fails_checkpoint_without_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(output, count=1, use_mock=True, max_workers=1)
            states_path = output / "gold_states.jsonl"
            states = read_jsonl(states_path, MatterStateSnapshot)
            states[0] = states[0].model_copy(update={"supporting_artifact_ids": []})
            write_jsonl(states_path, states)

            report = validate_dataset_dir(output, write_report=False)

        self.assertFalse(report.passed)
        self.assertIn(
            "checkpoint_without_artifact_support",
            {issue.code for issue in report.issues},
        )

    def test_validator_fails_duplicate_artifact_id(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(output, count=1, use_mock=True, max_workers=1)
            artifacts_path = output / "artifacts.jsonl"
            artifacts = read_jsonl(artifacts_path, GeneratedArtifact)
            write_jsonl(artifacts_path, [*artifacts, artifacts[0]])

            report = validate_dataset_dir(output, write_report=False)

        self.assertFalse(report.passed)
        self.assertIn("duplicate_artifact_id", {issue.code for issue in report.issues})

    def test_repair_fixes_common_validation_drift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(
                output,
                count=1,
                use_mock=True,
                max_workers=1,
                repair_attempts=0,
            )

            artifacts_path = output / "artifacts.jsonl"
            artifacts = read_jsonl(artifacts_path, GeneratedArtifact)
            artifacts[0] = artifacts[0].model_copy(update={"referenced_event_ids": []})
            write_jsonl(artifacts_path, artifacts)

            contradictions_path = output / "contradictions.jsonl"
            contradictions = read_jsonl(contradictions_path, Contradiction)
            contradictions[0] = contradictions[0].model_copy(
                update={"intentional": False, "resolution": None}
            )
            write_jsonl(contradictions_path, contradictions)

            questions_path = output / "questions.jsonl"
            questions = read_jsonl(questions_path, GoldQuestion)
            questions[0] = questions[0].model_copy(
                update={"required_evidence_artifact_ids": ["ART-999"]}
            )
            write_jsonl(questions_path, questions)

            initial_report = validate_dataset_dir(output, write_report=False)
            repair_result = repair_dataset_dir(output, initial_report)
            final_report = validate_dataset_dir(output, write_report=False)

        self.assertTrue(repair_result.changed)
        self.assertTrue(final_report.passed)

    def test_gdpval_export_writes_tasks_and_reference_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            build_dataset(output, count=1, use_mock=True, max_workers=1)

            result = export_gdpval_style_dataset(
                output,
                repo_id="example/legal-matter",
            )
            tasks = read_jsonl(output / "tasks.jsonl", GDPValTask)
            train_tasks = read_jsonl(output / "data" / "train.jsonl", GDPValTask)
            reference_files = list((output / "reference_files").rglob("*.*"))
            readme = (output / "README.md").read_text(encoding="utf-8")

        self.assertEqual(result.task_count, len(tasks))
        self.assertEqual(result.task_count, len(train_tasks))
        self.assertIn("path: data/train.jsonl", readme)
        self.assertGreater(result.reference_file_count, 0)
        self.assertGreater(len(reference_files), 0)
        self.assertTrue(tasks[0].reference_files)
        self.assertTrue(tasks[0].reference_file_urls[0].startswith(
            "https://huggingface.co/datasets/example/legal-matter/resolve/main/"
        ))
        self.assertTrue(tasks[0].reference_file_hf_uris[0].startswith(
            "hf://datasets/example/legal-matter@main/"
        ))
        self.assertEqual(tasks[0].deliverable_text, tasks[0].gold_answer)

    def test_cli_smoke_build(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "dataset"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "datagen.legal_matter",
                    "build",
                    "--mock",
                    "--count",
                    "1",
                    "--output",
                    str(output),
                    "--max-workers",
                    "1",
                    "--matter-concurrency",
                    "1",
                ],
                cwd=Path(__file__).resolve().parents[1],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Validation passed: True", result.stdout)

    def test_upload_uses_private_dataset_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            class CommitInfo:
                commit_url = "https://huggingface.co/datasets/example/legal/commit/abc"

            with (
                patch.dict(os.environ, {"HF_TOKEN": "hf_test"}),
                patch("datagen.legal_matter.upload.HfApi") as api_cls,
            ):
                api = api_cls.return_value
                api.upload_folder.return_value = CommitInfo()

                result = upload_dataset_folder(root, "example/legal", private=True)

            api.create_repo.assert_called_once_with(
                repo_id="example/legal",
                repo_type="dataset",
                private=True,
                exist_ok=True,
            )
            api.upload_folder.assert_called_once()
            _, kwargs = api.upload_folder.call_args
            self.assertEqual(kwargs["repo_type"], "dataset")
            self.assertEqual(result.commit_url, CommitInfo.commit_url)


if __name__ == "__main__":
    unittest.main()
