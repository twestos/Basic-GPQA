from __future__ import annotations

import argparse
import sys
from pathlib import Path

from datagen.legal_matter.generator import build_dataset
from datagen.legal_matter.gdpval_export import export_gdpval_style_dataset
from datagen.legal_matter.io import write_json
from datagen.legal_matter.repair import repair_dataset_dir
from datagen.legal_matter.types import DEFAULT_MATTER_DOMAIN, DEFAULT_MODEL, DOMAIN_CONFIGS
from datagen.legal_matter.upload import upload_dataset_folder
from datagen.legal_matter.validator import validate_dataset_dir


DEFAULT_OUTPUT = "datasets/legal_matter_chronology_v1"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m datagen.legal_matter",
        description="Generate, validate, and upload synthetic legal matter chronology datasets.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Build a local dataset folder.")
    build_parser.add_argument("--count", type=int, default=1)
    build_parser.add_argument("--output", type=Path, default=Path(DEFAULT_OUTPUT))
    build_parser.add_argument("--model", default=DEFAULT_MODEL)
    build_parser.add_argument(
        "--domains",
        nargs="+",
        choices=sorted(DOMAIN_CONFIGS),
        default=[DEFAULT_MATTER_DOMAIN],
        help="Matter domains to cycle through when count is greater than one.",
    )
    build_parser.add_argument(
        "--matter-concurrency",
        type=int,
        default=2,
        help="Number of independent matters to generate at once.",
    )
    build_parser.add_argument(
        "--artifact-concurrency",
        type=int,
        default=None,
        help="Total artifact-generation calls to run at once across all matters.",
    )
    build_parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Compatibility alias for --artifact-concurrency when that flag is omitted.",
    )
    build_parser.add_argument(
        "--all-concurrent",
        action="store_true",
        help=(
            "Start all matter pipelines at once and do not limit artifact "
            "generation concurrency."
        ),
    )
    build_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use deterministic fixture data instead of calling OpenRouter.",
    )
    build_parser.add_argument(
        "--force",
        action="store_true",
        help="Delete and rebuild the output directory if it already contains files.",
    )
    build_parser.add_argument(
        "--repair-attempts",
        type=int,
        default=2,
        help="Number of surgical validate/repair attempts after generation.",
    )
    build_parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Disable automatic repair after generation.",
    )

    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate an existing dataset folder.",
    )
    validate_parser.add_argument("--dataset-dir", type=Path, required=True)

    repair_parser = subparsers.add_parser(
        "repair",
        help="Run surgical metadata/artifact repairs on an existing dataset folder.",
    )
    repair_parser.add_argument("--dataset-dir", type=Path, required=True)
    repair_parser.add_argument("--max-attempts", type=int, default=2)

    export_parser = subparsers.add_parser(
        "export-gdpval",
        help="Create GDPVal-style tasks.jsonl and reference_files/ from a dataset folder.",
    )
    export_parser.add_argument("--dataset-dir", type=Path, required=True)
    export_parser.add_argument(
        "--repo-id",
        default=None,
        help="Optional Hugging Face dataset repo ID used to populate URL and hf:// URI columns.",
    )
    export_parser.add_argument("--revision", default="main")

    upload_parser = subparsers.add_parser(
        "upload",
        help="Upload a validated dataset folder to Hugging Face.",
    )
    upload_parser.add_argument("--dataset-dir", type=Path, required=True)
    upload_parser.add_argument("--repo-id", required=True)
    upload_parser.add_argument(
        "--export-gdpval",
        action="store_true",
        help="Create or refresh GDPVal-style tasks.jsonl and reference_files/ before upload.",
    )
    upload_parser.add_argument("--revision", default="main")
    visibility = upload_parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private",
        dest="private",
        action="store_true",
        default=True,
        help="Create or update a private dataset repo. This is the default.",
    )
    visibility.add_argument(
        "--public",
        dest="private",
        action="store_false",
        help="Create or update a public dataset repo.",
    )
    upload_parser.add_argument(
        "--allow-invalid",
        action="store_true",
        help="Upload even if validation fails.",
    )

    args = parser.parse_args(argv)

    if args.command == "build":
        report = build_dataset(
            output_dir=args.output,
            count=args.count,
            model=args.model,
            domains=args.domains,
            max_workers=args.max_workers,
            matter_concurrency=args.matter_concurrency,
            artifact_concurrency=args.artifact_concurrency,
            all_concurrent=args.all_concurrent,
            repair_attempts=0 if args.no_repair else args.repair_attempts,
            use_mock=args.mock,
            force=args.force,
        )
        print(f"Built dataset at {args.output}")
        print(f"Validation passed: {report.passed} ({report.issue_count} issues)")
        return 0 if report.passed else 1

    if args.command == "validate":
        report = validate_dataset_dir(args.dataset_dir)
        print(f"Validation passed: {report.passed} ({report.issue_count} issues)")
        if not report.passed:
            for issue in report.issues:
                print(f"- [{issue.severity}] {issue.code}: {issue.message}")
        return 0 if report.passed else 1

    if args.command == "repair":
        report = validate_dataset_dir(args.dataset_dir, write_report=False)
        repair_log = []
        for attempt in range(1, args.max_attempts + 1):
            if report.passed:
                break
            repair_result = repair_dataset_dir(args.dataset_dir, report)
            repair_log.append(
                {
                    "attempt": attempt,
                    "changed": repair_result.changed,
                    "fixes": repair_result.fixes,
                }
            )
            if not repair_result.changed:
                break
            report = validate_dataset_dir(args.dataset_dir, write_report=False)

        write_json(args.dataset_dir / "repair_report.json", {"attempts": repair_log})
        write_json(args.dataset_dir / "validation_report.json", report)
        print(f"Repair attempts: {len(repair_log)}")
        for attempt in repair_log:
            print(
                f"- attempt {attempt['attempt']}: "
                f"{len(attempt['fixes'])} fixes, changed={attempt['changed']}"
            )
        print(f"Validation passed: {report.passed} ({report.issue_count} issues)")
        return 0 if report.passed else 1

    if args.command == "export-gdpval":
        result = export_gdpval_style_dataset(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            revision=args.revision,
        )
        print(f"Wrote GDPVal-style tasks to {result.tasks_path}")
        print(f"Wrote Hugging Face viewer table to {result.train_path}")
        print(f"Copied {result.reference_file_count} reference files")
        print(f"Task rows: {result.task_count}")
        return 0

    if args.command == "upload":
        report = validate_dataset_dir(args.dataset_dir)
        if not report.passed and not args.allow_invalid:
            print(
                f"Validation failed with {report.issue_count} issues. "
                "Use --allow-invalid to upload anyway.",
                file=sys.stderr,
            )
            return 1
        if args.export_gdpval:
            export_result = export_gdpval_style_dataset(
                dataset_dir=args.dataset_dir,
                repo_id=args.repo_id,
                revision=args.revision,
            )
            print(
                "Prepared GDPVal-style export: "
                f"{export_result.task_count} tasks, "
                f"{export_result.reference_file_count} reference files, "
                f"viewer table {export_result.train_path}"
            )
        result = upload_dataset_folder(
            dataset_dir=args.dataset_dir,
            repo_id=args.repo_id,
            private=args.private,
        )
        print(f"Uploaded {result.dataset_dir} to {result.repo_id}")
        if result.commit_url:
            print(result.commit_url)
        return 0

    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
