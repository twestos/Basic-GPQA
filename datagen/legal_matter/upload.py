from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import HfApi


@dataclass(frozen=True)
class UploadResult:
    repo_id: str
    repo_type: str
    private: bool
    dataset_dir: str
    commit_url: str | None


def upload_dataset_folder(
    dataset_dir: Path | str,
    repo_id: str,
    private: bool = True,
) -> UploadResult:
    load_dotenv()
    token = os.getenv("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN is not set. Add it to .env or export it before uploading."
        )

    root = Path(dataset_dir)
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {root}")

    api = HfApi(token=token)
    api.create_repo(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    commit_info = api.upload_folder(
        folder_path=str(root),
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Upload synthetic legal matter chronology dataset",
    )
    commit_url = getattr(commit_info, "commit_url", None)
    return UploadResult(
        repo_id=repo_id,
        repo_type="dataset",
        private=private,
        dataset_dir=str(root),
        commit_url=commit_url,
    )
