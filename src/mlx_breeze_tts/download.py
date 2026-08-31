"""Pinned official checkpoint acquisition without duplicate local copies."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from huggingface_hub import snapshot_download

from .provenance import snapshot_revision


OFFICIAL_MODEL = "BreezeBlue/Breeze-TTS-2"
OFFICIAL_REVISION = "c1c8ca18b70b30822735633991d9ebf4898e47d4"
OFFICIAL_ALLOW_PATTERNS = (
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "audio_tokenizer/**",
)


def _files_and_bytes(snapshot: Path) -> tuple[list[str], int]:
    files: list[str] = []
    total = 0
    for path in sorted(snapshot.rglob("*")):
        if not path.is_file():
            continue
        files.append(path.relative_to(snapshot).as_posix())
        total += path.stat().st_size
    return files, total


def download_official_checkpoint(
    output_report: str | Path | None = None,
    *,
    downloader: Callable[..., str] = snapshot_download,
) -> tuple[Path, Path | None]:
    """Download the immutable official snapshot into the shared HF cache."""

    resolved = Path(
        downloader(
            OFFICIAL_MODEL,
            revision=OFFICIAL_REVISION,
            allow_patterns=list(OFFICIAL_ALLOW_PATTERNS),
        )
    ).expanduser().resolve()
    revision = snapshot_revision(resolved)
    if revision != OFFICIAL_REVISION:
        raise RuntimeError(
            f"Resolved revision mismatch: expected {OFFICIAL_REVISION}, got {revision}"
        )
    if not (resolved / "config.json").is_file():
        raise RuntimeError("Official snapshot is missing config.json")
    if not any(resolved.glob("*.safetensors")):
        raise RuntimeError("Official snapshot is missing safetensors weights")
    files, total = _files_and_bytes(resolved)
    report_path = None
    if output_report is not None:
        report_path = Path(output_report).expanduser().resolve()
        report = {
            "schema_version": 1,
            "status": "pass",
            "model_id": OFFICIAL_MODEL,
            "requested_revision": OFFICIAL_REVISION,
            "resolved_revision": revision,
            "snapshot": str(resolved),
            "artifact_bytes": total,
            "file_count": len(files),
            "files": files,
            "allow_patterns": list(OFFICIAL_ALLOW_PATTERNS),
            "downloaded_at": datetime.now(timezone.utc).isoformat(),
            "storage_policy": "shared_huggingface_cache_no_local_duplicate",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return resolved, report_path
