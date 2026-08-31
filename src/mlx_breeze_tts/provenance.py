"""Resolve immutable model and upstream identities from paths and metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")


def snapshot_revision(path: str | Path) -> str | None:
    path = Path(path).expanduser().resolve()
    for candidate in (path.name, path.parent.name):
        if _COMMIT_RE.fullmatch(candidate):
            return candidate
    return None


def checkpoint_provenance(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    config_path = path / "config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    metadata = config.get("mlx_breeze_tts") or {}
    artifact_revision = metadata.get("artifact_revision") or snapshot_revision(path)
    upstream_revision = metadata.get("upstream_revision") or metadata.get("revision")
    return {
        "artifact_revision": artifact_revision,
        "upstream_source": metadata.get("upstream_source") or metadata.get("source"),
        "upstream_revision": upstream_revision,
        "dtype": metadata.get("dtype") or config.get("torch_dtype"),
        "bits": metadata.get("bits"),
        "storage_mode": metadata.get("storage_mode"),
        "weight_sha256": metadata.get("weight_sha256") or {},
    }


def inherited_upstream_identity(
    source: str | Path, source_path: str | Path, requested_revision: str | None
) -> tuple[str, str | None]:
    provenance = checkpoint_provenance(source_path)
    upstream_source = provenance.get("upstream_source") or str(source)
    upstream_revision = (
        provenance.get("upstream_revision")
        or requested_revision
        or provenance.get("artifact_revision")
    )
    return upstream_source, upstream_revision
