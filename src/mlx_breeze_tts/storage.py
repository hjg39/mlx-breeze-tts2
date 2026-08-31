"""Filesystem-only artifact storage helpers; safe to import without MLX."""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def copy_immutable_auxiliary(source: Path, destination: Path) -> dict:
    """Hard-link immutable non-main-model assets when both paths share a volume."""

    linked_files = 0
    copied_files = 0
    logical_bytes = 0
    mutable_names = {"config.json", "audit.json", "audit.static.json"}
    for item in source.iterdir():
        if item.name in mutable_names:
            continue
        if item.name.startswith("model") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        if not item.is_dir() and not (
            item.suffix in {".json", ".txt", ".model", ".md"}
            or item.name.upper().startswith(("LICENSE", "NOTICE"))
        ):
            continue
        candidates = [path for path in item.rglob("*") if path.is_file()]
        if item.is_file():
            candidates = [item]
        for candidate in candidates:
            relative = candidate.relative_to(source)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            logical_bytes += candidate.stat().st_size
            try:
                os.link(candidate.resolve(), target)
                linked_files += 1
            except OSError:
                shutil.copy2(candidate, target)
                copied_files += 1
    return {
        "mode": "hardlink_with_copy_fallback",
        "linked_files": linked_files,
        "copied_files": copied_files,
        "logical_bytes": logical_bytes,
    }
