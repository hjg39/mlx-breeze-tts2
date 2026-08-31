"""Framework-neutral validation and serialization for parity snapshots."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .parity import EXACT_SECTIONS


MODEL_ID = "BreezeBlue/Breeze-TTS-2"
MODEL_REVISION = "c1c8ca18b70b30822735633991d9ebf4898e47d4"
CASE_ID = "reference-direction-prefill-v1"
WEIGHT_NAMES = (
    "lm_head.weight",
    "text_encoder_proj.weight",
    "backbone_model.norm.weight",
    "depth_decoder.model.norm.weight",
)
_REVISION = re.compile(r"^[0-9a-f]{40}$")


def tensor_values(value: Any) -> list:
    """Convert a selected small tensor slice into finite JSON numeric values."""

    if hasattr(value, "detach"):
        value = value.detach()
    dtype_name = str(getattr(value, "dtype", ""))
    if "bfloat16" in dtype_name:
        if hasattr(value, "float"):
            value = value.float()
        elif hasattr(value, "astype"):
            value = value.astype(np.float32)
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    if not np.all(np.isfinite(array)):
        raise ValueError("Parity tensor contains non-finite values")
    return array.tolist()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_snapshot(
    *,
    runtime: str,
    template_render: dict,
    token_ids: dict,
    reference_audio_codes: dict,
    masks: dict,
    weight_shapes: dict,
    deterministic_tokens: dict,
    intermediate_tensors: dict,
    runtime_provenance: dict,
) -> dict:
    snapshot = {
        "schema_version": 1,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "case_id": CASE_ID,
        "runtime": runtime,
        "runtime_provenance": runtime_provenance,
        "template_render": template_render,
        "token_ids": token_ids,
        "reference_audio_codes": reference_audio_codes,
        "masks": masks,
        "weight_shapes": weight_shapes,
        "deterministic_tokens": deterministic_tokens,
        "intermediate_tensors": intermediate_tensors,
    }
    missing = [name for name in EXACT_SECTIONS if snapshot.get(name) is None]
    if missing:
        raise ValueError(f"Parity snapshot missing exact sections: {missing}")
    if not _REVISION.fullmatch(snapshot["model_revision"]):
        raise ValueError("Parity model revision must be a 40-character commit")
    if set(weight_shapes) != set(WEIGHT_NAMES):
        raise ValueError("Parity snapshot does not contain the canonical weight shapes")
    if not intermediate_tensors:
        raise ValueError("Parity snapshot requires intermediate tensors")
    for name, value in intermediate_tensors.items():
        array = np.asarray(value, dtype=np.float64)
        if not array.size or not np.all(np.isfinite(array)):
            raise ValueError(f"Invalid parity intermediate tensor: {name}")
    return snapshot


def write_snapshot(snapshot: dict, output: str | Path) -> Path:
    destination = Path(output).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2))
    return destination
