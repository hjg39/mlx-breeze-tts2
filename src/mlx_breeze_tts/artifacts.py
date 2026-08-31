"""Metal-independent checkpoint inspection and low-disk BF16 materialization."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open

_DTYPE_BYTES = {
    "BOOL": 1,
    "I8": 1,
    "U8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_REQUIRED_MAIN_KEYS = {
    "backbone_model.norm.weight",
    "depth_decoder.model.embed_tokens.weight",
    "text_encoder_proj.weight",
}
_REQUIRED_CODEC_FILES = {
    "audio_tokenizer/config.json",
    "audio_tokenizer/model.safetensors",
}


def _tensor_nbytes(shape: list[int], dtype: str) -> int:
    elements = 1
    for dimension in shape:
        elements *= dimension
    return elements * _DTYPE_BYTES.get(dtype, 0)


def inspect_checkpoint(path: str | Path) -> dict:
    """Inspect safetensors headers and index consistency without importing MLX.

    This is intentionally a structural preflight. It cannot replace a strict
    ``Model.load_weights`` audit on a Metal-capable process.
    """

    root = Path(path).expanduser().resolve()
    issues: list[str] = []
    files = sorted(root.glob("model*.safetensors"))
    if not files:
        issues.append("no top-level model safetensors files")

    key_to_file: dict[str, str] = {}
    duplicates: set[str] = set()
    dtype_counts: Counter[str] = Counter()
    key_dtypes: dict[str, str] = {}
    tensor_bytes = 0
    file_reports = []
    for file in files:
        try:
            with safe_open(file, framework="np") as handle:
                keys = list(handle.keys())
                for key in keys:
                    if key in key_to_file:
                        duplicates.add(key)
                    key_to_file[key] = file.name
                    tensor = handle.get_slice(key)
                    dtype = tensor.get_dtype()
                    shape = list(tensor.get_shape())
                    dtype_counts[dtype] += 1
                    key_dtypes[key] = dtype
                    tensor_bytes += _tensor_nbytes(shape, dtype)
                file_reports.append(
                    {
                        "file": file.name,
                        "bytes": file.stat().st_size,
                        "tensor_count": len(keys),
                    }
                )
        except Exception as exc:  # corrupt/unrecognized safetensors
            issues.append(f"cannot inspect {file.name}: {exc}")

    if duplicates:
        issues.append("duplicate tensor keys: " + ", ".join(sorted(duplicates)))

    index_path = root / "model.safetensors.index.json"
    index_report = {"present": index_path.is_file(), "key_count": 0}
    if index_path.is_file():
        try:
            index = json.loads(index_path.read_text())
            weight_map = index.get("weight_map", {})
            index_report["key_count"] = len(weight_map)
            actual = set(key_to_file)
            indexed = set(weight_map)
            missing_from_files = sorted(indexed - actual)
            missing_from_index = sorted(actual - indexed)
            wrong_files = sorted(
                key
                for key in actual & indexed
                if weight_map[key] != key_to_file[key]
            )
            index_report.update(
                {
                    "missing_from_files": missing_from_files,
                    "missing_from_index": missing_from_index,
                    "wrong_files": wrong_files,
                }
            )
            if missing_from_files:
                issues.append("index references missing tensors")
            if missing_from_index:
                issues.append("safetensors contain unindexed tensors")
            if wrong_files:
                issues.append("index maps tensors to the wrong shard")
        except (OSError, json.JSONDecodeError, AttributeError) as exc:
            issues.append(f"invalid model index: {exc}")

    keys = set(key_to_file)
    missing_required = sorted(_REQUIRED_MAIN_KEYS - keys)
    if missing_required:
        issues.append("missing required main tensors: " + ", ".join(missing_required))
    missing_codec_files = sorted(
        name for name in _REQUIRED_CODEC_FILES if not (root / name).is_file()
    )
    if missing_codec_files:
        issues.append("missing codec files: " + ", ".join(missing_codec_files))

    codec_keys = {key for key in keys if key.startswith("codec_model.")}
    generated_or_dropped = {
        key
        for key in keys
        if key.endswith(".initialized") or "rotary_emb.inv_freq" in key
    }
    tied_source = "depth_decoder.model.embed_tokens.weight"
    tied_target = "backbone_model.embed_tokens.embed_audio_tokens.weight"
    runtime_keys = keys - codec_keys - generated_or_dropped
    if tied_source in runtime_keys:
        runtime_keys.add(tied_target)
    runtime_dtype_counts = Counter(
        key_dtypes[key] for key in runtime_keys if key in key_dtypes
    )

    return {
        "schema_version": 1,
        "audit_level": "static_preflight",
        "path": str(root),
        "files": file_reports,
        "tensor_count": len(keys),
        "runtime_tensor_count_after_sanitize": len(runtime_keys),
        "embedded_codec_tensor_count": len(codec_keys),
        "dropped_generated_tensor_count": len(generated_or_dropped),
        "runtime_tied_embedding_derived": tied_source in keys,
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "runtime_dtype_counts_after_sanitize": dict(
            sorted(runtime_dtype_counts.items())
        ),
        "tensor_bytes": tensor_bytes,
        "index": index_report,
        "issues": issues,
        "pass": not issues,
        "strict_metal_audit": "pending",
    }


def _sha256(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.suffix == ".safetensors":
        os.link(source.resolve(), target)
    else:
        shutil.copy2(source, target)


def materialize_linked_bf16(source: str | Path, output: str | Path) -> Path:
    """Create a zero-copy BF16 candidate from an official local snapshot.

    Only safetensors are hard-linked; mutable metadata is copied. The source
    and destination must be on the same filesystem. The result remains a
    candidate until the Metal-dependent strict audit passes.
    """

    source_path = Path(source).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    if source_path == destination or source_path in destination.parents:
        raise ValueError("Output must be outside the source checkpoint directory")
    if not source_path.is_dir():
        raise FileNotFoundError(f"Local source checkpoint not found: {source_path}")
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")

    source_audit = inspect_checkpoint(source_path)
    if not source_audit["pass"]:
        raise ValueError(f"Source checkpoint failed static preflight: {source_audit}")
    config_path = source_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    config = json.loads(config_path.read_text())
    if config.get("quantization") or config.get("quantization_config"):
        raise ValueError("Linked BF16 materialization requires an unquantized source")
    source_dtypes = set(source_audit["runtime_dtype_counts_after_sanitize"])
    if source_dtypes - {"BF16"}:
        raise ValueError(
            "Linked BF16 materialization requires all main tensors to be BF16; "
            f"found {sorted(source_dtypes)}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.mkdir(exist_ok=True)
    if source_path.stat().st_dev != destination.stat().st_dev:
        raise OSError("Source and output must be on the same filesystem")

    for item in source_path.rglob("*"):
        if not item.is_file():
            continue
        relative = item.relative_to(source_path)
        if relative == Path("config.json"):
            continue
        _copy_or_link(item, destination / relative)

    model_files = sorted(destination.glob("model*.safetensors"))
    config["torch_dtype"] = "bfloat16"
    config["mlx_breeze_tts"] = {
        "source": str(source_path),
        "dtype": "bfloat16",
        "bits": None,
        "group_size": None,
        "converter_version": "0.1.0",
        "storage_mode": "hardlink",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weight_sha256": {file.name: _sha256(file) for file in model_files},
        "strict_metal_audit": "pending",
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    audit = inspect_checkpoint(destination)
    (destination / "audit.static.json").write_text(json.dumps(audit, indent=2))
    if not audit["pass"]:
        raise RuntimeError(f"Linked checkpoint failed static preflight: {audit}")
    return destination
