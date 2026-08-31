"""Breeze-specific BF16 conversion, quantization, and strict audits."""

import copy
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig
from .loader import _apply_quantization, _load_weights, resolve_model_path
from .model import Model
from .provenance import inherited_upstream_identity


def audit_checkpoint(path: str | Path) -> dict:
    model_path = resolve_model_path(path)
    config = json.loads((model_path / "config.json").read_text())
    model = Model(ModelConfig.from_dict(config))
    supplied = Model.sanitize(_load_weights(model_path))
    _apply_quantization(model, config, supplied)
    parameters = dict(tree_flatten(model.parameters()))
    expected = set(parameters)
    missing = sorted(expected - set(supplied))
    unexpected = sorted(set(supplied) - expected)
    shape_mismatches = {
        name: {"expected": parameters[name].shape, "supplied": supplied[name].shape}
        for name in sorted(expected & set(supplied))
        if parameters[name].shape != supplied[name].shape
    }
    dtype_counts: dict[str, int] = {}
    for value in supplied.values():
        dtype_counts[str(value.dtype)] = dtype_counts.get(str(value.dtype), 0) + 1
    return {
        "path": str(model_path),
        "expected": len(expected),
        "supplied": len(supplied),
        "missing": missing,
        "unexpected": unexpected,
        "shape_mismatches": shape_mismatches,
        "dtype_counts": dtype_counts,
        "pass": not missing and not unexpected and not shape_mismatches,
    }


def _quantize(model: Model, config: dict, bits: int, group_size: int) -> None:
    def predicate(_path: str, module: nn.Module) -> bool:
        return (
            hasattr(module, "to_quantized")
            and hasattr(module, "weight")
            and module.weight.shape[-1] % group_size == 0
        )

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode="affine",
        class_predicate=predicate,
    )
    config["quantization"] = {
        "group_size": group_size,
        "bits": bits,
        "mode": "affine",
    }
    config["quantization_config"] = copy.deepcopy(config["quantization"])


def _save_weights(model: Model, output: Path, max_bytes: int = 5 << 30) -> None:
    weights = dict(tree_flatten(model.parameters()))
    shards: list[dict] = []
    current: dict = {}
    size = 0
    for name, weight in weights.items():
        if current and size + weight.nbytes > max_bytes:
            shards.append(current)
            current, size = {}, 0
        current[name] = weight
        size += weight.nbytes
    if current:
        shards.append(current)
    pattern = (
        "model.safetensors"
        if len(shards) == 1
        else "model-{index:05d}-of-{count:05d}.safetensors"
    )
    weight_map = {}
    for index, shard in enumerate(shards, 1):
        filename = (
            pattern
            if len(shards) == 1
            else pattern.format(index=index, count=len(shards))
        )
        mx.save_safetensors(str(output / filename), shard, metadata={"format": "mlx"})
        weight_map.update({name: filename for name in shard})
    (output / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": sum(w.nbytes for w in weights.values())},
                "weight_map": dict(sorted(weight_map.items())),
            },
            indent=2,
        )
    )


def convert(
    source: str | Path,
    output: str | Path,
    *,
    dtype: str = "bfloat16",
    bits: int | None = None,
    group_size: int = 64,
    revision: str | None = None,
) -> Path:
    if dtype not in {"bfloat16", "float16", "float32"}:
        raise ValueError("dtype must be bfloat16, float16, or float32")
    if bits not in {None, 4, 8}:
        raise ValueError("bits must be 4, 8, or omitted")
    source_path = resolve_model_path(source, revision=revision)
    destination = Path(output).expanduser()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {destination}")
    destination.mkdir(parents=True, exist_ok=True)

    config = json.loads((source_path / "config.json").read_text())
    source_quantization = config.get("quantization") or config.get(
        "quantization_config"
    )
    conversion_metadata = config.get("mlx_breeze_tts") or {}
    upstream_source, upstream_revision = inherited_upstream_identity(
        source, source_path, revision
    )
    if source_quantization and not bits:
        raise ValueError(
            "BF16 conversion requires the official unquantized source checkpoint"
        )
    if bits is not None and not (
        conversion_metadata.get("dtype") == "bfloat16"
        and conversion_metadata.get("bits") is None
    ):
        raise ValueError(
            "4-bit and 8-bit conversion must use a verified BF16 artifact "
            "created by this converter"
        )
    model = Model(ModelConfig.from_dict(config))
    supplied = Model.sanitize(_load_weights(source_path))
    model.load_weights(list(supplied.items()), strict=True)
    target_dtype = getattr(mx, dtype)
    model.load_weights(
        [(name, value.astype(target_dtype)) for name, value in supplied.items()],
        strict=True,
    )
    if bits is not None:
        _quantize(model, config, bits, group_size)
    mx.eval(model.parameters())
    _save_weights(model, destination)

    for item in source_path.iterdir():
        if item.name.startswith("model") and item.suffix == ".safetensors":
            continue
        if item.name == "model.safetensors.index.json":
            continue
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        elif item.is_file() and (
            item.suffix in {".json", ".txt", ".model", ".md"}
            or item.name.upper().startswith(("LICENSE", "NOTICE"))
        ):
            shutil.copy2(item, target)
    config["torch_dtype"] = dtype
    config["mlx_breeze_tts"] = {
        "source": str(source),
        "revision": revision,
        "upstream_source": upstream_source,
        "upstream_revision": upstream_revision,
        "dtype": dtype,
        "bits": bits,
        "group_size": group_size if bits else None,
        "converter_version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weight_sha256": {
            item.name: hashlib.sha256(item.read_bytes()).hexdigest()
            for item in sorted(destination.glob("model*.safetensors"))
        },
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    report = audit_checkpoint(destination)
    (destination / "audit.json").write_text(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError(f"Converted checkpoint failed strict audit: {report}")
    return destination
