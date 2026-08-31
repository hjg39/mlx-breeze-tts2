"""Breeze-specific BF16 conversion, quantization, and strict audits."""

import copy
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig
from .loader import _apply_quantization, _load_weights, resolve_model_path
from .model import Model


def audit_checkpoint(path: str | Path) -> dict:
    model_path = resolve_model_path(path)
    config = json.loads((model_path / "config.json").read_text())
    model = Model(ModelConfig.from_dict(config))
    supplied = Model.sanitize(_load_weights(model_path))
    _apply_quantization(model, config, supplied)
    expected = {name for name, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - set(supplied))
    unexpected = sorted(set(supplied) - expected)
    return {
        "path": str(model_path),
        "expected": len(expected),
        "supplied": len(supplied),
        "missing": missing,
        "unexpected": unexpected,
        "pass": not missing and not unexpected,
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
        elif item.is_file() and item.suffix in {".json", ".txt", ".model"}:
            shutil.copy2(item, target)
    config["torch_dtype"] = dtype
    config["mlx_breeze_tts"] = {
        "source": str(source),
        "revision": revision,
        "dtype": dtype,
        "bits": bits,
        "group_size": group_size if bits else None,
    }
    (destination / "config.json").write_text(json.dumps(config, indent=2))
    report = audit_checkpoint(destination)
    (destination / "audit.json").write_text(json.dumps(report, indent=2))
    if not report["pass"]:
        raise RuntimeError(f"Converted checkpoint failed strict audit: {report}")
    return destination
