"""Checkpoint discovery and strict MLX loading."""

import glob
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .config import ModelConfig
from .model import Model

DEFAULT_MODEL = "LunaFox/Breeze-TTS-2-mlx-4bit"
ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.model",
    "*.txt",
    "audio_tokenizer/**",
]


def resolve_model_path(
    model: str | Path = DEFAULT_MODEL, revision: str | None = None
) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path
    if str(model).startswith(("/", "./", "../", "~")):
        raise FileNotFoundError(f"Local model path not found: {model}")
    return Path(
        snapshot_download(str(model), revision=revision, allow_patterns=ALLOW_PATTERNS)
    )


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    files = sorted(glob.glob(str(model_path / "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights: dict[str, mx.array] = {}
    for file in files:
        weights.update(mx.load(file))
    return weights


def _apply_quantization(
    model: Model, config: dict, weights: dict[str, mx.array]
) -> None:
    quantization = config.get("quantization") or config.get("quantization_config")
    if not quantization:
        return
    group_size = int(quantization.get("group_size", 64))
    bits = int(quantization.get("bits", 4))

    def predicate(path: str, module: nn.Module):
        if not hasattr(module, "to_quantized") or not hasattr(module, "weight"):
            return False
        if module.weight.shape[-1] % group_size:
            return False
        if path in quantization and isinstance(quantization[path], dict):
            return quantization[path]
        return f"{path}.scales" in weights

    nn.quantize(
        model,
        group_size=group_size,
        bits=bits,
        mode=quantization.get("mode", "affine"),
        class_predicate=predicate,
    )


def load(
    model: str | Path = DEFAULT_MODEL,
    *,
    revision: str | None = None,
    lazy: bool = False,
) -> Model:
    model_path = resolve_model_path(model, revision=revision)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found at {config_path}")
    config_data = json.loads(config_path.read_text())
    instance = Model(ModelConfig.from_dict(config_data))
    weights = Model.sanitize(_load_weights(model_path))
    _apply_quantization(instance, config_data, weights)
    instance.load_weights(list(weights.items()), strict=True)
    instance = Model.post_load_hook(instance, model_path)
    if not lazy:
        mx.eval(instance.parameters())
    instance.eval()
    return instance
