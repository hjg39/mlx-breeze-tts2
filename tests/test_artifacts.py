import json
import os
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file

from mlx_breeze_tts.artifacts import inspect_checkpoint, materialize_linked_bf16
from mlx_breeze_tts.cli import _parser


def _fixture(root: Path) -> Path:
    root.mkdir()
    weights = {
        "backbone_model.norm.weight": np.ones((2,), dtype=np.float16),
        "depth_decoder.model.embed_tokens.weight": np.ones((2, 2), dtype=np.float16),
        "text_encoder_proj.weight": np.ones((2, 2), dtype=np.float16),
    }
    save_file(weights, root / "model.safetensors")
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 20},
                "weight_map": {key: "model.safetensors" for key in sorted(weights)},
            }
        )
    )
    (root / "config.json").write_text(json.dumps({"torch_dtype": "float16"}))
    codec = root / "audio_tokenizer"
    codec.mkdir()
    (codec / "config.json").write_text("{}")
    save_file(
        {"codec.weight": np.ones((2,), dtype=np.float16)}, codec / "model.safetensors"
    )
    return root


def test_static_inspection_checks_index_without_metal(tmp_path):
    source = _fixture(tmp_path / "source")
    report = inspect_checkpoint(source)
    assert report["pass"] is True
    assert report["tensor_count"] == 3
    assert report["runtime_tied_embedding_derived"] is True
    assert report["strict_metal_audit"] == "pending"

    index = json.loads((source / "model.safetensors.index.json").read_text())
    index["weight_map"]["missing.weight"] = "model.safetensors"
    (source / "model.safetensors.index.json").write_text(json.dumps(index))
    failed = inspect_checkpoint(source)
    assert failed["pass"] is False
    assert "index references missing tensors" in failed["issues"]


def test_static_inspection_rejects_quantization_policy_metadata_mismatch(tmp_path):
    source = _fixture(tmp_path / "source")
    config = json.loads((source / "config.json").read_text())
    config["quantization"] = {
        "bits": 4,
        "group_size": 64,
        "mode": "affine",
        "policy": "full",
    }
    config["mlx_breeze_tts"] = {
        "quantization_policy": {
            "name": "full",
            "quantized_modules": ["lm_head"],
        }
    }
    (source / "config.json").write_text(json.dumps(config))
    report = inspect_checkpoint(source)
    assert report["pass"] is False
    assert any("module list" in issue for issue in report["issues"])


def test_low_disk_commands_parse_without_importing_mlx(tmp_path):
    inspected = _parser().parse_args(["inspect-checkpoint", str(tmp_path)])
    assert inspected.command == "inspect-checkpoint"
    linked = _parser().parse_args(
        ["link-bf16", "--source", str(tmp_path), "--output", str(tmp_path / "o")]
    )
    assert linked.command == "link-bf16"


def test_linked_bf16_rejects_non_bf16_source(tmp_path):
    source = _fixture(tmp_path / "source")
    try:
        materialize_linked_bf16(source, tmp_path / "output")
    except ValueError as exc:
        assert "all main tensors to be BF16" in str(exc)
    else:
        raise AssertionError("float16 fixture should be rejected")


def test_copy_helper_links_safetensors(tmp_path, monkeypatch):
    source = _fixture(tmp_path / "source")
    from mlx_breeze_tts import artifacts

    monkeypatch.setattr(
        artifacts,
        "inspect_checkpoint",
        lambda path: {
            "pass": True,
            "dtype_counts": {"BF16": 3},
            "runtime_dtype_counts_after_sanitize": {"BF16": 3},
        },
    )
    output = materialize_linked_bf16(source, tmp_path / "output")
    assert (
        os.stat(source / "model.safetensors").st_ino
        == os.stat(output / "model.safetensors").st_ino
    )
    assert (
        os.stat(source / "audio_tokenizer/model.safetensors").st_ino
        == os.stat(output / "audio_tokenizer/model.safetensors").st_ino
    )
    metadata = json.loads((output / "config.json").read_text())["mlx_breeze_tts"]
    assert metadata["storage_mode"] == "hardlink"
    assert metadata["strict_metal_audit"] == "pending"
