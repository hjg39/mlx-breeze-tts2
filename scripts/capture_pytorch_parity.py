#!/usr/bin/env python3
"""Capture an official PyTorch prefill snapshot at a recorded source commit."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from mlx_breeze_tts.parity_capture import (
    WEIGHT_NAMES,
    build_snapshot,
    file_sha256,
    tensor_values,
    write_snapshot,
)


def _source_revision(repository: Path) -> str:
    revision = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if dirty:
        raise RuntimeError("Official PyTorch source checkout must be clean")
    return revision


def _eager_attention_config(config: dict) -> dict:
    """Copy a Breeze config and force its nested text encoder to eager attention."""

    patched = json.loads(json.dumps(config))
    patched["_attn_implementation"] = "eager"
    text_encoder = patched.get("text_encoder_config")
    if not isinstance(text_encoder, dict):
        raise TypeError("Breeze checkpoint is missing text_encoder_config")
    text_encoder["preferred_attn_implementation"] = "eager"
    text_encoder["_attn_implementation"] = "eager"
    return patched


@contextmanager
def _eager_checkpoint_overlay(model_path: Path) -> Iterator[Path]:
    """Expose a temporary config overlay without copying or modifying model weights."""

    config = json.loads((model_path / "config.json").read_text())
    with tempfile.TemporaryDirectory(prefix="breeze-eager-") as temporary:
        overlay = Path(temporary)
        for child in model_path.iterdir():
            if child.name != "config.json":
                os.symlink(child, overlay / child.name, target_is_directory=child.is_dir())
        (overlay / "config.json").write_text(
            json.dumps(_eager_attention_config(config), indent=2) + "\n"
        )
        yield overlay


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-repo", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    repository = args.official_repo.expanduser().resolve()
    source_revision = _source_revision(repository)
    model_path = args.model.expanduser().resolve()
    model_config = json.loads((model_path / "config.json").read_text())
    inherited_revision = (model_config.get("mlx_breeze_tts") or {}).get(
        "upstream_revision"
    )
    if inherited_revision != "c1c8ca18b70b30822735633991d9ebf4898e47d4":
        raise RuntimeError("PyTorch checkpoint does not inherit the approved revision")
    sys.path.insert(0, str(repository))
    import torch
    from breeze_infer.runtime import load_runtime, set_all_seeds
    from breeze_infer.templates import get_template, prepare_inputs

    set_all_seeds(42)
    with _eager_checkpoint_overlay(model_path) as runtime_model_path:
        tokenizer, model, audio_tokenizer = load_runtime(
            runtime_model_path,
            device=args.device,
            attn_implementation="eager",
        )
    request = {
        "id": "parity",
        "speaker": "S0",
        "ref_audio_path": str(args.ref_audio.expanduser().resolve()),
        "ref_text": args.ref_text,
        "text": args.text,
        "instruction": args.instruction,
    }
    template = get_template("ref_edit_tata")
    inputs = prepare_inputs(
        tokenizer,
        audio_tokenizer,
        model,
        [request],
        template,
        guidance_scale=1.0,
        guidance_scale_ref=None,
        guidance_scale_ins=None,
    )
    with torch.inference_mode():
        merged = model._merge_input_ids_with_input_values(
            inputs["input_ids"],
            inputs["input_values"],
            text_ids_mask=inputs["text_ids_mask"],
            text_ids_len=inputs["text_ids_len"],
            attention_mask=inputs["attention_mask"],
        )
        prompt = merged["inputs_embeds"]
        hidden = model.backbone_model(
            inputs_embeds=prompt,
            attention_mask=inputs["attention_mask"],
            use_cache=False,
        )[0]
        logits = model.lm_head(hidden)
        codec_config = model.config.codec_config
        if isinstance(codec_config, dict):
            codec_size = int(codec_config.get("codebook_size", 2048))
        else:
            codec_size = int(getattr(codec_config, "codebook_size", 2048))
        masked = logits[:, -1, :].clone()
        masked[:, codec_size : int(model.config.vocab_size)] = -torch.inf
        first_token = int(torch.argmax(masked, dim=-1).item())

    codes = inputs["input_values"]
    frame_count = int(codes.shape[1])
    segments = template.build_segments(request)
    template_render = {
        "template": template.name,
        "segments": [
            {"type": "text", "text": segments[0]["text"]},
            {"type": "audio", "frames": frame_count, "append_eos": True},
            {"type": "text", "text": segments[2]["text"]},
        ],
    }
    parameters = dict(model.named_parameters())
    weight_shapes = {name: list(parameters[name].shape) for name in WEIGHT_NAMES}
    snapshot = build_snapshot(
        runtime="pytorch",
        template_render=template_render,
        token_ids={"input_ids": tensor_values(inputs["input_ids"])},
        reference_audio_codes={"shape": list(codes.shape), "values": tensor_values(codes)},
        masks={
            "attention_mask": tensor_values(inputs["attention_mask"]),
            "text_ids_mask": tensor_values(inputs["text_ids_mask"]),
            "text_ids_len": tensor_values(inputs["text_ids_len"]),
        },
        weight_shapes=weight_shapes,
        deterministic_tokens={"backbone_argmax": [first_token]},
        intermediate_tensors={
            "prompt_first_32": tensor_values(prompt[0, 0, :32]),
            "prompt_last_32": tensor_values(prompt[0, -1, :32]),
            "backbone_last_32": tensor_values(hidden[0, -1, :32]),
            "logits_last_64": tensor_values(logits[0, -1, :64]),
        },
        runtime_provenance={
            "official_source": "https://github.com/breezeblue-ai/breeze-tts",
            "official_source_revision": source_revision,
            "device": args.device,
            "torch_version": torch.__version__,
            "checkpoint": str(model_path),
            "attention_implementation": "eager",
            "text_encoder_attention_implementation": "eager",
            "attention_override_reason": (
                "FlashAttention2 is unavailable on Apple Silicon; official API "
                "supports eager attention"
            ),
            "reference_audio_sha256": file_sha256(args.ref_audio),
        },
    )
    print(write_snapshot(snapshot, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
