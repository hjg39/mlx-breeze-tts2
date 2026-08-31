#!/usr/bin/env python3
"""Capture the matching standalone MLX prefill snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from mlx_breeze_tts.parity_capture import (
    MODEL_REVISION,
    WEIGHT_NAMES,
    build_snapshot,
    file_sha256,
    tensor_values,
    write_snapshot,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--ref-audio", type=Path, required=True)
    parser.add_argument("--ref-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mlx.core as mx
    from mlx.utils import tree_flatten

    from mlx_breeze_tts.loader import load
    from mlx_breeze_tts.provenance import read_checkpoint_provenance

    model_path = args.model.expanduser().resolve()
    provenance = read_checkpoint_provenance(model_path)
    if provenance.get("upstream_revision") != MODEL_REVISION:
        raise RuntimeError("MLX checkpoint does not inherit the approved revision")
    model = load(model_path)
    speaker = model._speaker("S0")
    ref_codes = model._encode_reference(args.ref_audio.expanduser().resolve())
    ref_render = model.tokenizer.decode(
        model.tokenizer(f"{speaker}{args.ref_text}", add_special_tokens=True)["input_ids"],
        skip_special_tokens=False,
    )
    target_text = f"{speaker}<ins_bos>{args.instruction}<ins_eos>{args.text}"
    target_render = model.tokenizer.decode(
        model.tokenizer(target_text, add_special_tokens=True)["input_ids"],
        skip_special_tokens=False,
    )
    frame_count = int(ref_codes.shape[1])
    audio_render = "<|AUDIO|>" * frame_count + "<|audio_eos|>"
    rendered = (ref_render, audio_render, target_render)
    final_text = "".join(rendered)
    input_ids = model.tokenizer(
        final_text, add_special_tokens=False, return_tensors="np"
    )["input_ids"]
    lengths = [
        len(model.tokenizer(value, add_special_tokens=False)["input_ids"])
        for value in rendered
    ]
    text_mask = [True] * lengths[0] + [False] * lengths[1] + [True] * lengths[2]
    attention_mask = [[1] * len(text_mask)]
    prompt = model._prompt_embeddings(
        args.text,
        voice="S0",
        instruct=args.instruction,
        ref_audio=args.ref_audio.expanduser().resolve(),
        ref_text=args.ref_text,
    )
    if prompt.shape[1] != len(text_mask):
        raise RuntimeError(
            f"MLX prompt length {prompt.shape[1]} differs from template length {len(text_mask)}"
        )
    hidden = model.backbone_model(input_embeddings=prompt)
    logits = model.lm_head(hidden)
    masked = model._mask_reserved_codec_logits(logits[:, -1, :])
    first_token = int(mx.argmax(masked, axis=-1).item())
    mx.eval(prompt, hidden, logits, ref_codes)
    parameters = dict(tree_flatten(model.parameters()))
    weight_shapes = {name: list(parameters[name].shape) for name in WEIGHT_NAMES}
    snapshot = build_snapshot(
        runtime="mlx",
        template_render={
            "template": "ref_edit_tata",
            "segments": [
                {"type": "text", "text": f"{speaker}{args.ref_text}"},
                {"type": "audio", "frames": frame_count, "append_eos": True},
                {"type": "text", "text": target_text},
            ],
        },
        token_ids={"input_ids": tensor_values(input_ids)},
        reference_audio_codes={"shape": list(ref_codes.shape), "values": tensor_values(ref_codes)},
        masks={
            "attention_mask": attention_mask,
            "text_ids_mask": [text_mask],
            "text_ids_len": [lengths[0], lengths[2]],
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
            "checkpoint": str(model_path),
            "checkpoint_provenance": provenance,
            "mlx_version": getattr(mx, "__version__", "unknown"),
            "reference_audio_sha256": file_sha256(args.ref_audio),
        },
    )
    print(write_snapshot(snapshot, args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
