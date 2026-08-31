"""Deterministic, auditable quantization policies for Breeze modules."""

from __future__ import annotations

POLICIES = ("full", "sensitive-bf16")
POLICY_VERSION = 1

SENSITIVE_BF16_PREFIXES = (
    "lm_head",
    "embed_text_tokens",
    "text_encoder.embed_tokens",
    "text_encoder_proj",
    "backbone_model.embed_tokens.embed_audio_tokens",
    "backbone_model.embed_tokens.audio_embeds_projector",
    "depth_decoder.model.embed_tokens",
    "depth_decoder.model.backbone_hidden_state_projector",
    "depth_decoder.model.inputs_embeds_projector",
)


def validate_policy(policy: str) -> str:
    if policy not in POLICIES:
        raise ValueError(f"quantization policy must be one of: {', '.join(POLICIES)}")
    return policy


def should_quantize_path(path: str, policy: str) -> bool:
    validate_policy(policy)
    if policy == "full":
        return True
    return not any(
        path == prefix or path.startswith(prefix + ".")
        for prefix in SENSITIVE_BF16_PREFIXES
    )
