"""Deterministic, auditable quantization policies for Breeze modules."""

from __future__ import annotations

POLICIES = ("full", "sensitive-bf16")
POLICY_VERSION = 2

SENSITIVE_BF16_PREFIXES = (
    "lm_head",
    "embed_text_tokens",
    "text_encoder",
    "text_encoder_proj",
    "backbone_model.embed_tokens.embed_audio_tokens",
    "backbone_model.embed_tokens.audio_embeds_projector",
    "depth_decoder",
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
