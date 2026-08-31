import pytest

from mlx_breeze_tts.cli import _parser
from mlx_breeze_tts.quantization import (
    SENSITIVE_BF16_PREFIXES,
    should_quantize_path,
    validate_policy,
)


def test_full_policy_quantizes_every_eligible_path():
    assert should_quantize_path("lm_head", "full") is True
    assert should_quantize_path("backbone_model.layers.0.mlp.up_proj", "full") is True


def test_sensitive_policy_preserves_declared_modules_only():
    for prefix in SENSITIVE_BF16_PREFIXES:
        assert should_quantize_path(prefix, "sensitive-bf16") is False
    assert (
        should_quantize_path(
            "backbone_model.layers.0.self_attn.q_proj", "sensitive-bf16"
        )
        is True
    )


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="policy"):
        validate_policy("unknown")


def test_convert_cli_exposes_reproducible_policy_choice():
    args = _parser().parse_args(
        [
            "convert",
            "--source",
            "bf16",
            "--output",
            "4bit",
            "--bits",
            "4",
            "--quantization-policy",
            "sensitive-bf16",
        ]
    )
    assert args.quantization_policy == "sensitive-bf16"
