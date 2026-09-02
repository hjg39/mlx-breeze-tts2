import math

import pytest

from mlx_breeze_tts.validation import validate_generation_args


def _valid(**overrides):
    values = {
        "text": "hello",
        "voice": "S0",
        "instruct": None,
        "cfg_scale": 1.0,
        "max_tokens": 1500,
        "temperature": 0.9,
        "top_p": 1.0,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "seed": 42,
        "stream": False,
        "streaming_interval": 2.0,
        "fast_depth": True,
    }
    values.update(overrides)
    return values


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("text", "  "),
        ("voice", 1),
        ("instruct", []),
        ("cfg_scale", math.nan),
        ("cfg_scale", 0),
        ("max_tokens", 1.5),
        ("max_tokens", True),
        ("max_tokens", 0),
        ("temperature", math.inf),
        ("temperature", -0.1),
        ("top_p", math.nan),
        ("top_p", 1.1),
        ("top_k", 2.5),
        ("top_k", -1),
        ("repetition_penalty", math.nan),
        ("repetition_penalty", 0),
        ("seed", True),
        ("seed", -1),
        ("stream", 1),
        ("fast_depth", 1),
        ("streaming_interval", math.inf),
        ("streaming_interval", 0),
    ],
)
def test_generation_arguments_fail_fast(field, value):
    with pytest.raises(ValueError, match=field):
        validate_generation_args(**_valid(**{field: value}))


def test_generation_argument_boundaries_pass():
    validate_generation_args(
        **_valid(
            cfg_scale=None,
            temperature=0.0,
            top_p=0.0,
            top_k=0,
            seed=None,
            stream=True,
        )
    )
