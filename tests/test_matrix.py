import pytest

from mlx_breeze_tts.evidence import REQUIRED_CAPABILITIES, REQUIRED_EVENTS
from mlx_breeze_tts.matrix import (
    ReferencePair,
    build_acceptance_matrix,
    matrix_coverage,
    reference_pair,
)


def test_acceptance_matrix_covers_every_completion_capability_once():
    en = ReferencePair("en.wav", "Exact English transcript.", "en")
    zh = ReferencePair("zh.wav", "精确的中文转写。", "zh")
    cases = build_acceptance_matrix(en, zh)
    coverage = matrix_coverage(cases)
    assert coverage["pass"] is True
    capabilities = {case["capability"] for case in cases}
    assert REQUIRED_CAPABILITIES | REQUIRED_EVENTS <= capabilities
    assert len(cases) == 23
    clears_throat = next(
        case for case in cases if case["capability"] == "event_zh_clears_throat"
    )
    assert clears_throat["seed"] == 1
    sigh = next(case for case in cases if case["capability"] == "event_en_sigh")
    assert sigh["seed"] == 3
    cough = next(case for case in cases if case["capability"] == "event_en_cough")
    assert cough["seed"] == 7


def test_missing_references_are_explicit_not_silently_omitted():
    cases = build_acceptance_matrix()
    clone_cases = [case for case in cases if "clone" in case["capability"]]
    assert len(clone_cases) == 4
    assert all(case["skip_reason"] for case in clone_cases)
    assert next(case for case in cases if case["capability"] == "voice_direction_en")[
        "skip_reason"
    ]


def test_reference_pair_must_be_exactly_paired_and_language_bounded():
    assert reference_pair(None, None, "en") is None
    with pytest.raises(ValueError, match="supplied together"):
        reference_pair("voice.wav", None, "en")
    with pytest.raises(ValueError, match="en or zh"):
        ReferencePair("voice.wav", "text", "fr")
