import pytest

from mlx_breeze_tts.parity_capture import (
    WEIGHT_NAMES,
    build_snapshot,
    file_sha256,
    tensor_values,
)


def _snapshot_kwargs():
    return {
        "runtime": "mlx",
        "template_render": {"segments": []},
        "token_ids": {"input_ids": [[1]]},
        "reference_audio_codes": {"shape": [1, 1, 16], "values": [[[0] * 16]]},
        "masks": {"attention_mask": [[1]]},
        "weight_shapes": {name: [1] for name in WEIGHT_NAMES},
        "deterministic_tokens": {"backbone_argmax": [1]},
        "intermediate_tensors": {"hidden": [0.25]},
        "runtime_provenance": {"checkpoint": "fixture"},
    }


def test_capture_document_is_revision_pinned_and_complete():
    snapshot = build_snapshot(**_snapshot_kwargs())
    assert snapshot["schema_version"] == 1
    assert len(snapshot["model_revision"]) == 40
    assert snapshot["case_id"] == "reference-direction-prefill-v1"


def test_capture_rejects_missing_weights_or_nonfinite_tensors():
    values = _snapshot_kwargs()
    values["weight_shapes"] = {}
    with pytest.raises(ValueError, match="canonical weight"):
        build_snapshot(**values)
    with pytest.raises(ValueError, match="non-finite"):
        tensor_values([float("nan")])


def test_capture_hashes_reference_bytes(tmp_path):
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    assert len(file_sha256(reference)) == 64
