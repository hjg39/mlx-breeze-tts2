import json

import pytest

from mlx_breeze_tts.parity import apply_parity_evidence, compare_parity_snapshots


def _snapshot(offset=0.0):
    return {
        "schema_version": 1,
        "model_id": "BreezeBlue/Breeze-TTS-2",
        "model_revision": "pinned",
        "case_id": "tiny-fixed-input-v1",
        "template_render": {"clone": "[S0]reference"},
        "token_ids": {"clone": [1, 2, 3]},
        "reference_audio_codes": {"shape": [1, 16, 2], "values": [4, 5]},
        "masks": {"full": [[True, True]]},
        "weight_shapes": {"lm_head.weight": [8, 4]},
        "deterministic_tokens": {"temperature_0": [7, 6]},
        "intermediate_tensors": {"text_hidden": [[1.0 + offset, 2.0]]},
    }


def test_parity_comparison_and_non_destructive_merge(tmp_path):
    pytorch = tmp_path / "pytorch.json"
    mlx = tmp_path / "mlx.json"
    evidence = tmp_path / "parity.json"
    summary = tmp_path / "summary.json"
    merged = tmp_path / "summary.parity.json"
    pytorch.write_text(json.dumps(_snapshot()))
    mlx.write_text(json.dumps(_snapshot(offset=1e-5)))
    summary.write_text(json.dumps({"validation": {"max_cer": 0.0}}))

    compare_parity_snapshots(pytorch, mlx, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    apply_parity_evidence(summary, evidence, merged)
    result = json.loads(merged.read_text())
    assert result["validation"]["pytorch_parity"] == "pass"
    assert json.loads(summary.read_text())["validation"].get("pytorch_parity") is None


def test_parity_comparison_fails_on_numeric_or_exact_mismatch(tmp_path):
    pytorch = tmp_path / "pytorch.json"
    mlx = tmp_path / "mlx.json"
    evidence = tmp_path / "parity.json"
    pytorch.write_text(json.dumps(_snapshot()))
    changed = _snapshot(offset=1.0)
    changed["token_ids"]["clone"] = [9]
    mlx.write_text(json.dumps(changed))

    compare_parity_snapshots(pytorch, mlx, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is False
    assert report["checks"]["token_ids"]["status"] == "fail"
    assert report["checks"]["intermediate_tensors"]["status"] == "fail"
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"validation": {}}))
    output = tmp_path / "out.json"
    apply_parity_evidence(summary, evidence, output)
    merged = json.loads(output.read_text())
    assert merged["validation"]["pytorch_parity"] == "fail"
    assert merged["validation"]["pytorch_parity_evidence"]["status"] == "fail"

    pytorch.write_text("tampered")
    with pytest.raises(ValueError, match="incomplete"):
        apply_parity_evidence(summary, evidence, tmp_path / "tampered.json")


def test_parity_rejects_invalid_tolerances(tmp_path):
    with pytest.raises(ValueError, match="finite"):
        compare_parity_snapshots("a", "b", tmp_path / "out.json", atol=-1)
