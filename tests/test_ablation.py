import hashlib
import json

from mlx_breeze_tts.ablation import (
    apply_quantization_evidence,
    compare_quantization_candidates,
)
from mlx_breeze_tts.parity import compare_parity_snapshots

EVENTS = [
    f"event_{language}_{event}"
    for language in ("en", "zh")
    for event in ("laugh", "cough", "clears_throat", "sigh")
]


def _summary(policy, size, *, cer=0.01):
    return {
        "model_revision": "a" * 40,
        "model_provenance": {
            "bits": 4,
            "artifact_bytes": size,
            "quantization_policy": {"name": policy, "version": 1},
        },
        "artifact_audit": {"pass": True},
        "interfaces": {
            "python": "pass",
            "cli": "pass",
            "http": "pass",
            "streaming": "pass",
        },
        "performance": {
            "load_time_s": 1.0,
            "load_peak_memory_gb": 2.0,
            "peak_memory_gb": 3.0,
            "steady_state_rtf": 1.2,
            "long_text_rtf": 1.3,
            "streaming_ttfa_s": 0.5,
        },
        "validation": {
            "max_cer": cer,
            "corpus_cer": cer,
            "clone_cosine_min": 0.5,
            "clone_p10_min": 0.4,
            "leakage_max": 0.2,
            "seed_reproducibility": "pass",
            "sampling_path": "pass",
            "waveform_integrity": "pass",
            "pytorch_parity": "pass",
        },
        "samples": [
            {"capability": capability, "manual_event": "audible"}
            for capability in EVENTS
        ],
    }


def _snapshot():
    return {
        "schema_version": 1,
        "model_id": "BreezeBlue/Breeze-TTS-2",
        "model_revision": "a" * 40,
        "case_id": "tiny-fixed-input-v1",
        "template_render": {},
        "token_ids": {},
        "reference_audio_codes": {},
        "masks": {},
        "weight_shapes": {},
        "deterministic_tokens": {},
        "intermediate_tensors": {"hidden": [1.0]},
    }


def _write_candidate(tmp_path, name, payload, *, parity_fail=False):
    pytorch = tmp_path / f"{name}-pytorch.json"
    mlx = tmp_path / f"{name}-mlx.json"
    evidence = tmp_path / f"{name}-parity.json"
    pytorch.write_text(json.dumps(_snapshot()))
    candidate = _snapshot()
    if parity_fail:
        candidate["weight_shapes"] = {"different": [1]}
    mlx.write_text(json.dumps(candidate))
    compare_parity_snapshots(pytorch, mlx, evidence)
    payload["validation"]["pytorch_parity"] = (
        "fail" if parity_fail else "pass"
    )
    payload["validation"]["pytorch_parity_evidence"] = {
        "path": str(evidence),
        "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(payload))
    return path


def test_ablation_selects_smallest_passing_candidate_and_merges(tmp_path):
    full = _write_candidate(tmp_path, "full", _summary("full", 100))
    sensitive = _write_candidate(
        tmp_path, "sensitive", _summary("sensitive-bf16", 120)
    )
    evidence = tmp_path / "ablation.json"
    merged = tmp_path / "selected.json"
    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    assert report["selected_policy"] == "full"
    apply_quantization_evidence(full, evidence, merged)
    result = json.loads(merged.read_text())
    assert result["validation"]["quantization_ablation"] == "pass"


def test_ablation_rejects_smaller_candidate_when_quality_fails(tmp_path):
    full = _write_candidate(tmp_path, "full", _summary("full", 100, cer=0.2))
    sensitive = _write_candidate(
        tmp_path, "sensitive", _summary("sensitive-bf16", 120)
    )
    evidence = tmp_path / "ablation.json"
    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    assert report["selected_policy"] == "sensitive-bf16"
    assert report["candidates"][0]["pass"] is False


def test_ablation_fails_when_neither_candidate_passes(tmp_path):
    full = _write_candidate(tmp_path, "full", _summary("full", 100, cer=0.2))
    sensitive = _write_candidate(
        tmp_path, "sensitive", _summary("sensitive-bf16", 120, cer=0.3)
    )
    evidence = tmp_path / "ablation.json"
    compare_quantization_candidates(full, sensitive, evidence)
    assert json.loads(evidence.read_text())["pass"] is False


def test_ablation_does_not_reject_a_candidate_only_for_missing_evidence(tmp_path):
    full = tmp_path / "full.json"
    full.write_text(json.dumps(_summary("full", 100)))
    sensitive = _write_candidate(
        tmp_path, "sensitive", _summary("sensitive-bf16", 120)
    )
    evidence = tmp_path / "ablation.json"

    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is False
    assert report["selected_policy"] is None
    assert report["candidates"][0]["status"] == "pending"
    assert "candidate evidence is incomplete" in report["issues"]


def test_ablation_can_stop_a_candidate_on_hash_bound_parity_failure(tmp_path):
    full = _write_candidate(
        tmp_path, "full", _summary("full", 100), parity_fail=True
    )
    sensitive = _write_candidate(
        tmp_path, "sensitive", _summary("sensitive-bf16", 120)
    )
    evidence = tmp_path / "ablation.json"

    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    assert report["selected_policy"] == "sensitive-bf16"
    assert report["candidates"][0]["status"] == "fail"
