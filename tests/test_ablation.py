import json

from mlx_breeze_tts.ablation import (
    apply_quantization_evidence,
    compare_quantization_candidates,
)


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


def test_ablation_selects_smallest_passing_candidate_and_merges(tmp_path):
    full = tmp_path / "full.json"
    sensitive = tmp_path / "sensitive.json"
    evidence = tmp_path / "ablation.json"
    merged = tmp_path / "selected.json"
    full.write_text(json.dumps(_summary("full", 100)))
    sensitive.write_text(json.dumps(_summary("sensitive-bf16", 120)))
    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    assert report["selected_policy"] == "full"
    apply_quantization_evidence(full, evidence, merged)
    result = json.loads(merged.read_text())
    assert result["validation"]["quantization_ablation"] == "pass"


def test_ablation_rejects_smaller_candidate_when_quality_fails(tmp_path):
    full = tmp_path / "full.json"
    sensitive = tmp_path / "sensitive.json"
    evidence = tmp_path / "ablation.json"
    full.write_text(json.dumps(_summary("full", 100, cer=0.2)))
    sensitive.write_text(json.dumps(_summary("sensitive-bf16", 120)))
    compare_quantization_candidates(full, sensitive, evidence)
    report = json.loads(evidence.read_text())
    assert report["pass"] is True
    assert report["selected_policy"] == "sensitive-bf16"
    assert report["candidates"][0]["pass"] is False


def test_ablation_fails_when_neither_candidate_passes(tmp_path):
    full = tmp_path / "full.json"
    sensitive = tmp_path / "sensitive.json"
    evidence = tmp_path / "ablation.json"
    full.write_text(json.dumps(_summary("full", 100, cer=0.2)))
    sensitive.write_text(json.dumps(_summary("sensitive-bf16", 120, cer=0.3)))
    compare_quantization_candidates(full, sensitive, evidence)
    assert json.loads(evidence.read_text())["pass"] is False
