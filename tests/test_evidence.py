import hashlib
import json
from pathlib import Path

from mlx_breeze_tts.evidence import (
    REQUIRED_CAPABILITIES,
    REQUIRED_EVENTS,
    REQUIRED_VARIANTS,
    verify_evidence_bundle,
)


def test_empty_bundle_fails_all_variants(tmp_path):
    report = verify_evidence_bundle(tmp_path)
    assert not report["pass"]
    assert report["issue_count"] == 3
    assert {item["variant"] for item in report["issues"]} == set(REQUIRED_VARIANTS)


def test_pending_numeric_fields_fail_closed_instead_of_crashing(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
        path = tmp_path / variant / "summary.json"
        summary = json.loads(path.read_text())
        summary["validation"]["corpus_cer"] = None
        summary["validation"]["clone_cosine_min"] = "pending"
        path.write_text(json.dumps(summary))
    report = verify_evidence_bundle(tmp_path)
    assert report["pass"] is False
    assert any(item["gate"] == "content" for item in report["issues"])
    assert any(item["gate"] == "speaker" for item in report["issues"])


def test_missing_parity_waveform_and_performance_fail_closed(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
    path = tmp_path / "bf16/summary.json"
    summary = json.loads(path.read_text())
    summary["validation"].pop("pytorch_parity")
    summary["validation"].pop("waveform_integrity")
    summary["performance"].pop("streaming_ttfa_s")
    path.write_text(json.dumps(summary))

    report = verify_evidence_bundle(tmp_path)
    gates = {item["gate"] for item in report["issues"] if item["variant"] == "bf16"}
    assert {"pytorch_parity", "waveform_integrity", "performance"} <= gates


def _write_complete_variant(root: Path, variant: str):
    directory = root / variant
    directory.mkdir()
    capabilities = REQUIRED_CAPABILITIES | REQUIRED_EVENTS
    samples = [
        {
            "capability": capability,
            "samples": 100,
            "clipping_fraction": 0.0,
            "manual_event": "audible" if capability in REQUIRED_EVENTS else None,
        }
        for capability in capabilities
    ]
    parity = directory / "parity.json"
    checks = {
        name: {"status": "pass"}
        for name in (
            "model_id",
            "model_revision",
            "case_id",
            "template_render",
            "token_ids",
            "reference_audio_codes",
            "masks",
            "weight_shapes",
            "deterministic_tokens",
            "intermediate_tensors",
        )
    }
    parity.write_text(json.dumps({"schema_version": 1, "pass": True, "checks": checks}))
    parity_sha256 = hashlib.sha256(parity.read_bytes()).hexdigest()
    reviews = directory / "manual_reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "exported_at": "2026-09-01T00:00:00Z",
                "model_revision": "a" * 40,
                "reviews": [
                    {"capability": capability, "manual_event": "audible"}
                    for capability in sorted(REQUIRED_EVENTS)
                ],
            }
        )
    )
    bits = None if variant == "bf16" else (8 if variant == "8bit" else 4)
    model_provenance = {"artifact_revision": "a" * 40, "bits": bits}
    quantization_validation = {}
    if bits is not None:
        model_provenance["quantization_policy"] = {"name": "full", "version": 1}
        ablation = directory / "quantization-ablation.json"
        ablation.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "pass": True,
                    "selected_policy": "full",
                    "bits": bits,
                    "model_revision": "a" * 40,
                }
            )
        )
        quantization_validation = {
            "quantization_ablation": "pass",
            "quantization_ablation_evidence": {
                "path": str(ablation),
                "sha256": hashlib.sha256(ablation.read_bytes()).hexdigest(),
                "selected_policy": "full",
            },
        }
    summary = {
        "model_revision": "a" * 40,
        "model_provenance": model_provenance,
        "artifact_audit": {
            "pass": True,
            "missing": [],
            "unexpected": [],
            "shape_mismatches": {},
        },
        "interfaces": {name: "pass" for name in ("python", "cli", "http", "streaming")},
        "samples": samples,
        "validation": {
            "max_cer": 0.05,
            "corpus_cer": 0.05,
            "clone_cosine_min": 0.25,
            "clone_p10_min": 0.25,
            "leakage_max": 0.64,
            "seed_reproducibility": "pass",
            "sampling_path": "pass",
            "waveform_integrity": "pass",
            "pytorch_parity": "pass",
            "pytorch_parity_evidence": {
                "path": str(parity),
                "sha256": parity_sha256,
            },
            "manual_listening_evidence": {
                "path": str(reviews),
                "sha256": hashlib.sha256(reviews.read_bytes()).hexdigest(),
                "review_count": len(REQUIRED_EVENTS),
                "reviewed_capabilities": sorted(REQUIRED_EVENTS),
            },
            **quantization_validation,
        },
        "performance": {
            "load_time_s": 1.0,
            "load_peak_memory_gb": 2.0,
            "peak_memory_gb": 3.0,
            "steady_state_rtf": 1.5,
            "long_text_rtf": 1.8,
            "streaming_ttfa_s": 0.5,
        },
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    (directory / "report.md").write_text("pass")
    (directory / "index.html").write_text("pass")
    (directory / "sample.wav").write_bytes(b"RIFF")


def test_complete_synthetic_bundle_satisfies_schema(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
    report = verify_evidence_bundle(tmp_path)
    assert report["pass"]
    assert report["issues"] == []


def test_mutable_or_missing_model_revision_fails_provenance_gate(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
    path = tmp_path / "4bit/summary.json"
    summary = json.loads(path.read_text())
    summary["model_revision"] = "main"
    path.write_text(json.dumps(summary))
    report = verify_evidence_bundle(tmp_path)
    assert any(
        item["variant"] == "4bit" and item["gate"] == "provenance"
        for item in report["issues"]
    )


def test_reviewed_summary_takes_precedence_over_raw_generation(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
    reviewed = json.loads((tmp_path / "4bit/summary.json").read_text())
    reviewed["samples"][0]["samples"] = 0
    (tmp_path / "4bit/summary.reviewed.json").write_text(json.dumps(reviewed))
    report = verify_evidence_bundle(tmp_path)
    assert report["pass"] is False
    assert any("summary.reviewed.json" in item for item in report["checked"])


def test_tampered_manual_review_fails_hash_gate(tmp_path):
    for variant in REQUIRED_VARIANTS:
        _write_complete_variant(tmp_path, variant)
    reviews = tmp_path / "bf16/manual_reviews.json"
    reviews.write_text(reviews.read_text() + "\n")

    report = verify_evidence_bundle(tmp_path)
    assert any(
        item["variant"] == "bf16" and item["gate"] == "manual_listening_evidence"
        for item in report["issues"]
    )
