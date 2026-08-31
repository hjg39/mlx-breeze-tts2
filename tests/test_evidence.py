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
    summary = {
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
            "clone_cosine_min": 0.25,
            "clone_p10_min": 0.25,
            "leakage_max": 0.64,
            "seed_reproducibility": "pass",
            "sampling_path": "pass",
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
