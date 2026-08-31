import json

import pytest

from mlx_breeze_tts.http_evidence import apply_http_evidence, http_report_passes


def _evidence(revision: str) -> dict:
    checks = {
        name: True
        for name in (
            "health_200",
            "health_sample_rate",
            "speech_200",
            "content_type_pcm",
            "sample_rate_header",
            "sample_format_header",
            "cache_control",
            "nonempty_even_pcm",
        )
    }
    return {
        "schema_version": 1,
        "model_revision": revision,
        "probe": {"status": "pass", "speech_status_code": 200, "checks": checks},
    }


def test_http_evidence_is_revision_bound_and_non_destructive(tmp_path):
    revision = "a" * 40
    summary = tmp_path / "summary.json"
    evidence = tmp_path / "http.json"
    output = tmp_path / "merged.json"
    summary.write_text(
        json.dumps(
            {
                "model_revision": revision,
                "interfaces": {"http": "fail"},
                "validation": {},
            }
        )
    )
    evidence.write_text(json.dumps(_evidence(revision)))

    apply_http_evidence(summary, evidence, output)
    merged = json.loads(output.read_text())
    assert merged["interfaces"]["http"] == "pass"
    assert len(merged["validation"]["http_evidence"]["sha256"]) == 64
    assert json.loads(summary.read_text())["interfaces"]["http"] == "fail"


def test_http_evidence_rejects_failure_or_different_revision(tmp_path):
    revision = "a" * 40
    assert http_report_passes(_evidence(revision), revision)
    assert not http_report_passes(_evidence("b" * 40), revision)
    failed = _evidence(revision)
    failed["probe"]["checks"]["speech_200"] = False
    assert not http_report_passes(failed, revision)

    summary = tmp_path / "summary.json"
    evidence = tmp_path / "http.json"
    summary.write_text(json.dumps({"model_revision": revision}))
    evidence.write_text(json.dumps(_evidence("b" * 40)))
    with pytest.raises(ValueError, match="matching passing"):
        apply_http_evidence(summary, evidence, tmp_path / "output.json")
