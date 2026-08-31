import json

import pytest

from mlx_breeze_tts.objective import (
    apply_objective_metrics,
    character_error_rate,
    normalize_text,
    objective_template,
    reference_leakage_similarity,
)


def test_text_metrics_strip_events_and_match_skill_leakage_semantics():
    assert normalize_text("(laugh) Hello, 世界!", strip_events=True) == "hello世界"
    assert character_error_rate("[笑] 欢迎回来。", "欢迎回来") == 0.0
    assert character_error_rate("hello", "hallo") == pytest.approx(0.2)
    assert reference_leakage_similarity("Exact prompt", "exact prompt") == 1.0
    assert reference_leakage_similarity("Exact prompt", "different target") < 0.65


def _summary(path):
    samples = [
        {
            "capability": "voice_design_en",
            "text": "Hello world.",
            "expected_text": "Hello world.",
            "audio": "design.wav",
            "status": "audio_generated",
        },
        {
            "capability": "voice_clone_en",
            "text": "Target sentence.",
            "expected_text": "Target sentence.",
            "audio": "clone.wav",
            "status": "audio_generated",
            "ref_audio": "ref.wav",
            "ref_text": "Unrelated reference prompt.",
        },
    ]
    path.write_text(json.dumps({"samples": samples, "validation": {}}))


def test_template_resolves_audio_and_objective_merge_aggregates(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    template_path = objective_template(summary, tmp_path / "template.json")
    template = json.loads(template_path.read_text())
    assert template["rows"][0]["audio"] == str((tmp_path / "design.wav").resolve())
    for row in template["rows"]:
        row["asr_text"] = (
            "Hello world" if row["capability"] == "voice_design_en" else "Target sentence"
        )
        if row["capability"] == "voice_clone_en":
            row["speaker_cosine"] = 0.8
            row["speaker_p10"] = 0.7
            row["speaker_backend"] = "ecapa"
    template_path.write_text(json.dumps(template))
    output = apply_objective_metrics(
        summary, template_path, tmp_path / "summary.metrics.json"
    )
    merged = json.loads(output.read_text())
    validation = merged["validation"]
    assert validation["max_cer"] == 0.0
    assert validation["clone_cosine_min"] == 0.8
    assert validation["clone_p10_min"] == 0.7
    assert validation["leakage_max"] < 0.65
    assert validation["objective_metrics"] == "complete"


def test_missing_or_invalid_metrics_fail_closed(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    metrics = tmp_path / "metrics.json"
    metrics.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "rows": [
                    {"capability": "voice_design_en", "asr_text": "Hello world"},
                    {
                        "capability": "voice_clone_en",
                        "asr_text": "Target sentence",
                        "speaker_cosine": 1.1,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        apply_objective_metrics(summary, metrics, tmp_path / "output.json")
    document = json.loads(metrics.read_text())
    document["rows"][1]["speaker_cosine"] = None
    metrics.write_text(json.dumps(document))
    output = apply_objective_metrics(summary, metrics, tmp_path / "output.json")
    validation = json.loads(output.read_text())["validation"]
    assert validation["objective_metrics"] == "partial"
    assert validation["clone_cosine_min"] is None
