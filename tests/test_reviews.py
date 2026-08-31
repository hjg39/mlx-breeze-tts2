import json

import pytest

from mlx_breeze_tts.reviews import apply_reviews


def _summary(path):
    path.write_text(
        json.dumps(
            {
                "samples": [
                    {"capability": "event_en_laugh"},
                    {"capability": "voice_design_en"},
                ],
                "validation": {"manual_listening": "pending"},
            }
        )
    )


def test_review_merge_is_validated_and_writes_a_new_summary(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": None,
                "exported_at": "2026-09-01T00:00:00Z",
                "reviews": [
                    {
                        "capability": "event_en_laugh",
                        "manual_content": "pass",
                        "manual_voice": "n/a",
                        "manual_event": "audible",
                        "manual_notes": "clearly audible",
                    },
                    {
                        "capability": "voice_design_en",
                        "manual_content": "pass",
                        "manual_voice": "pass",
                        "manual_event": "n/a",
                    },
                ],
            }
        )
    )
    output = apply_reviews(summary, reviews, tmp_path / "reviewed.json")
    merged = json.loads(output.read_text())
    assert merged["validation"]["manual_listening"] == "reviewed"
    assert merged["samples"][0]["manual_event"] == "audible"
    evidence = merged["validation"]["manual_listening_evidence"]
    assert len(evidence["sha256"]) == 64
    assert evidence["review_count"] == 2
    assert "manual_event" not in json.loads(summary.read_text())["samples"][0]


def test_review_merge_rejects_unknown_duplicate_and_in_place_output(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "reviews": [
                    {"capability": "unknown"},
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="Unknown"):
        apply_reviews(summary, reviews, tmp_path / "out.json")
    with pytest.raises(ValueError, match="differ"):
        apply_reviews(summary, reviews, summary)


def test_review_merge_rejects_different_model_revision(tmp_path):
    summary = tmp_path / "summary.json"
    _summary(summary)
    payload = json.loads(summary.read_text())
    payload["model_revision"] = "a" * 40
    summary.write_text(json.dumps(payload))
    reviews = tmp_path / "reviews.json"
    reviews.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "model_revision": "b" * 40,
                "reviews": [],
            }
        )
    )

    with pytest.raises(ValueError, match="model_revision"):
        apply_reviews(summary, reviews, tmp_path / "output.json")
