import json

from mlx_breeze_tts.provenance import (
    checkpoint_provenance,
    inherited_upstream_identity,
    snapshot_revision,
)


def test_hugging_face_snapshot_revision_is_resolved_from_path(tmp_path):
    revision = "a" * 40
    snapshot = tmp_path / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text(json.dumps({"torch_dtype": "bfloat16"}))
    assert snapshot_revision(snapshot) == revision
    provenance = checkpoint_provenance(snapshot)
    assert provenance["artifact_revision"] == revision
    assert provenance["dtype"] == "bfloat16"


def test_quantized_conversion_inherits_original_upstream_identity(tmp_path):
    source = tmp_path / "bf16"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "mlx_breeze_tts": {
                    "source": "local-intermediate",
                    "upstream_source": "BreezeBlue/Breeze-TTS-2",
                    "upstream_revision": "b" * 40,
                    "dtype": "bfloat16",
                }
            }
        )
    )
    assert inherited_upstream_identity(source, source, None) == (
        "BreezeBlue/Breeze-TTS-2",
        "b" * 40,
    )


def test_requested_revision_wins_when_source_has_no_upstream_metadata(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "config.json").write_text("{}")
    assert inherited_upstream_identity("org/model", source, "c" * 40) == (
        "org/model",
        "c" * 40,
    )
