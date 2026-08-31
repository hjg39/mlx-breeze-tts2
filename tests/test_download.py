import json
from pathlib import Path

import pytest

from mlx_breeze_tts.download import (
    OFFICIAL_MODEL,
    OFFICIAL_REVISION,
    download_official_checkpoint,
)
from mlx_breeze_tts import cli


def _snapshot(tmp_path: Path, revision: str = OFFICIAL_REVISION) -> Path:
    snapshot = tmp_path / "models--BreezeBlue--Breeze-TTS-2" / "snapshots" / revision
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"weights")
    return snapshot


def test_download_is_pinned_validated_and_reported(tmp_path):
    snapshot = _snapshot(tmp_path)
    calls = []

    def downloader(model, **kwargs):
        calls.append((model, kwargs))
        return str(snapshot)

    resolved, report_path = download_official_checkpoint(
        tmp_path / "download.json", downloader=downloader
    )
    report = json.loads(report_path.read_text())
    assert resolved == snapshot
    assert calls[0][0] == OFFICIAL_MODEL
    assert calls[0][1]["revision"] == OFFICIAL_REVISION
    assert report["status"] == "pass"
    assert report["resolved_revision"] == OFFICIAL_REVISION
    assert report["artifact_bytes"] == 9


def test_download_rejects_wrong_or_incomplete_snapshot(tmp_path):
    wrong = _snapshot(tmp_path, "a" * 40)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        download_official_checkpoint(downloader=lambda *_args, **_kwargs: str(wrong))

    incomplete = tmp_path / "models--BreezeBlue--Breeze-TTS-2" / "snapshots" / OFFICIAL_REVISION
    incomplete.mkdir(parents=True, exist_ok=True)
    (incomplete / "config.json").write_text("{}")
    (incomplete / "model.safetensors").unlink(missing_ok=True)
    with pytest.raises(RuntimeError, match="missing safetensors"):
        download_official_checkpoint(
            downloader=lambda *_args, **_kwargs: str(incomplete)
        )


def test_download_cli_reports_failure_without_traceback(monkeypatch, capsys):
    def fail(_output):
        raise RuntimeError("network unavailable")

    monkeypatch.setattr(
        "mlx_breeze_tts.download.download_official_checkpoint",
        fail,
    )
    assert cli.main(["download-official"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload == {"status": "fail", "error": "network unavailable"}
