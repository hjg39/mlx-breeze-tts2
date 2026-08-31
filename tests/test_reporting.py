from mlx_breeze_tts.reporting import (
    benchmark_markdown,
    listening_html,
    render_report_bundle,
)


def _report():
    return {
        "created_at": "2026-08-31T00:00:00Z",
        "resolved_model_path": "/models/breeze",
        "status": "audio_generated_evaluation_pending",
        "samples": [
            {
                "capability": "voice_design_en",
                "text": "Hello <world>",
                "instruct": "calm & clear",
                "audio": "voice_design_en.wav",
                "duration_s": 2.0,
                "elapsed_s": 2.2,
                "rtf": 1.1,
                "clipping_fraction": 0.0,
                "peak_dbfs": -2.0,
                "rms_dbfs": -20.0,
                "status": "audio_generated",
                "asr": "pending",
                "speaker_similarity": "not_applicable",
            }
        ],
    }


def test_markdown_contains_metrics_and_pending_boundary():
    output = benchmark_markdown(_report())
    assert "voice_design_en" in output
    assert "1.10" in output
    assert "`pending`" in output
    assert "PyTorch parity" in output
    assert "Streaming TTFA" in output


def test_listening_html_has_audio_controls_and_escaped_content():
    output = listening_html(_report())
    assert '<audio controls preload="none" src="voice_design_en.wav">' in output
    assert "Hello &lt;world&gt;" in output
    assert "calm &amp; clear" in output
    assert "Manual listening" in output
    assert "Export manual_reviews.json" in output
    assert 'data-field="manual_event"' in output
    assert "Repeated tail" in output
    assert "Stream break" in output


def test_report_bundle_writes_markdown_and_listening_pages(tmp_path):
    output = render_report_bundle(_report(), tmp_path)
    assert (output / "report.md").is_file()
    assert (output / "index.html").is_file()
    assert (output / "report.html").read_text() == (output / "index.html").read_text()
