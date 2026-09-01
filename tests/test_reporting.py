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
            },
            {
                "capability": "event_en_laugh",
                "text": "(laugh) Hello.",
                "instruct": None,
                "audio": "event_en_laugh.wav",
                "duration_s": 1.0,
                "elapsed_s": 1.0,
                "rtf": 1.0,
                "clipping_fraction": 0.0,
                "peak_dbfs": -2.0,
                "rms_dbfs": -20.0,
                "status": "audio_generated",
            },
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
    event_page = (output / "events.html").read_text()
    assert "event_en_laugh" in event_page
    assert "voice_design_en" not in event_page
    assert 'link.download="manual_reviews-' in event_page


def test_final_report_reflects_saved_event_verdict_and_release_status():
    report = _report()
    report["samples"][1]["manual_event"] = "audible"
    report["validation"] = {
        "objective_metrics": "complete",
        "pytorch_parity": "pass",
    }
    report["interfaces"] = {"http": "pass"}
    report["model_provenance"] = {"bits": None}

    markdown = benchmark_markdown(report)
    rendered = listening_html(report, event_only=True)
    assert "Overall status: `release_pass`" in markdown
    assert "Manual event listening: `pass`" in markdown
    assert '<option selected>audible</option>' in rendered
    assert "release_pass" in rendered
