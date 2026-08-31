"""Reproducible local capability benchmark with machine-readable evidence."""

import html
import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from .audio import write_audio
from .loader import load, resolve_model_path


def _audio_metrics(audio: mx.array, sample_rate: int) -> dict:
    values = np.asarray(audio, dtype=np.float32)
    rms = float(np.sqrt(np.mean(values**2))) if values.size else 0.0
    peak = float(np.max(np.abs(values))) if values.size else 0.0
    return {
        "samples": int(values.size),
        "duration_s": values.size / sample_rate,
        "rms_dbfs": 20 * math.log10(max(rms, 1e-12)),
        "peak_dbfs": 20 * math.log10(max(peak, 1e-12)),
        "clipping_fraction": float(np.mean(np.abs(values) >= 0.999))
        if values.size
        else 0.0,
    }


def _markdown(report: dict) -> str:
    lines = [
        "# MLX Breeze TTS 2 benchmark",
        "",
        f"- Created: `{report['created_at']}`",
        f"- Model path: `{report['resolved_model_path']}`",
        f"- Overall status: `{report['status']}`",
        "",
        "| Capability | Duration | Elapsed | RTF | Clipping | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for item in report["samples"]:
        lines.append(
            f"| {item['capability']} | {item['duration_s']:.2f}s | "
            f"{item['elapsed_s']:.2f}s | {item['rtf']:.2f} | "
            f"{item['clipping_fraction']:.6f} | {item['status']} |"
        )
    lines.extend(
        [
            "",
            "ASR, speaker similarity, leakage, and manual listening are explicitly "
            "`pending` in this runtime report. Run the evaluation workflow before "
            "making content, identity, or event-audibility claims.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_benchmark(
    model_id: str,
    output: str | Path,
    *,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    seed: int = 42,
) -> Path:
    if bool(ref_audio) != bool(ref_text):
        raise ValueError("ref_audio and ref_text must be supplied together")
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    resolved = resolve_model_path(model_id)
    model = load(resolved)
    cases = [
        {
            "capability": "voice_design_en",
            "text": "Welcome aboard. Your journey begins now.",
            "instruct": "A warm, thoughtful young woman with a clear voice.",
        },
        {
            "capability": "voice_design_zh_event",
            "text": "[笑] 欢迎来到今晚的故事时间。",
            "instruct": "一位温柔自信的年轻女性，声音清晰，语气亲切。",
        },
        {
            "capability": "streaming_en",
            "text": "(sigh) This response is streamed as it is generated.",
            "instruct": "A calm and clear voice.",
            "stream": True,
        },
    ]
    if ref_audio:
        cases.extend(
            [
                {
                    "capability": "voice_clone_en",
                    "text": "Please read this sentence in the reference voice.",
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                },
                {
                    "capability": "voice_direction_en",
                    "text": "We need to discuss what happened last night.",
                    "instruct": "Speak slowly with a restrained, serious tone.",
                    "ref_audio": ref_audio,
                    "ref_text": ref_text,
                },
            ]
        )

    samples = []
    for case in cases:
        started = time.perf_counter()
        chunks = list(
            model.generate(
                text=case["text"],
                instruct=case.get("instruct"),
                ref_audio=case.get("ref_audio"),
                ref_text=case.get("ref_text"),
                cfg_scale=4 if case.get("instruct") else 1,
                seed=seed,
                stream=case.get("stream", False),
                streaming_interval=1.0,
            )
        )
        elapsed = time.perf_counter() - started
        audio = mx.concatenate([chunk.audio for chunk in chunks])
        filename = f"{case['capability']}.wav"
        write_audio(output / filename, audio, model.sample_rate)
        metrics = _audio_metrics(audio, model.sample_rate)
        samples.append(
            {
                **case,
                **metrics,
                "audio": filename,
                "elapsed_s": elapsed,
                "rtf": elapsed / max(metrics["duration_s"], 1e-6),
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
                "chunks": len(chunks),
                "ttfa_s": chunks[0].time_to_first_audio_seconds
                if case.get("stream") and chunks
                else None,
                "status": "audio_generated",
                "asr": "pending",
                "speaker_similarity": "pending",
                "event_audibility": "pending_manual_listening"
                if "event" in case["capability"] or case.get("stream")
                else "not_applicable",
            }
        )

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "resolved_model_path": str(resolved),
        "hardware": platform.platform(),
        "seed": seed,
        "status": "audio_generated_evaluation_pending",
        "samples": samples,
        "validation": {
            "asr": "pending",
            "speaker_similarity": "pending",
            "leakage": "pending",
            "manual_listening": "pending",
        },
    }
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    markdown = _markdown(report)
    (output / "report.md").write_text(markdown)
    (output / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'><title>MLX Breeze benchmark</title>"
        "<style>body{max-width:960px;margin:40px auto;font:16px system-ui;"
        "white-space:pre-wrap}</style><body>" + html.escape(markdown) + "</body>"
    )
    return json_path
