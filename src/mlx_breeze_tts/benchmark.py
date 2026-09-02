"""Reproducible local capability benchmark with machine-readable evidence."""

import json
import math
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .matrix import build_acceptance_matrix, matrix_coverage, reference_pair
from .objective import objective_template
from .provenance import checkpoint_provenance
from .reporting import render_report_bundle


def _resolve_model_path(model_id: str) -> Path:
    from .loader import resolve_model_path

    return resolve_model_path(model_id)


def _load_model(path: Path):
    from .loader import load

    return load(path)


def _mlx_core():
    import mlx.core as mx

    return mx


def _write_audio(path: Path, audio, sample_rate: int) -> None:
    from .audio import write_audio

    write_audio(path, audio, sample_rate)


def _probe_http(model, seed: int) -> dict:
    try:
        import asyncio

        from httpx import ASGITransport, AsyncClient

        from .server import create_app

        async def request_pair():
            transport = ASGITransport(
                app=create_app(model=model, inline_inference=True)
            )
            async with AsyncClient(
                transport=transport, base_url="http://mlx-breeze.local"
            ) as client:
                health = await client.get("/health")
                speech = await client.post(
                    "/v1/audio/speech",
                    data={
                        "text": "HTTP interface validation completed successfully.",
                        "seed": str(seed),
                    },
                )
            return health, speech

        health, speech = asyncio.run(request_pair())
        checks = {
            "health_200": health.status_code == 200,
            "health_sample_rate": health.json().get("sample_rate") == 24_000,
            "speech_200": speech.status_code == 200,
            "content_type_pcm": speech.headers.get("content-type") == "audio/pcm",
            "sample_rate_header": speech.headers.get("x-sample-rate") == "24000",
            "sample_format_header": speech.headers.get("x-sample-format") == "s16le",
            "cache_control": speech.headers.get("cache-control") == "no-store",
            "nonempty_even_pcm": bool(speech.content) and len(speech.content) % 2 == 0,
        }
        report = {
            "status": "pass" if all(checks.values()) else "fail",
            "checks": checks,
            "speech_status_code": speech.status_code,
        }
        if speech.status_code != 200:
            report["speech_response"] = speech.text[:1000]
        return report
    except ImportError as exc:
        return {"status": "pending", "detail": f"server extra unavailable: {exc}"}
    except Exception as exc:
        return {"status": "fail", "detail": f"{type(exc).__name__}: {exc}"}


def _audio_metrics(audio, sample_rate: int) -> dict:
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


def _waveform_diagnostics(
    audio, sample_rate: int, chunk_lengths: list[int] | None = None
) -> dict:
    """Detect non-finite output, exact-like repeated tails, and stream jumps."""
    values = np.asarray(audio, dtype=np.float32).reshape(-1)
    finite = values[np.isfinite(values)]
    nonfinite_count = int(values.size - finite.size)
    clipping_fraction = float(np.mean(np.abs(finite) >= 0.999)) if finite.size else 0.0

    window = min(sample_rate // 2, values.size // 2)
    tail_similarity = None
    repeated_tail = False
    if window >= max(32, sample_rate // 20):
        previous = values[-2 * window : -window]
        tail = values[-window:]
        denominator = float(np.linalg.norm(previous) * np.linalg.norm(tail))
        if denominator > 1e-8:
            tail_similarity = float(np.dot(previous, tail) / denominator)
            repeated_tail = tail_similarity >= 0.995

    boundary_jumps = []
    offset = 0
    for length in (chunk_lengths or [])[:-1]:
        offset += length
        if 0 < offset < values.size:
            boundary_jumps.append(float(abs(values[offset] - values[offset - 1])))
    max_boundary_jump = max(boundary_jumps, default=0.0)
    return {
        "nonfinite_count": nonfinite_count,
        "severe_clipping": clipping_fraction > 0.001,
        "tail_repeat_similarity": tail_similarity,
        "repeated_tail": repeated_tail,
        "stream_boundary_jump_max": max_boundary_jump,
        "stream_discontinuity": max_boundary_jump > 0.5,
    }


def run_benchmark(
    model_id: str,
    output: str | Path,
    *,
    ref_audio: str | None = None,
    ref_text: str | None = None,
    ref_audio_en: str | None = None,
    ref_text_en: str | None = None,
    ref_audio_zh: str | None = None,
    ref_text_zh: str | None = None,
    seed: int = 42,
    fast_depth: bool = False,
    invoked_via_cli: bool = False,
    probe_http: bool = True,
) -> Path:
    if not isinstance(fast_depth, bool):
        raise ValueError("fast_depth must be a boolean")
    if ref_audio_en and ref_audio:
        raise ValueError("Use either legacy ref_audio or ref_audio_en, not both")
    if ref_text_en and ref_text:
        raise ValueError("Use either legacy ref_text or ref_text_en, not both")
    ref_audio_en = ref_audio_en or ref_audio
    ref_text_en = ref_text_en or ref_text
    reference_en = reference_pair(ref_audio_en, ref_text_en, "en")
    reference_zh = reference_pair(ref_audio_zh, ref_text_zh, "zh")
    mx = _mlx_core()
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()
    output = Path(output).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    resolved = _resolve_model_path(model_id)
    provenance = checkpoint_provenance(resolved)
    load_started = time.perf_counter()
    model = _load_model(resolved)
    load_time = time.perf_counter() - load_started
    load_peak_memory_gb = mx.get_peak_memory() / 1e9
    cases = build_acceptance_matrix(reference_en, reference_zh)

    samples = []
    for case in cases:
        if callable(reset_peak_memory):
            reset_peak_memory()
        if case.get("skip_reason"):
            samples.append(
                {
                    **case,
                    "audio": "",
                    "samples": 0,
                    "duration_s": 0.0,
                    "elapsed_s": 0.0,
                    "rtf": 0.0,
                    "peak_memory_gb": mx.get_peak_memory() / 1e9,
                    "chunks": 0,
                    "ttfa_s": None,
                    "clipping_fraction": 0.0,
                    "peak_dbfs": -240.0,
                    "rms_dbfs": -240.0,
                    "status": "missing_input",
                    "asr": "pending",
                    "speaker_similarity": "pending",
                }
            )
            continue

        iterations = int(case.get("iterations", 1))
        equality_repetitions = int(case.get("repetitions_for_equality", 1))
        repetitions = max(iterations, equality_repetitions)
        elapsed_runs = []
        rtf_runs = []
        arrays = []
        chunk_lengths = []
        chunks = []
        first_chunk = None
        for _ in range(repetitions):
            started = time.perf_counter()
            kwargs = {
                "text": case["text"],
                "instruct": case.get("instruct"),
                "ref_audio": case.get("ref_audio"),
                "ref_text": case.get("ref_text"),
                "cfg_scale": 4 if case.get("instruct") else 1,
                "seed": case.get("seed", seed),
                "stream": case.get("stream", False),
                "streaming_interval": 1.0,
                "temperature": case.get("temperature", 0.9),
                "top_p": case.get("top_p", 1.0),
                "top_k": case.get("top_k", 50),
                "repetition_penalty": case.get("repetition_penalty", 1.1),
                "fast_depth": fast_depth,
            }
            if case.get("cancel_after_first_chunk"):
                stream = model.generate(**kwargs)
                first_chunk = next(stream)
                stream.close()
                kwargs.update(
                    {
                        "text": case["post_cancel_text"],
                        "stream": False,
                        "instruct": None,
                        "cfg_scale": 1,
                    }
                )
                chunks = list(model.generate(**kwargs))
            else:
                chunks = list(model.generate(**kwargs))
                if chunks and first_chunk is None:
                    first_chunk = chunks[0]
            elapsed = time.perf_counter() - started
            audio = (
                mx.concatenate([chunk.audio for chunk in chunks])
                if chunks
                else mx.zeros((0,), dtype=mx.float32)
            )
            metrics = _audio_metrics(audio, model.sample_rate)
            elapsed_runs.append(elapsed)
            rtf_runs.append(elapsed / max(metrics["duration_s"], 1e-6))
            arrays.append(np.asarray(audio, dtype=np.float32))
            chunk_lengths = [int(np.asarray(chunk.audio).size) for chunk in chunks]

        elapsed = elapsed_runs[-1]
        audio_values = arrays[-1]
        audio = mx.array(audio_values)
        filename = f"{case['capability']}.wav"
        _write_audio(output / filename, audio, model.sample_rate)
        metrics = _audio_metrics(audio, model.sample_rate)
        diagnostics = _waveform_diagnostics(
            audio, model.sample_rate, chunk_lengths if case.get("stream") else None
        )
        reproducible = (
            all(np.array_equal(arrays[0], item) for item in arrays[1:])
            if equality_repetitions > 1
            else None
        )
        samples.append(
            {
                **case,
                **metrics,
                **diagnostics,
                "expected_text": case.get("post_cancel_text", case["text"]),
                "audio": filename,
                "elapsed_s": elapsed,
                "rtf": rtf_runs[-1],
                "peak_memory_gb": mx.get_peak_memory() / 1e9,
                "chunks": len(chunks)
                + (1 if case.get("cancel_after_first_chunk") else 0),
                "ttfa_s": first_chunk.time_to_first_audio_seconds
                if case.get("stream") and first_chunk
                else None,
                "elapsed_runs_s": elapsed_runs,
                "rtf_runs": rtf_runs,
                "steady_state_rtf": (
                    sum(rtf_runs[1:]) / len(rtf_runs[1:]) if iterations > 1 else None
                ),
                "seed_exact_match": reproducible,
                "status": "audio_generated" if metrics["samples"] else "empty_audio",
                "asr": "pending",
                "speaker_similarity": "pending",
            }
        )

    by_capability = {sample["capability"]: sample for sample in samples}
    streaming_pass = all(
        by_capability.get(name, {}).get("status") == "audio_generated"
        for name in ("streaming_en", "stream_cancel")
    )
    sampling_pass = (
        by_capability.get("sampling_controls", {}).get("status") == "audio_generated"
    )
    seed_pass = by_capability.get("seed_reproducibility", {}).get("seed_exact_match")
    waveform_pass = all(
        sample.get("status") == "missing_input"
        or (
            sample.get("nonfinite_count") == 0
            and not sample.get("severe_clipping")
            and not sample.get("repeated_tail")
            and not sample.get("stream_discontinuity")
        )
        for sample in samples
    )
    http_probe = _probe_http(model, seed) if probe_http else {"status": "pending"}
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": provenance.get("artifact_revision")
        or provenance.get("upstream_revision"),
        "model_provenance": provenance,
        "resolved_model_path": str(resolved),
        "hardware": platform.platform(),
        "seed": seed,
        "runtime_options": {"fast_depth": fast_depth},
        "status": "audio_generated_evaluation_pending",
        "matrix_coverage": matrix_coverage(cases),
        "artifact_audit": {
            "pass": True,
            "missing": [],
            "unexpected": [],
            "shape_mismatches": {},
            "method": "benchmark loaded the checkpoint through strict loader",
        },
        "interfaces": {
            "python": "pass",
            "cli": "pass" if invoked_via_cli else "pending",
            "http": http_probe["status"],
            "streaming": "pass" if streaming_pass else "fail",
        },
        "interface_details": {"http": http_probe},
        "performance": {
            "load_time_s": load_time,
            "load_peak_memory_gb": load_peak_memory_gb,
            "peak_memory_gb": max(
                (sample["peak_memory_gb"] for sample in samples), default=0.0
            ),
            "steady_state_rtf": by_capability.get("steady_state", {}).get(
                "steady_state_rtf"
            ),
            "long_text_rtf": by_capability.get("long_text", {}).get("rtf"),
            "streaming_ttfa_s": by_capability.get("streaming_en", {}).get("ttfa_s"),
        },
        "samples": samples,
        "validation": {
            "max_cer": None,
            "clone_cosine_min": None,
            "clone_p10_min": None,
            "leakage_max": None,
            "seed_reproducibility": "pass" if seed_pass else "fail",
            "sampling_path": "pass" if sampling_pass else "fail",
            "waveform_integrity": "pass" if waveform_pass else "fail",
            "pytorch_parity": "pending",
            "manual_listening": "pending",
        },
    }
    json_path = output / "summary.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    objective_template(json_path, output / "objective_metrics.json")
    render_report_bundle(report, output)
    return json_path
