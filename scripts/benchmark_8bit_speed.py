#!/usr/bin/env python3
"""Compare cold/warmed 8-bit generation and capture a separate stage profile."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import mlx.core as mx
import numpy as np

from mlx_breeze_tts import load, write_audio
from mlx_breeze_tts.loader import resolve_model_path
from mlx_breeze_tts.provenance import checkpoint_provenance
from mlx_breeze_tts.speed_reporting import (
    finalize_stage_profile,
    percentile,
    render_speed_markdown,
)

ROOT = Path(__file__).resolve().parents[1]


CASES = (
    {
        "name": "steady_state",
        "text": "Steady state performance is measured after warmup.",
        "instruct": None,
        "cfg_scale": 1.0,
        "target_rtf": 2.0,
    },
    {
        "name": "voice_design_cfg4",
        "text": "Welcome aboard. Your journey begins now.",
        "instruct": "A warm, thoughtful young woman with a clear voice.",
        "cfg_scale": 4.0,
        "target_rtf": 4.0,
    },
)


def _generate(
    model,
    case: dict,
    seed: int,
    *,
    fast_depth: bool,
    stage_profile: dict | None = None,
) -> tuple[mx.array, dict]:
    reset_peak_memory = getattr(mx, "reset_peak_memory", None)
    if callable(reset_peak_memory):
        reset_peak_memory()
    started = time.perf_counter()
    chunks = list(
        model.generate(
            text=case["text"],
            instruct=case["instruct"],
            cfg_scale=case["cfg_scale"],
            max_tokens=1500,
            seed=seed,
            fast_depth=fast_depth,
            _stage_profile=stage_profile,
        )
    )
    audio = mx.concatenate([chunk.audio for chunk in chunks])
    mx.eval(audio)
    elapsed = time.perf_counter() - started
    duration = int(audio.shape[0]) / model.sample_rate
    if duration <= 0:
        raise RuntimeError(f"{case['name']} generated empty audio")
    materialize_started = time.perf_counter()
    values = np.asarray(audio, dtype=np.float32)
    materialize_seconds = time.perf_counter() - materialize_started
    measurement = {
        "elapsed_s": elapsed,
        "duration_s": duration,
        "rtf": elapsed / duration,
        "peak_memory_gb": mx.get_peak_memory() / 1e9,
        "output_materialization_s": materialize_seconds,
        "sha256": hashlib.sha256(values.tobytes()).hexdigest(),
    }
    return audio, measurement


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode", choices=("compare", "eager", "fast"), default="compare"
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    args.output.mkdir(parents=True, exist_ok=True)
    resolved_model = resolve_model_path(args.model)
    provenance = checkpoint_provenance(resolved_model)
    load_started = time.perf_counter()
    model = load(resolved_model)
    load_seconds = time.perf_counter() - load_started
    results = []

    modes = ("eager", "fast") if args.mode == "compare" else (args.mode,)
    cold_mode = modes[0]
    cold_case = CASES[0]
    _, model_cold_run = _generate(
        model,
        cold_case,
        args.seed,
        fast_depth=cold_mode == "fast",
    )
    model_cold_run.update({"mode": cold_mode, "case": cold_case["name"]})
    for mode in modes:
        fast_depth = mode == "fast"
        for case in CASES:
            if mode == cold_mode and case["name"] == cold_case["name"]:
                prewarm_run = dict(model_cold_run)
            else:
                _, prewarm_run = _generate(
                    model, case, args.seed, fast_depth=fast_depth
                )
            runs = []
            final_audio = None
            for _ in range(args.runs):
                audio, measurement = _generate(
                    model, case, args.seed, fast_depth=fast_depth
                )
                runs.append(measurement)
                final_audio = audio

            stage_profile = {}
            _, profiled_run = _generate(
                model,
                case,
                args.seed,
                fast_depth=fast_depth,
                stage_profile=stage_profile,
            )
            stage_profile = finalize_stage_profile(stage_profile, profiled_run)
            output_wav = args.output / f"{mode}_{case['name']}.wav"
            wav_write_started = time.perf_counter()
            write_audio(output_wav, final_audio, model.sample_rate)
            wav_write_seconds = time.perf_counter() - wav_write_started
            rtfs = [run["rtf"] for run in runs]
            median_rtf = statistics.median(rtfs)
            results.append(
                {
                    **case,
                    "mode": mode,
                    "prewarm_runs": 1,
                    "prewarm_run": prewarm_run,
                    "measured_runs": args.runs,
                    "median_rtf": median_rtf,
                    "p90_rtf": percentile(rtfs, 0.9),
                    "best_rtf": min(rtfs),
                    "exact_reproducible": len({run["sha256"] for run in runs}) == 1,
                    "speed_target_pass": (
                        median_rtf <= case["target_rtf"] if fast_depth else None
                    ),
                    "audio": str(output_wav),
                    "wav_write_s": wav_write_seconds,
                    "stage_profile": stage_profile,
                    "runs": runs,
                }
            )

    by_mode_case = {(result["mode"], result["name"]): result for result in results}
    exact_matches = {}
    if args.mode == "compare":
        for case in CASES:
            eager = by_mode_case[("eager", case["name"])]
            fast = by_mode_case[("fast", case["name"])]
            exact_matches[case["name"]] = (
                eager["runs"][0]["sha256"] == fast["runs"][0]["sha256"]
            )

    fast_results = [result for result in results if result["mode"] == "fast"]
    speed_pass = (
        all(result["speed_target_pass"] for result in fast_results)
        if fast_results
        else None
    )
    reproducibility_pass = all(result["exact_reproducible"] for result in results)
    exact_match_pass = all(exact_matches.values()) if exact_matches else None

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "resolved_model_path": str(resolved_model),
        "model_provenance": provenance,
        "runtime_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "mlx": importlib.metadata.version("mlx"),
        },
        "measurement_scope": {
            "model_load_in_rtf": False,
            "reference_preprocessing_in_rtf": True,
            "wav_write_in_rtf": False,
            "output_materialization_in_rtf": False,
            "stage_profile_in_rtf": False,
            "load_seconds": load_seconds,
        },
        "model_cold_run": model_cold_run,
        "results": results,
        "fast_matches_eager": exact_matches,
        "validation": {
            "speed_targets": speed_pass,
            "fixed_seed_reproducibility": reproducibility_pass,
            "fast_eager_exact_match": exact_match_pass,
            "quality_review_required": exact_match_pass is False,
        },
        "release_acceptance": "pending",
        "pending_release_gates": [
            "23-case fast-path capability matrix",
            "objective ASR, speaker similarity, and leakage metrics",
            "precision-specific PyTorch parity",
            "waveform, streaming, HTTP, event, and cancellation regression gates",
            "manual listening review",
        ],
        "pass": (speed_pass is not False and reproducibility_pass),
    }
    output_json = args.output / "speed.json"
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    output_markdown = args.output / "report.md"
    output_markdown.write_text(render_speed_markdown(report))
    print(output_json)
    print(output_markdown)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
