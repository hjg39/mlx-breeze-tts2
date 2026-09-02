"""Metal-independent aggregation and rendering for speed evidence."""

import math


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def finalize_stage_profile(profile: dict, measurement: dict) -> dict:
    stages = profile.get("stages", {})
    stage_total = sum(stage["seconds"] for stage in stages.values())
    for stage in stages.values():
        stage["percent"] = (
            100.0 * stage["seconds"] / stage_total if stage_total > 0 else 0.0
        )
    profile["stage_total_s"] = stage_total
    profile["wall_s"] = measurement["elapsed_s"]
    profile["unattributed_s"] = max(0.0, measurement["elapsed_s"] - stage_total)
    profile["output_materialization_s"] = measurement["output_materialization_s"]
    profile["peak_memory_gb"] = measurement["peak_memory_gb"]
    profile["excluded_from_speed_acceptance"] = True
    return profile


def render_speed_markdown(report: dict) -> str:
    validation = report["validation"]
    model_cold = report["model_cold_run"]
    lines = [
        "# Breeze 8-bit MLX Speed Evidence",
        "",
        f"- Runtime commit: `{report['runtime_commit']}`",
        f"- Model: `{report['resolved_model_path']}`",
        f"- Speed evidence: `{'pass' if report['pass'] else 'fail'}`",
        f"- Release acceptance: `{report['release_acceptance']}`",
        f"- Model-cold run: `{model_cold['mode']} / {model_cold['case']}` at RTF `{model_cold['rtf']:.3f}`",
        "- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.",
        "",
        "## First-use and warmed results",
        "",
        "The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.",
        "",
        "| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |",
        "|---|---|---:|---:|---:|---:|---|---:|",
    ]
    for result in report["results"]:
        target = result.get("target_rtf")
        target_text = f"≤ {target:.3f}" if target is not None else "n/a"
        gate = result.get("speed_target_pass")
        gate_text = "pass" if gate is True else "fail" if gate is False else "n/a"
        peak = max(run["peak_memory_gb"] for run in result["runs"])
        lines.append(
            "| {mode} | {name} | {cold:.3f} | {median:.3f} | {p90:.3f} | "
            "{target} | {gate} | {peak:.3f} |".format(
                mode=result["mode"],
                name=result["name"],
                cold=result["prewarm_run"]["rtf"],
                median=result["median_rtf"],
                p90=result["p90_rtf"],
                target=target_text,
                gate=gate_text,
                peak=peak,
            )
        )

    lines.extend(["", "## Synchronized stage profiles", ""])
    for result in report["results"]:
        lines.extend(
            [
                f"### {result['mode']} / {result['name']}",
                "",
                "This diagnostic run is excluded from speed acceptance.",
                "",
                "| Stage | Calls | Seconds | Share |",
                "|---|---:|---:|---:|",
            ]
        )
        for name, stage in result["stage_profile"].get("stages", {}).items():
            lines.append(
                f"| {name} | {stage['calls']} | {stage['seconds']:.4f} | "
                f"{stage['percent']:.1f}% |"
            )
        lines.append("")

    lines.extend(
        [
            "## Validation",
            "",
            f"- Speed targets: `{validation['speed_targets']}`",
            f"- Fixed-seed reproducibility: `{validation['fixed_seed_reproducibility']}`",
            f"- Fast/eager exact waveform match: `{validation['fast_eager_exact_match']}`",
            "",
            "## Pending release gates",
            "",
        ]
    )
    lines.extend(f"- {gate}" for gate in report["pending_release_gates"])
    return "\n".join(lines) + "\n"
