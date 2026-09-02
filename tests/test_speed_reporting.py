from mlx_breeze_tts.speed_reporting import (
    finalize_stage_profile,
    percentile,
    render_speed_markdown,
)


def test_percentile_uses_nearest_rank():
    assert percentile([3.0, 1.0, 2.0, 5.0, 4.0], 0.9) == 5.0


def test_stage_profile_aggregation_is_explicitly_diagnostic():
    profile = {
        "stages": {
            "depth_decode": {"seconds": 3.0, "calls": 2},
            "backbone_decode": {"seconds": 1.0, "calls": 2},
        }
    }
    measurement = {
        "elapsed_s": 5.0,
        "output_materialization_s": 0.1,
        "peak_memory_gb": 4.0,
    }

    result = finalize_stage_profile(profile, measurement)

    assert result["stage_total_s"] == 4.0
    assert result["unattributed_s"] == 1.0
    assert result["stages"]["depth_decode"]["percent"] == 75.0
    assert result["excluded_from_speed_acceptance"] is True


def test_markdown_keeps_release_acceptance_pending():
    report = {
        "runtime_commit": "abc1234",
        "resolved_model_path": "/models/8bit",
        "pass": True,
        "release_acceptance": "pending",
        "results": [
            {
                "mode": "fast",
                "name": "steady_state",
                "cold_run": {"rtf": 2.5},
                "median_rtf": 1.8,
                "p90_rtf": 1.9,
                "target_rtf": 2.0,
                "speed_target_pass": True,
                "runs": [{"peak_memory_gb": 4.0}],
                "stage_profile": {
                    "stages": {
                        "depth_decode": {
                            "calls": 10,
                            "seconds": 1.0,
                            "percent": 50.0,
                        }
                    }
                },
            }
        ],
        "validation": {
            "speed_targets": True,
            "fixed_seed_reproducibility": True,
            "fast_eager_exact_match": True,
        },
        "pending_release_gates": ["manual listening review"],
    }

    markdown = render_speed_markdown(report)

    assert "Speed evidence: `pass`" in markdown
    assert "Release acceptance: `pending`" in markdown
    assert "| fast | steady_state | 2.500 | 1.800 | 1.900 |" in markdown
    assert "manual listening review" in markdown
