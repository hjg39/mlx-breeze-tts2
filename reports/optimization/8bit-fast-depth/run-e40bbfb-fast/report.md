# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `e40bbfbf55675da56945e82178c6a81c5bb62cce`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `pass`
- Release acceptance: `pending`
- Model-cold run: `fast / steady_state` at RTF `1.042`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| fast | steady_state | 1.042 | 1.026 | 1.027 | ≤ 2.000 | pass | 8.683 |
| fast | voice_design_cfg4 | 1.330 | 1.321 | 1.323 | ≤ 4.000 | pass | 8.334 |

## Streaming TTFA comparison

Eager and fast measurements alternate by pair to reduce thermal-order bias.

- Eager median TTFA: `2.314 s`
- Fast median TTFA: `1.106 s`
- Fast/eager regression: `-52.2%`
- TTFA regression gate: `pass`

## Synchronized stage profiles

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.0408 | 1.1% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0212 | 0.6% |
| backbone_head_and_sample | 43 | 0.0325 | 0.9% |
| depth_decode | 42 | 3.0692 | 86.2% |
| backbone_decode | 42 | 0.3334 | 9.4% |
| waveform_codec_decode | 1 | 0.0649 | 1.8% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.0708 | 1.9% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0352 | 0.9% |
| backbone_head_and_sample | 35 | 0.0312 | 0.8% |
| depth_decode | 34 | 3.1536 | 84.2% |
| backbone_decode | 34 | 0.4001 | 10.7% |
| waveform_codec_decode | 1 | 0.0531 | 1.4% |

## Validation

- Speed targets: `True`
- Streaming TTFA regression: `True`
- Fixed-seed reproducibility: `True`
- Fast/eager exact waveform match: `None`
- Additional quality review required: `True`

## Pending release gates

- 23-case fast-path capability matrix
- objective ASR, speaker similarity, and leakage metrics
- precision-specific PyTorch parity
- waveform, streaming, HTTP, event, and cancellation regression gates
- manual listening review
