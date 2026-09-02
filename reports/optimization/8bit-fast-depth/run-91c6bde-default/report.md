# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `91c6bde45f00e1319a355a2930e6ec65902c728d`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `pass`
- Release acceptance: `pending`
- Model-cold run: `fast / steady_state` at RTF `1.169`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| fast | steady_state | 1.169 | 1.148 | 1.162 | ≤ 2.000 | pass | 8.683 |
| fast | voice_design_cfg4 | 1.503 | 1.534 | 1.566 | ≤ 4.000 | pass | 8.334 |

## Streaming TTFA comparison

Eager and fast measurements alternate by pair to reduce thermal-order bias.

- Eager median TTFA: `2.866 s`
- Fast median TTFA: `1.304 s`
- Fast/eager regression: `-54.5%`
- TTFA regression gate: `pass`

## Synchronized stage profiles

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.0436 | 1.1% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0233 | 0.6% |
| backbone_head_and_sample | 43 | 0.0441 | 1.1% |
| depth_decode | 42 | 3.3991 | 86.4% |
| backbone_decode | 42 | 0.3543 | 9.0% |
| waveform_codec_decode | 1 | 0.0687 | 1.7% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.0782 | 1.8% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0377 | 0.9% |
| backbone_head_and_sample | 35 | 0.0378 | 0.9% |
| depth_decode | 34 | 3.7681 | 85.0% |
| backbone_decode | 34 | 0.4427 | 10.0% |
| waveform_codec_decode | 1 | 0.0672 | 1.5% |

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
