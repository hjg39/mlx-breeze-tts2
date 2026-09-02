# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `851abbc6764de615fc6ad32fc5e2665c08a10bc0`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `pass`
- Release acceptance: `pending`
- Model-cold run: `fast / steady_state` at RTF `1.066`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| fast | steady_state | 1.066 | 1.046 | 1.049 | ≤ 2.000 | pass | 8.683 |
| fast | voice_design_cfg4 | 1.442 | 1.398 | 1.444 | ≤ 4.000 | pass | 8.334 |

## Streaming TTFA comparison

Eager and fast measurements alternate by pair to reduce thermal-order bias.

- Eager median TTFA: `2.331 s`
- Fast median TTFA: `1.112 s`
- Fast/eager regression: `-52.3%`
- TTFA regression gate: `pass`

## Synchronized stage profiles

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.0428 | 1.2% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0237 | 0.7% |
| backbone_head_and_sample | 43 | 0.0264 | 0.7% |
| depth_decode | 42 | 3.0691 | 86.8% |
| backbone_decode | 42 | 0.3043 | 8.6% |
| waveform_codec_decode | 1 | 0.0678 | 1.9% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.0728 | 1.9% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0365 | 1.0% |
| backbone_head_and_sample | 35 | 0.0269 | 0.7% |
| depth_decode | 34 | 3.2885 | 85.7% |
| backbone_decode | 34 | 0.3607 | 9.4% |
| waveform_codec_decode | 1 | 0.0528 | 1.4% |

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
