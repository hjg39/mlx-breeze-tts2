# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `16510b62734c74fe7162e5778397fe10e6292861`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `fail`
- Release acceptance: `pending`
- Model-cold run: `eager / steady_state` at RTF `2.663`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| eager | steady_state | 2.663 | 2.741 | 3.023 | ≤ 2.000 | n/a | 8.434 |
| eager | voice_design_cfg4 | 6.701 | 8.277 | 30.725 | ≤ 4.000 | n/a | 8.438 |
| fast | steady_state | 7.392 | 4.478 | 6.466 | ≤ 2.000 | fail | 8.683 |
| fast | voice_design_cfg4 | 6.546 | 5.452 | 6.040 | ≤ 4.000 | fail | 8.334 |

## Synchronized stage profiles

### eager / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.0606 | 0.7% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0322 | 0.3% |
| backbone_head_and_sample | 38 | 0.0481 | 0.5% |
| depth_decode | 37 | 8.6976 | 93.6% |
| backbone_decode | 37 | 0.3585 | 3.9% |
| waveform_codec_decode | 1 | 0.0919 | 1.0% |

### eager / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 1.6723 | 1.2% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 1.3128 | 0.9% |
| backbone_head_and_sample | 37 | 0.3339 | 0.2% |
| depth_decode | 36 | 132.1427 | 93.1% |
| backbone_decode | 36 | 6.0242 | 4.2% |
| waveform_codec_decode | 1 | 0.4044 | 0.3% |

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.1158 | 0.9% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0370 | 0.3% |
| backbone_head_and_sample | 43 | 0.1840 | 1.4% |
| depth_decode | 42 | 11.5533 | 85.2% |
| backbone_decode | 42 | 1.4734 | 10.9% |
| waveform_codec_decode | 1 | 0.2003 | 1.5% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.2076 | 1.5% |
| prompt_assembly | 2 | 0.0001 | 0.0% |
| backbone_prefill | 1 | 0.0587 | 0.4% |
| backbone_head_and_sample | 35 | 0.1523 | 1.1% |
| depth_decode | 34 | 10.9814 | 78.6% |
| backbone_decode | 34 | 2.4839 | 17.8% |
| waveform_codec_decode | 1 | 0.0938 | 0.7% |

## Validation

- Speed targets: `False`
- Fixed-seed reproducibility: `True`
- Fast/eager exact waveform match: `False`
- Additional quality review required: `True`

## Pending release gates

- 23-case fast-path capability matrix
- objective ASR, speaker similarity, and leakage metrics
- precision-specific PyTorch parity
- waveform, streaming, HTTP, event, and cancellation regression gates
- manual listening review
