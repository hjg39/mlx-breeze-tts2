# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `3194b93654203309d9004a4497a26377220dfe21`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `fail`
- Release acceptance: `pending`
- Model-cold run: `eager / steady_state` at RTF `5.641`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| eager | steady_state | 5.641 | 8.460 | 13.127 | ≤ 2.000 | n/a | 8.434 |
| eager | voice_design_cfg4 | 5.383 | 5.636 | 6.431 | ≤ 4.000 | n/a | 8.439 |
| fast | steady_state | 5.246 | 3.922 | 6.632 | ≤ 2.000 | fail | 8.683 |
| fast | voice_design_cfg4 | 3.029 | 2.823 | 3.076 | ≤ 4.000 | pass | 8.334 |

## Synchronized stage profiles

### eager / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.5599 | 3.9% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.1689 | 1.2% |
| backbone_head_and_sample | 38 | 0.0459 | 0.3% |
| depth_decode | 37 | 12.8019 | 89.8% |
| backbone_decode | 37 | 0.5130 | 3.6% |
| waveform_codec_decode | 1 | 0.1623 | 1.1% |

### eager / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.2139 | 0.9% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.1749 | 0.7% |
| backbone_head_and_sample | 37 | 0.0582 | 0.2% |
| depth_decode | 36 | 22.2218 | 91.4% |
| backbone_decode | 36 | 1.4151 | 5.8% |
| waveform_codec_decode | 1 | 0.2413 | 1.0% |

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.0931 | 1.2% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0702 | 0.9% |
| backbone_head_and_sample | 43 | 0.0864 | 1.2% |
| depth_decode | 42 | 6.0367 | 80.5% |
| backbone_decode | 42 | 1.0313 | 13.8% |
| waveform_codec_decode | 1 | 0.1817 | 2.4% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.1004 | 1.3% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0753 | 1.0% |
| backbone_head_and_sample | 35 | 0.0326 | 0.4% |
| depth_decode | 34 | 6.6423 | 88.4% |
| backbone_decode | 34 | 0.5462 | 7.3% |
| waveform_codec_decode | 1 | 0.1155 | 1.5% |

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
