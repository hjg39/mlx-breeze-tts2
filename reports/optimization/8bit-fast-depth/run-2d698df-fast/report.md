# Breeze 8-bit MLX Speed Evidence

- Runtime commit: `2d698dffc5803c0d1482a9e77657ba0c0388b18e`
- Model: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Speed evidence: `pass`
- Release acceptance: `pending`
- Model-cold run: `fast / steady_state` at RTF `1.338`
- Model loading, output materialization, stage profiling, and WAV writing are excluded from RTF.

## First-use and warmed results

The report contains exactly one true model-cold run. Other first-use rows warm their specific path and case after the model is already loaded.

| Mode | Case | First-use RTF | Median warm RTF | P90 | Target | Result | Peak GB |
|---|---|---:|---:|---:|---:|---|---:|
| fast | steady_state | 1.338 | 1.712 | 2.947 | ≤ 2.000 | pass | 8.683 |
| fast | voice_design_cfg4 | 2.209 | 2.902 | 3.666 | ≤ 4.000 | pass | 8.334 |

## Synchronized stage profiles

### fast / steady_state

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 1 | 0.2815 | 5.8% |
| prompt_assembly | 1 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.0251 | 0.5% |
| backbone_head_and_sample | 43 | 0.0509 | 1.1% |
| depth_decode | 42 | 3.7597 | 77.9% |
| backbone_decode | 42 | 0.5851 | 12.1% |
| waveform_codec_decode | 1 | 0.1230 | 2.5% |

### fast / voice_design_cfg4

This diagnostic run is excluded from speed acceptance.

| Stage | Calls | Seconds | Share |
|---|---:|---:|---:|
| text_encoder_and_projection | 2 | 0.3049 | 4.0% |
| prompt_assembly | 2 | 0.0000 | 0.0% |
| backbone_prefill | 1 | 0.2023 | 2.6% |
| backbone_head_and_sample | 35 | 0.0505 | 0.7% |
| depth_decode | 34 | 6.3980 | 83.8% |
| backbone_decode | 34 | 0.5894 | 7.7% |
| waveform_codec_decode | 1 | 0.0928 | 1.2% |

## Validation

- Speed targets: `True`
- Fixed-seed reproducibility: `True`
- Fast/eager exact waveform match: `None`
- Additional quality review required: `False`

## Pending release gates

- 23-case fast-path capability matrix
- objective ASR, speaker similarity, and leakage metrics
- precision-specific PyTorch parity
- waveform, streaming, HTTP, event, and cancellation regression gates
- manual listening review
