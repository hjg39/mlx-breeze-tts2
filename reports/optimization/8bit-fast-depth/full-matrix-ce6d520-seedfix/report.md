# MLX Breeze TTS 2 benchmark

- Created: `2026-09-02T14:04:04.090173+00:00`
- Model path: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Overall status: `audio_generated_evaluation_pending`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.72s | 7.49s | 2.75 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.60s | 4.38s | 2.74 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.72s | 3.93s | 1.44 | 0.000 | 0.740 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.32s | 14.31s | 1.39 | 0.030 | 0.672 | 0.037 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 5.28s | 7.19s | 1.36 | 0.000 | 0.729 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 5.76s | 7.76s | 1.35 | 0.000 | 0.642 | 0.000 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 3.04s | 6.00s | 1.97 | 0.000 | 0.700 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 3.84s | 7.52s | 1.96 | 0.000 | 0.741 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 6.48s | 8.40s | 1.30 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 4.64s | 5.92s | 1.28 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 2.72s | 4.90s | 1.80 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 15.76s | 19.82s | 1.26 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 3.36s | 4.22s | 1.26 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 4.48s | 5.64s | 1.26 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 4.00s | 5.05s | 1.26 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 2.48s | 3.11s | 1.25 | 0.043 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 1.92s | 2.47s | 1.29 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 2.32s | 2.98s | 1.28 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 2.64s | 3.39s | 1.28 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 2.72s | 3.39s | 1.25 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 4.16s | 5.38s | 1.29 | 0.222 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 2.24s | 2.94s | 1.31 | 0.200 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 3.20s | 4.34s | 1.35 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `0.989 s`
- Load peak memory: `6.202 GB`
- Run peak memory: `10.820 GB`
- Steady-state RTF: `1.258`
- Long-text RTF: `1.258`
- Streaming TTFA: `1.330 s`

## Validation gates

- Standard-content corpus CER: `0.0013`
- Maximum per-case CER (diagnostic): `0.0303`
- Minimum clone cosine: `0.6415`
- Minimum clone P10: `0.6239`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pass`
- Manual event listening: `pending`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
