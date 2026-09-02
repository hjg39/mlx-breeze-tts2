# MLX Breeze TTS 2 benchmark

- Created: `2026-09-02T15:23:00.010949+00:00`
- Model path: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Overall status: `audio_generated_evaluation_pending`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.72s | 4.14s | 1.52 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.60s | 2.43s | 1.52 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.72s | 3.23s | 1.19 | 0.000 | 0.740 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.32s | 11.86s | 1.15 | 0.030 | 0.672 | 0.037 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 5.28s | 6.09s | 1.15 | 0.000 | 0.729 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 5.76s | 6.67s | 1.16 | 0.000 | 0.642 | 0.000 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 3.04s | 4.94s | 1.62 | 0.000 | 0.700 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 3.84s | 6.41s | 1.67 | 0.000 | 0.741 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 6.48s | 7.68s | 1.19 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 4.64s | 5.55s | 1.20 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 2.72s | 4.54s | 1.67 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 15.76s | 18.81s | 1.19 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 3.36s | 4.61s | 1.37 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 4.48s | 6.10s | 1.36 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 4.00s | 5.23s | 1.31 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 2.48s | 3.36s | 1.35 | 0.043 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 4.08s | 5.98s | 1.47 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 2.32s | 3.43s | 1.48 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 2.56s | 3.75s | 1.46 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 2.72s | 4.13s | 1.52 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 4.16s | 6.36s | 1.53 | 0.222 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 2.24s | 3.53s | 1.57 | 0.200 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 3.20s | 4.67s | 1.46 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `1.039 s`
- Load peak memory: `6.202 GB`
- Run peak memory: `10.820 GB`
- Steady-state RTF: `1.387`
- Long-text RTF: `1.194`
- Streaming TTFA: `1.293 s`

## Validation gates

- Standard-content corpus CER: `0.0013`
- Maximum per-case CER (diagnostic): `0.0303`
- Minimum clone cosine: `0.6415`
- Minimum clone P10: `0.6239`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pass`
- Manual event listening: `pass`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
