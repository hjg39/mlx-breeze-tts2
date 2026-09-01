# MLX Breeze TTS 2 benchmark

- Created: `2026-08-31T16:25:20.689857+00:00`
- Model path: `models/breeze-8bit-sensitive-bf16-v2`
- Overall status: `release_pass`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.56s | 14.45s | 5.64 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.52s | 8.71s | 5.73 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.32s | 6.65s | 2.86 | 0.000 | 0.749 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.32s | 28.60s | 2.77 | 0.333 | 0.626 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 6.16s | 16.67s | 2.71 | 0.000 | 0.585 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 4.56s | 12.46s | 2.73 | 0.000 | 0.628 | 0.000 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 3.52s | 19.55s | 5.56 | 0.000 | 0.713 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 3.84s | 21.07s | 5.49 | 0.000 | 0.698 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 3.04s | 8.43s | 2.77 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 3.44s | 8.95s | 2.60 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 2.64s | 9.77s | 3.70 | 0.027 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 16.24s | 42.15s | 2.60 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 2.88s | 7.91s | 2.75 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 5.04s | 12.94s | 2.57 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 3.76s | 9.72s | 2.59 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 2.72s | 7.25s | 2.66 | 0.130 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 4.16s | 11.39s | 2.74 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 3.76s | 10.20s | 2.71 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 3.04s | 7.95s | 2.61 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 3.68s | 9.38s | 2.55 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 3.68s | 9.16s | 2.49 | 0.111 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 2.48s | 6.27s | 2.53 | 0.600 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 4.08s | 10.15s | 2.49 | 0.100 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `2.273 s`
- Load peak memory: `6.202 GB`
- Run peak memory: `10.954 GB`
- Steady-state RTF: `2.713`
- Long-text RTF: `2.596`
- Streaming TTFA: `2.746 s`

## Validation gates

- Standard-content corpus CER: `0.0179`
- Maximum per-case CER (diagnostic): `0.3333`
- Minimum clone cosine: `0.5850`
- Minimum clone P10: `0.5728`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pass`
- Manual event listening: `pass`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
