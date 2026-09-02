# MLX Breeze TTS 2 benchmark

- Created: `2026-09-02T13:26:07.875833+00:00`
- Model path: `/Users/vanch/mlx-breeze-tts2/models/breeze-8bit-sensitive-bf16-v2`
- Overall status: `audio_generated_evaluation_pending`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.72s | 20.76s | 7.63 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.60s | 12.04s | 7.52 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.72s | 12.80s | 4.70 | 0.000 | 0.740 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.32s | 49.76s | 4.82 | 0.030 | 0.672 | 0.037 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 5.28s | 30.32s | 5.74 | 0.000 | 0.729 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 5.76s | 19.51s | 3.39 | 2.300 | 0.642 | 0.044 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 3.04s | 29.12s | 9.58 | 0.000 | 0.700 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 3.84s | 41.49s | 10.80 | 0.000 | 0.741 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 6.48s | 45.54s | 7.03 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 4.64s | 35.27s | 7.60 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 2.72s | 25.61s | 9.42 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 15.76s | 70.45s | 4.47 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 3.36s | 5.51s | 1.64 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 4.48s | 7.74s | 1.73 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 4.00s | 6.14s | 1.53 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 2.48s | 3.84s | 1.55 | 0.043 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 1.92s | 2.98s | 1.55 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 2.32s | 3.47s | 1.50 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 2.64s | 4.18s | 1.58 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 2.72s | 4.11s | 1.51 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 4.16s | 6.09s | 1.46 | 0.222 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 1.92s | 2.86s | 1.49 | 0.200 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 3.20s | 5.33s | 1.67 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `2.153 s`
- Load peak memory: `6.202 GB`
- Run peak memory: `10.820 GB`
- Steady-state RTF: `2.825`
- Long-text RTF: `4.470`
- Streaming TTFA: `8.019 s`

## Validation gates

- Standard-content corpus CER: `0.0015`
- Maximum per-case CER (diagnostic): `0.0303`
- Minimum clone cosine: `0.6415`
- Minimum clone P10: `0.6239`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pending`
- Manual event listening: `pending`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
