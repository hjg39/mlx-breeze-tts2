# MLX Breeze TTS 2 benchmark

- Created: `2026-08-31T16:19:21.844685+00:00`
- Model path: `models/breeze-bf16`
- Overall status: `audio_generated_evaluation_pending`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.64s | 18.56s | 7.03 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.60s | 11.29s | 7.05 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.56s | 8.86s | 3.46 | 0.000 | 0.809 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.40s | 35.11s | 3.38 | 0.121 | 0.660 | 0.037 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 5.68s | 19.09s | 3.36 | 0.000 | 0.695 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 5.20s | 17.48s | 3.36 | 0.000 | 0.599 | 0.000 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 3.52s | 23.86s | 6.78 | 0.000 | 0.714 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 3.84s | 25.80s | 6.72 | 0.000 | 0.699 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 3.20s | 10.80s | 3.38 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 3.36s | 11.35s | 3.38 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 2.56s | 12.20s | 4.77 | 0.027 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 15.36s | 53.13s | 3.46 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 2.88s | 11.21s | 3.89 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 5.04s | 19.52s | 3.87 | 0.043 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 5.04s | 20.59s | 4.09 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 1.60s | 6.15s | 3.85 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 4.08s | 16.32s | 4.00 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 2.88s | 12.50s | 4.34 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 3.12s | 12.82s | 4.11 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 3.28s | 12.19s | 3.72 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 3.60s | 12.13s | 3.37 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 2.72s | 8.98s | 3.30 | 0.600 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 4.08s | 13.50s | 3.31 | 0.100 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `2.799 s`
- Load peak memory: `7.524 GB`
- Run peak memory: `12.029 GB`
- Steady-state RTF: `3.777`
- Long-text RTF: `3.459`
- Streaming TTFA: `3.603 s`

## Validation gates

- Standard-content corpus CER: `0.0104`
- Maximum per-case CER (diagnostic): `0.1212`
- Minimum clone cosine: `0.5992`
- Minimum clone P10: `0.5940`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pass`
- Manual listening: `pending`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
