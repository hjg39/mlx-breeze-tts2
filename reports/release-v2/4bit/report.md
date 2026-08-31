# MLX Breeze TTS 2 benchmark

- Created: `2026-08-31T16:31:33.524532+00:00`
- Model path: `models/breeze-4bit-sensitive-bf16-v2`
- Overall status: `audio_generated_evaluation_pending`

| Capability | Duration | Elapsed | RTF | CER | Cosine | Leakage | Clipping | Repeat tail | Stream break | Status |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| voice_design_en | 2.80s | 14.84s | 5.30 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_design_zh | 1.76s | 9.30s | 5.28 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| voice_clone_en | 2.32s | 6.23s | 2.69 | 0.000 | 0.743 | 0.169 | 0.000000 | no | no | audio_generated |
| voice_clone_zh | 10.48s | 26.49s | 2.53 | 0.000 | 0.656 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_en_to_zh | 4.96s | 12.50s | 2.52 | 0.091 | 0.692 | 0.000 | 0.000000 | no | no | audio_generated |
| cross_clone_zh_to_en | 5.60s | 13.94s | 2.49 | 2.300 | 0.641 | 0.044 | 0.000000 | no | no | audio_generated |
| voice_direction_en | 4.72s | 24.30s | 5.15 | 0.000 | 0.757 | 0.142 | 0.000000 | no | no | audio_generated |
| voice_direction_zh | 4.24s | 22.21s | 5.24 | 0.000 | 0.742 | 0.056 | 0.000000 | no | no | audio_generated |
| nonstream_en | 3.04s | 8.21s | 2.70 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| streaming_en | 3.84s | 9.69s | 2.52 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| stream_cancel | 3.52s | 11.54s | 3.28 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| long_text | 16.40s | 41.32s | 2.52 | 0.018 | pending | pending | 0.000000 | no | no | audio_generated |
| steady_state | 3.28s | 8.09s | 2.47 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| sampling_controls | 3.68s | 9.38s | 2.55 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| seed_reproducibility | 4.24s | 11.92s | 2.81 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_laugh | 3.76s | 11.11s | 2.96 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_cough | 3.76s | 10.02s | 2.66 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_clears_throat | 2.88s | 7.72s | 2.68 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_en_sigh | 4.24s | 11.39s | 2.69 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_laugh | 3.12s | 8.19s | 2.63 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_cough | 2.80s | 7.76s | 2.77 | 0.111 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_clears_throat | 2.48s | 6.79s | 2.74 | 0.000 | pending | pending | 0.000000 | no | no | audio_generated |
| event_zh_sigh | 5.04s | 14.18s | 2.81 | 0.100 | pending | pending | 0.000000 | no | no | audio_generated |

## Performance

- Model load: `2.303 s`
- Load peak memory: `5.498 GB`
- Run peak memory: `10.294 GB`
- Steady-state RTF: `2.509`
- Long-text RTF: `2.520`
- Streaming TTFA: `2.686 s`

## Validation gates

- Standard-content corpus CER: `0.0060`
- Maximum per-case CER (diagnostic): `0.0177`
- Minimum clone cosine: `0.6413`
- Minimum clone P10: `0.6324`
- Maximum reference leakage: `0.1695`
- Waveform integrity: `pass`
- PyTorch parity: `pass`
- Manual listening: `pending`

ASR, speaker similarity, leakage, and manual listening are explicitly `pending` unless their report fields contain measured evidence.
