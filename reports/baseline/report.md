# Breeze TTS 2 MLX 实机 Smoke 报告

- 日期：2026-08-31
- 机器：Apple M3 Max，128 GB 统一内存
- MLX 源码：`/Users/vanch/mlx-audio`，提交 `512f608`
- 模型：`LunaFox/Breeze-TTS-2-mlx-4bit`
- 模型修订：`27be05f01bd8aad9628022c2bac6ded0119eef8a`
- 上游模型修订：`c1c8ca18b70b30822735633991d9ebf4898e47d4`
- 许可：BreezeBlue Research and Non-Commercial License

## 结论

现有 `mlx-audio` Breeze TTS 2 实现并非架构骨架。真实 4-bit 权重在本机完成了英文声音设计、英文声音克隆、英文声音指导、中文事件文本和增量流式保存。五条样本的 Whisper 回读均与目标文本一致；所有 WAV 均为 24 kHz 单声道且未检测到削波。

声音克隆样本的 SpeechBrain ECAPA 原始余弦为 `0.7759`，分段 P10 为 `0.7745`，高于 `0.25` 验证阈值，判定为 `usable`。

这只能证明现有实现适合作为继续开发的基线，不能证明“原版全部功能”已经完成。

## 实机结果

| 能力 | 时长 | 耗时 | RTF | 峰值内存 | ASR | 状态 |
|---|---:|---:|---:|---:|---|---|
| 英文声音设计 | 2.72s | 23.66s | 8.70 | 4.20 GB | 完全匹配 | pass |
| 英文声音克隆 | 5.76s | 13.80s | 2.40 | 6.53 GB | 完全匹配 | pass |
| 英文声音指导 | 4.00s | 17.31s | 4.33 | 5.83 GB | 完全匹配 | pass |
| 中文声音设计与 `[笑]` | 3.60s | 14.85s | 4.12 | 5.71 GB | 正文匹配 | 事件听感 pending |
| 英文流式与 `(sigh)` | 4.16s | 26.02s | 6.25（整体推导） | 4.14 GB | 正文匹配 | 事件听感 pending |

流式路径产出 5 个 chunk。冷启动端到端首个 chunk 约 12.94 秒，不能与上游 H100 CUDA Graph 的 warmed-up TTFA 宣称直接比较。

## 风险与缺口

- 当前使用非官方 4-bit checkpoint；官方 BF16 到 MLX 的可复现转换尚未验证。
- 克隆和声音指导音量分别约为 `-34.8` 与 `-33.0 dBFS RMS`，明显低于声音设计样本。
- 事件是否真实可听仍需人工听测或专用事件检测，ASR 只能证明正文没有丢失。
- 现有 CLI 的逐 chunk RTF 使用累计耗时除以单 chunk 时长，后续 chunk 数值不可直接作为性能结论。
- 上游 HTTP 流式 API、原版 PyTorch 数值/行为对齐、批量并发以及独立 BF16/8-bit 转换仍为 `missing evidence`。

完整机器可读结果见 `summary.json`。
