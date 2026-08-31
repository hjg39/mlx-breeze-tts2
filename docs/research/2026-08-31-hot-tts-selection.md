# 2026-08-31 新 TTS 项目筛选与 Breeze TTS 2 立项依据

## 结论

Breeze TTS 2 仍是本项目的合理移植目标。它不是 GitHub stars 最高的候选，
但在发布新鲜度、X 传播、差异化能力和 Apple Silicon 原生实现缺口四项上
组合最强：官方于 2026-08-25 开源权重与 PyTorch 推理；公开接口同时覆盖
声音设计、声音克隆、声音指导、内联声音事件和单并发流式 PCM API。

本次选择不是质量排名。所有质量和速度结论仍以本地评测证据为准。

## GitHub 候选快照

数据读取于 2026-08-31；stars/forks 是易变快照，不应作为长期固定事实。

| 项目 | GitHub 热度 | 官方公开能力 | Apple Silicon 迁移价值 | 决策 |
|---|---:|---|---|---|
| [Breeze TTS 2](https://github.com/breezeblue-ai/breeze-tts) | 164 stars / 16 forks | 中英双语；设计、克隆、指导、事件、流式 API | 原版仅 CUDA；功能面完整，MLX 缺口明确 | 选择 |
| [Audio8 TTS](https://github.com/Audio8-AI/Audio8_TTS) | 1.2k stars / 103 forks | 0.6B preview、11 个推荐语言、零样本克隆 | 规模较小且许可宽松，但 preview 能力主要是多语种克隆 | 候补 |
| [Sopro](https://github.com/samuel-vitorino/sopro) | 937 stars / 38 forks | 120M、四语种、克隆、流式；官方报告 M3 CPU RTF 0.24 | 已有 CPU/MPS/ONNX 路径，MLX 移植新增价值较低 | 候补 |
| [smalltts](https://github.com/smallbraineng/smalltts) | 63 stars / 6 forks | 克隆、声音事件、ONNX、训练代码 | 新且轻量，但公开成熟度与验证面较窄 | 观察 |
| [Wren-TTS](https://github.com/shangeth/wren-tts) | 3 stars / 0 forks | 360M/0.5B、Mimi codec、8 语种克隆、表达标签 | 架构适合 MLX，但当前社区热度不足 | 观察 |
| [LuxTTS-mlx](https://github.com/jishnuvenugopal/LuxTTS-mlx) | 1 star / 0 forks | 已有 MLX、48 kHz 克隆 | 已完成独立 MLX 路径，不是新的移植缺口 | 排除 |

## X 热度证据

通过用户现有 Chrome 登录态读取 X 的热门搜索结果；只做只读核查，未发帖、
点赞或改变账户状态。

| 帖子 | 日期 | 可见互动快照 |
|---|---|---:|
| [BreezeBlue 官方开源帖](https://x.com/BreezeBlueX/status/2092647083132273018) | 2026-08-27 | 123,657 浏览、354 喜欢、44 转帖、21 回复、305 书签 |
| [Artificial Analysis 相关帖](https://x.com/ArtificialAnlys/status/2092399623839326550) | 2026-08-26 | 384,176 浏览、738 喜欢、72 转帖、18 回复、594 书签 |
| [Hugging Apps 演示帖](https://x.com/HuggingApps/status/2092532619598614848) | 2026-08-26 | 67,380 浏览、244 喜欢、28 转帖、7 回复、259 书签 |
| [BreezeBlue 技术理念帖](https://x.com/BreezeBlueX/status/2093784424026714342) | 2026-08-30 | 1,059 浏览、9 喜欢 |

这些数字证明短期传播热度，不证明模型能力或质量。

## 证据冲突与边界

- X 上的转述内容出现“支持 50 种语言”，但
  [官方 GitHub README](https://github.com/breezeblue-ai/breeze-tts)
  明确写的是 English/Chinese bilingual support。本项目只声明和测试中英双语。
- 官方 H100 的 `<40 ms` TTFA 与 `0.32 RTF` 是 warmed-up CUDA fast path，
  不能外推到 Apple Silicon MLX。
- GitHub/X 热度只用于项目发现和优先级判断；功能完成必须由源码/API 对齐证明，
  质量与速度必须由本机 WAV、ASR、speaker similarity、RTF 和听测证明。
- 模型权重、衍生模型与自托管输出受研究非商用许可约束；源码许可不能替代模型许可。

## 选择标准

1. 2026 年新发布且源码、权重和接口可审计。
2. X 或 GitHub 至少一个渠道有可观察热度，且不是只靠项目自述。
3. 原版功能面足够独特，移植后不只是另一个基础克隆器。
4. Apple Silicon 尚缺独立、可测试、无 `mlx_audio` 生产依赖的完整运行时。
5. 权重规模适合 M3 Max 128 GB；本地磁盘条件单独作为执行门禁。

按上述标准，Breeze TTS 2 的选择成立；Audio8 TTS 与 Sopro 保留为下一轮候选，
但不在当前已批准独立项目中扩张范围。
