# MLX Breeze TTS 2 独立移植设计

## 1. 文档状态

- 日期：2026-08-31
- 状态：设计已获用户确认，等待文档复核
- 目标项目：`/Users/vanch/mlx-breeze-tts2`
- 目标平台：Apple Silicon macOS，首测机器为 Apple M3 Max、128 GB 统一内存
- 上游源码：`breezeblue-ai/breeze-tts`
- 上游源码基线：2026-08-25 发布的官方 PyTorch 推理实现；实施开始时固定精确 Git 提交
- 上游模型：`BreezeBlue/Breeze-TTS-2`
- 已审计模型修订：`c1c8ca18b70b30822735633991d9ebf4898e47d4`
- 已验证 MLX 参考：`Blaizzy/mlx-audio` 中 Breeze TTS 2 实现，源码提交 `512f608`
- 已验证 4-bit 参考模型：`LunaFox/Breeze-TTS-2-mlx-4bit`，修订 `27be05f01bd8aad9628022c2bac6ded0119eef8a`

## 2. 背景与结论

Breeze TTS 2 于 2026-08-25 开放权重和 PyTorch 推理代码，提供中英文声音设计、声音克隆、带参考音频的声音指导、文本内声音事件和实时流式生成。官方运行时依赖 Linux、PyTorch 与 CUDA，并以 H100 的 CUDA Graph 快路径报告低延迟。

本机现有 `mlx-audio` 已包含初版 Breeze MLX 实现。2026-08-31 的真实权重 smoke 证明其架构并非骨架：30 项单元测试通过，英文声音设计、英文克隆、英文声音指导、中文事件文本和流式保存均成功；五条样本的 Whisper 回读匹配目标正文，克隆 ECAPA 原始余弦为 `0.7759`，判定 `usable`。但是该实现依赖通用 `mlx-audio` 运行面，只验证了非官方 4-bit 权重，缺少官方 BF16 可复现转换、官方 HTTP API、完整事件听测和 PyTorch 行为对齐。

因此采用独立项目和证据驱动的混合移植路线：以官方 PyTorch 实现为唯一行为契约，审计后迁入现有已验证 MLX 算法，但生产推理不依赖 `mlx-audio`。所有来源、许可证、权重修订和验证结果必须可审计。

## 3. 目标与非目标

### 3.1 目标

1. 建立独立的、可安装的 `mlx-breeze-tts2` Python 项目。
2. 在 MLX 中实现官方推理模型的完整计算图：T5Gemma2 文本编码器、Qwen3 主干、深度解码器、16-codebook 生成以及 Qwen3-TTS 音频 tokenizer。
3. 保留官方公开功能：声音设计、声音克隆、声音指导、中英文声音事件、非流式生成和增量流式生成。
4. 提供独立 Python API、与官方语义一致的 CLI、`/health` 与 `/v1/audio/speech` HTTP API。
5. 提供从官方 checkpoint 到 MLX BF16、8-bit 和 4-bit 的可复现转换，严格验证权重键与形状。
6. 在本地实机生成 JSON、Markdown、WAV 和网页听测报告，并以运行证据判定完成度。

### 3.2 非目标

- 不复刻 CUDA Graph、Flash Attention 2 或 H100 专属实现细节。
- 不承诺与 H100 的 TTFA 或 RTF 数值相同。
- 不扩展官方未公开支持的语言、并发批处理、多说话人单请求或训练能力。
- 不把量化版本要求为逐采样点等同于 BF16 或 PyTorch；要求功能、正文、克隆身份和控制行为通过统一验收。
- 不移除或绕过上游模型的研究与非商业许可限制。

## 4. “原版全部功能”的边界

以下能力全部需要可运行证据，不能仅由 README 或单元测试宣称：

1. **Voice Design**：无参考音频，使用自然语言 instruction 设计音色和表达。
2. **Voice Clone**：参考音频与精确 `ref_text` 成对输入，生成目标正文并保留说话人身份。
3. **Voice Direction**：同时使用参考音频、参考转写和 instruction，在保留身份的同时控制语气、情绪、速度和表达。
4. **Vocal Events**：英文 `(laugh)`、`(cough)`、`(clears throat)`、`(sigh)`；中文 `[笑]`、`[咳嗽]`、`[清嗓子]`、`[叹气]`。
5. **Non-streaming**：生成完整 24 kHz 单声道 WAV 或内存数组。
6. **Streaming**：增量产生连续 PCM chunk；拼接结果无缺口、重复或尾部状态泄漏。
7. **Sampling Controls**：seed、CFG、temperature、top-p、top-k、repetition penalty、最大生成 token。
8. **CLI**：官方输入语义、默认值和参数校验；MLX 转换与服务命令作为独立扩展。
9. **Python API**：加载一次模型并执行设计、克隆、指导、非流式与流式请求。
10. **HTTP API**：`/health` 和 multipart `/v1/audio/speech`，返回 PCM16 小端流及官方响应头。

## 5. 架构

```text
官方 Breeze 源码与 checkpoint
        │
        ▼
转换器与严格权重审计
        │
        ▼
MLX 核心模型
├── T5Gemma2 双向文本编码器
├── Qwen3 自回归主干与 KV cache
├── 16-codebook 深度解码器
└── Qwen3-TTS codec 编码/解码
        │
        ▼
统一运行时
├── Voice Design
├── Voice Clone
├── Voice Direction
├── Non-streaming
└── Incremental Streaming
        │
        ├── Python API
        ├── CLI
        └── FastAPI 服务
                │
                ▼
单元对齐、实机矩阵、ASR、SIM、泄漏、音频异常、RTF、TTFA、听测
```

### 5.1 建议目录

```text
mlx-breeze-tts2/
├── mlx_breeze_tts2/
│   ├── model/
│   │   ├── config.py
│   │   ├── text_encoder.py
│   │   ├── backbone.py
│   │   ├── depth_decoder.py
│   │   └── model.py
│   ├── codec/
│   ├── conversion/
│   │   ├── convert.py
│   │   ├── quantize.py
│   │   └── audit.py
│   ├── runtime/
│   │   ├── templates.py
│   │   ├── sampling.py
│   │   ├── streaming.py
│   │   └── engine.py
│   ├── api.py
│   └── cli.py
├── benchmark/
├── tests/
├── docs/
├── .ai_project.md
├── .ai_memory.md
├── LICENSE
├── NOTICE
├── pyproject.toml
└── README.md
```

模块边界以稳定接口隔离：模型层不导入 CLI 或 FastAPI；转换层不参与运行时推理；codec 可独立做编码、完整解码和增量解码测试；服务层只调用统一 engine。

## 6. 关键数据流

### 6.1 声音设计

1. 校验正文、instruction 和采样参数。
2. 使用 `[S0]<ins_bos>instruction<ins_eos>text` 构造正向 prompt。
3. CFG 启用时以无 instruction prompt 构造负向分支。
4. 文本编码器产生条件表示，主干逐帧生成第一 codebook。
5. 深度解码器生成其余 15 个 codebook。
6. codec 生成完整音频或增量 chunk。

### 6.2 声音克隆与指导

1. `ref_audio` 与非空 `ref_text` 必须同时存在。
2. 参考音频重采样为 codec 所需格式并编码为 16-codebook token。
3. 克隆 prompt 顺序为参考转写、参考音频 token、目标正文。
4. 声音指导在目标正文前加入 instruction；CFG 负向分支保留参考身份但去除 instruction。
5. 不以目标正文冒充参考转写；参考转写错误作为输入错误或评测风险明确记录。

### 6.3 流式生成

1. 每个请求建立独立主干 KV cache、codec streaming state 和计时器。
2. 以 codec frame rate 将 `streaming_interval` 转为 chunk frame 数。
3. chunk 指标分别记录首块耗时、当前块生成耗时、累计耗时和累计音频时长；不使用累计耗时除以单块时长作为 RTF。
4. 正常结束、异常、取消和客户端断开都重置 codec 状态。
5. 拼接全部 chunk 后必须与同参数完整解码在正文和连续性上等价。

## 7. 公共接口

### 7.1 Python API

核心入口提供 `load_model()` 与长期存活的 `BreezeEngine`。生成请求统一接受：

- `text`
- `instruction`
- `ref_audio`
- `ref_text`
- `speaker`，默认 `S0`
- `seed`，默认 `42`
- `cfg_scale`，默认 `1.0`
- `temperature`，默认 `0.9`
- `top_p`，默认 `1.0`
- `top_k`，默认 `50`
- `repetition_penalty`，默认 `1.1`
- `max_new_tokens`，默认与官方一致为 `1500`
- `stream`
- `streaming_interval`

非流式返回完整音频、采样率、token 数、音频时长、耗时和内存；流式返回包含 `audio`、`sample_rate`、chunk 序号、是否最终块、TTFA/块耗时/累计耗时的结果对象。

### 7.2 CLI

保留官方生成语义，并提供明确子命令：

- `mlx-breeze generate`
- `mlx-breeze serve`
- `mlx-breeze convert`
- `mlx-breeze audit-checkpoint`
- `mlx-breeze benchmark`

所有路径必须支持绝对路径和 Hugging Face repo/revision；最终报告记录实际解析后的 revision。

### 7.3 HTTP API

- `GET /health`
  - 未加载：503 与 `{"status":"loading"}`。
  - 已加载：200 与 `{"status":"ok","sample_rate":24000}`。
- `POST /v1/audio/speech`
  - multipart 字段仅包含官方公开参数：`text`、`instruction`、`cfg_scale`、`ref_audio`、`ref_text`、`seed`。
  - 返回 `audio/pcm`、PCM16 little-endian。
  - 响应头包含 `X-Sample-Rate: 24000`、`X-Sample-Format: s16le`、`Cache-Control: no-store`。
  - 单并发；已有请求时返回 409。

CLI/Python 可公开更多采样参数；HTTP 兼容端点不擅自扩张官方签名。需要扩展时另建版本化端点，不改变 `/v1/audio/speech`。

## 8. 权重转换与量化

1. 固定官方模型 revision并保存来源元数据。
2. 读取 safetensors index，对全部 1115 个已审计权重项建立确定性映射。
3. 转换完成后要求 `missing=0`、`unexpected=0`、shape mismatch 为零。
4. 保留 tokenizer、generation config、audio tokenizer 配置和许可证文件。
5. BF16 为行为对齐基准；8-bit 和 4-bit 从已验证 BF16 MLX artifact 生成。
6. 量化规则按模块显式声明；norm、小尺寸 embedding、敏感 codec 层是否保留 BF16由消融证据决定。
7. artifact metadata记录上游 revision、转换器版本、量化参数、文件哈希和生成时间。
8. 任何宽松加载只能用于诊断，不能进入发布或最终 benchmark。

## 9. 错误处理与资源安全

- 空正文、非法 CFG、temperature、top-p、top-k、repetition penalty、seed 或生成长度立即失败。
- 参考音频与参考转写不成对时立即失败。
- 空上传、不可解码音频、无效采样率和缺失 checkpoint给出具体错误。
- API 输入错误返回 400；并发冲突返回 409；未加载返回 503；内部错误保留结构化日志但不泄漏本地敏感路径。
- 上传文件在成功、异常和客户端断开时删除。
- OOM、提前 EOS、零帧、空音频、codec 异常和状态重置失败进入结构化诊断。
- 不通过裁剪掩盖空音频、错读、重复或尾部异常；必须先定位根因。

## 10. 测试与证据

### 10.1 单元与组件对齐

- 配置解析、RoPE、T5Gemma2 full/sliding mask、RMSNorm和 KV cache。
- 文本/audio embedding及权重映射。
- 第一 codebook 与其余 15 codebook采样。
- EOS、reserved token mask、CFG、repetition penalty和随机 seed。
- codec 完整编码/解码、增量解码和状态重置。
- CLI 参数别名、默认值及错误。
- API 路由、签名、状态码、PCM16编码、响应头和临时文件清理。

### 10.2 PyTorch 对齐

- 对固定小输入比较模板渲染、token ID、参考音频 code、mask、权重形状和关键中间张量。
- BF16 在合理数值容差内比较组件输出；容差按算子和张量 dtype明确记录。
- temperature 0 或固定 seed 的短样本比较 token 生成行为。
- 完整波形不要求逐采样点相同，但正文、时长范围、事件、身份和控制方向必须通过统一验证。

### 10.3 实机功能矩阵

至少覆盖：

- 英文/中文声音设计。
- 英文/中文同语种克隆。
- 英文参考说中文、中文参考说英文的跨语言克隆，作为探索项单独报告。
- 英文/中文声音指导。
- 八个公开声音事件逐项听测。
- 非流式、流式、流式拼接与客户端取消。
- BF16、8-bit、4-bit 三种 artifact。
- 冷启动单条、同进程稳态多条与长文本。

### 10.4 完成门禁

以下条件必须全部有证据：

- 三种 artifact 均可严格加载。
- 标准正文 CER 不高于 `0.05`。
- 克隆 ECAPA 原始余弦与分段 P10均高于 `0.25`。
- 参考文本泄漏分数低于 `0.65`。
- 无空音频、严重削波、重复尾部和流式断裂。
- 八个事件均有人工听测结论；事件未听见则该能力为 fail。
- Python API、CLI、HTTP API以及非流式/流式均通过。
- 固定 seed 可复现；采样参数变化确实进入对应采样路径。
- API 参数、状态码、PCM格式和响应头与官方契约一致。
- RTF、TTFA、峰值内存、模型加载时间和音频时长均被记录。
- JSON、Markdown、WAV和 HTML 听测报告完整生成。

RTF 大于 1 不自动判定功能失败，但必须列为性能缺口。不得把 H100 warmed-up TTFA与 M3 Max 冷启动 TTFA直接比较。

## 11. 交付物

1. 独立源代码仓库 `/Users/vanch/mlx-breeze-tts2`。
2. 可安装 Python 包与锁定依赖。
3. 官方 BF16、8-bit、4-bit 转换和审计命令。
4. Python、CLI、HTTP 使用文档。
5. 自动化测试与 PyTorch 对齐测试。
6. 实机 benchmark manifest、每条 WAV、JSON、Markdown和 HTML 听测报告。
7. 能力摘要：适用场景、不适用场景、性能、许可和已知失败模式。
8. `.ai_project.md`、不超过五行增量的 `.ai_memory.md` 以及关键决策记录。

## 12. 实施顺序

1. 建立独立项目、许可证与来源清单。
2. 迁入并隔离已验证 MLX 组件，保持原子提交。
3. 完成官方 BF16 转换和严格权重审计。
4. 对齐模板、token、模型前向、采样和 codec。
5. 完成 Python API和 CLI。
6. 完成 HTTP 流式服务和资源清理。
7. 修正流式指标，建立冷启动与稳态计时。
8. 生成 8-bit、4-bit并做量化消融。
9. 运行完整实机矩阵、ASR、泄漏、声纹、异常和听测。
10. 完成能力边界、报告、项目记忆和交付审计。

## 13. 完成定义

只有第 10.4 节的所有门禁均为 `pass`，且交付物完整存在，才能声明“Breeze TTS 2 原版全部功能的独立 MLX 移植完成”。任何缺少官方 BF16 转换、HTTP API、事件听测、真实权重实机结果或流式验证的状态都只能标为 `pending` 或 `missing evidence`。

CUDA Graph/H100 性能不属于功能等价，但本地性能结果必须完整、可复现并与测试范围匹配。模型权重、衍生模型和生成输出继续受 BreezeBlue Research and Non-Commercial License约束。
