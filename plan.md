# Claude 辅助音乐制作 — 计划

## 核心理念

~~用 Claude + REAPER MCP 屏蔽 DAW 的复杂度，让用户专注于"听和判断"。~~ （Phase 1.5 实验证伪）

**修正后的认知（2026-06-15）：**
Claude 能当高效的"REAPER 脚本执行器"，但不能当编曲助手。缺的不是手（工具够了），是耳朵和审美。原计划的"Claude 分析 → 用户判断"模型方向反了。DAW 辅助制作路线已废弃，转向 agent 音频理解研究方向。

## 目标风格

- 主方向：DnB（170-176 BPM）
- 子类探索：Liquid DnB, Neuro DnB, Artcore
- 用户背景：音游玩家，主听 EDM（DnB / Hardcore / Trance / Dubstep），大量听歌积累

## 环境搭建（Phase 1） ✅ 已完成

- [x] 安装 REAPER — v7.74, `C:\Program Files\REAPER (x64)`
- [x] 安装 REAPER MCP Server — xDarkzx/Reaper-MCP v0.3.0, Lua 文件桥接, 163 工具
  - 仓库: `C:\Users\Yufeng Ying\Reaper-MCP`
  - Lua 脚本: `reaper_scripts\reaper_mcp_server.lua`（需在 REAPER 中保持运行）
- [x] 在 Claude Code 中配置 MCP 连接 — 用户级 `reaper-mcp` stdio server
- [x] 确认已有 VST：Serum 1.35b1（`C:\Program Files\Steinberg\VSTPlugins\Serum_x64.dll`）
- [x] OTT (Xfer Records) — 从 FL Plugins 复制到用户级 VST2 目录
- [x] 验证 MCP 基本操作：创建轨道、写入 MIDI、加载 VST、插入音频采样
- [ ] ~~安装免费 VST 音源~~ — 暂缓，见 Phase 1 实验结论

### FL Studio 资源复用盘点
- **可用 VST**: Serum (x64) + SerumFX + OTT + Sylenth1 (免安装版)
- **Serum 预制**: Kawaii Bass (~250 fxp), UK Hardcore/Stonebank (~137 fxp) — 已在 Serum Presets 目录
- **采样包**: KSHMR Vol.3 (~4800 files), Kawaii Bass, UK Hardcore, 人声/干声等
- **FL 原生插件**: Sytrus/Harmor/Gross Beat 等均为 FL 专有格式，不可用于 REAPER

## 审美校准（Phase 1.5） ⚠️ 实验失败，需重新设计

### 已验证的工具链
- [x] Demucs 4-stem 分轨（htdemucs）— drums/bass/vocals/other，CPU 模式可用
- [x] scipy/numpy 频谱分析 — RMS、spectral centroid、onset detection、chroma
- [x] ffmpeg 切片 + 自动导入 REAPER
- [ ] ~~librosa~~ — numba 依赖安装失败，用 scipy 替代
- [ ] ~~Basic Pitch~~ — 未安装

### 实验结论（2026-06-15）

**在 Sphalerite (CS4W, RYOQUCHA) 上测试了两个段落：**

1. **Bars 79-95（间奏→trap buildup）**: Claude 将 buildup 误判为 drop，未识别出钢琴主导→trap 过渡
2. **Bars 41-56（complextro 1st drop）**: Claude 将 complextro 音色拼贴误判为"旋律层加厚"，未识别出 bass 类效果器的瞬态拼贴特征

**核心瓶颈不在工具，在能力：**
- 没有听觉 → 数字能算出来但解读全靠猜，两次结构判断都错了
- 无法区分音色类型 → growl / reese / wobble / stab 在频谱上差异不够显著
- Demucs 4-stem 分类按频段而非乐器 → 钢琴低音区进 bass 轨，FX 散落各轨
- 变速曲目 BPM 适配完全做不到

**结论：Claude 能当高效的"REAPER 脚本执行器"，但不能当编曲助手。缺的不是手（工具够了），是耳朵和审美。**

原计划的"Claude 分析 → 用户判断"模型方向反了，可行的分工是"用户描述结构 → Claude 量化执行"。

## ~~流程验证（Phase 2）~~ ❌ 已废弃

随 DAW 辅助路线一起废弃。Claude 无法做音色预筛（MCP 读不到 Serum preset）、无法判断生成结果好不好（没有听觉）。

## ~~扒带分析~~ ❌ 已废弃

依赖 Claude 理解音频内容，Phase 1.5 实验已证明不可行。扒带的核心需求（音频→工程参数逆向）转入下方研究方向。

## ~~完整制作（Phase 3）~~ ❌ 已废弃

依赖 Phase 2 验证通过，前置条件不成立。

## 开放研究方向：Agent 音频理解问题（2026-06-15）

### 问题定义

让无音频感知能力的 agent 理解乐曲。这里的"理解"不是信号分析，而是**逆向工程**——从成品还原原始工程设计（扒带）。

### 为什么现有方案不够

**实验证伪的路线：**
- Demucs 分轨 + 频谱分析 → agent 能算数字但解读全错（buildup 当 drop，complextro 拼贴当旋律加厚）
- 正向生成 + 特征匹配逆向 → 只能逆向自己能合成的东西，无法处理非平凡组件
- embedding 相似度 → 绕过理解不等于理解，无法还原工程参数

**人类扒带依赖的三样东西 agent 都缺：**
1. 先验知识 — 听到音色知道合成方式，因为自己做过
2. 经验积累 — 知道特定风格的惯用手法，不需要从零推理
3. 试错验证 — 在 DAW 里复刻，听对不对，调了再听

### 核心难点

- **审美输入问题** — agent 需要外部数据校准审美方向，但无法"听"参考曲目
- **非平凡组件** — 即使是 trance 这种相对程式化的风格，优秀作品也有大量非平凡的音色设计和 FX 处理，不能只分析简单的东西
- **复杂度与自然语言理解同级** — 人类花几十年练耳才能扒带，特征匹配绕不过去
- 这是一个**开放的、悬而未决的问题**，目前没有靠谱方案

### 可能的探索方向（未验证）

1. **Agent-native DAW** — 不走现有 DAW，造一个 API-first 的合成环境，音色是代码不是 preset，所有操作可编程。解决"手"的问题但不解决"耳朵"的问题
2. **多模态模型原生音频理解** — 等模型本身能处理音频输入（Gemini 已有初步能力），但不在我们能控制的范围内
3. **Trance 作为约束场景** — 结构程式化、音色类型化，参数空间相对有限，可能是最接近可行的切入点。但"好听的 trance"和"能跑的 trance"之间差距巨大
4. **人在回路的渐进式知识积累** — 每次用户纠正 agent 判断时结构化存储，逐步建立映射。本质是用对话训练认知，效率存疑

### 相关研究调研（2026-06-15）

**音频逆向 + RL：**
- SynthRL（IJCAI 2025）— RL 训练模型从音频反推合成器参数，音频距离作为 reward，优于监督学习，能跨域泛化。但限于单合成器参数估计
- InverSynth 系列 — 监督学习做合成器参数逆向，需要大量 (参数, 音频) 配对数据
- 结论：RL + 音频距离作为 verifiable reward 这条路被验证了，理论上可扩展到更复杂的搜索空间

**音乐生成 AI 开源方案：**
- ACE-Step 3.5B — LM 做结构规划 + DiT 做声学渲染，Apache 2.0，与 agent 工作流天然契合
- YuE 7B — LLaMA2 架构，万亿 token 训练，Apache 2.0
- MusicGen (Meta) — Transformer + EnCodec tokenizer，开源
- Suno/Udio — 闭源，不可复用
- 这些方案全部绕过工程状态，直接 文本→音频，无法精确控制

**音频理解 LLM（2024-2026）：**
- 通用多模态 LLM（Gemini 2.5、GPT-4o、Qwen2.5-Omni）支持音频输入，但音乐理解不是重点
- 音乐专用模型：乐器分类 91%、音高识别 77%、和弦识别 59%、段落切分可用但粗糙（MuQ 2025、Chordformer 2025）
- 所有模型都在做分类/标注，没有任何模型在做音频→工程参数的逆向
- Audio tokenizer 趋势：从波形保真（EnCodec）转向语义感知（EntangleCodec 2026、ALMTokenizer 2025），语义 token 显著优于压缩 token

### 深层问题分析

**工程状态表示的断层：**
- 工程→音频是无损/确定性的，工程本身是结构化的，LLM 理解没问题
- 但音频→工程是逆向问题，目前无解
- 现有音乐 AI 全部绕过工程状态（文本→黑箱→音频），快速出成果但丧失精确控制力
- 自然语言信息带宽远低于工程文件，无法承载音频的无损表示

**两个核心瓶颈：**
1. 感知瓶颈（听不了）— 音频 LLM 有初步能力但离"听懂编曲"差一个量级，需要训练解决
2. 先验瓶颈（没见过）— 公开完整工程文件极少，各 DAW 格式不互通（.flp/.als/.logicx/.rpp 全是私有格式），唯一可移植的 MIDI 和音频又分别丢失了音色信息和结构信息。没有"音乐工程的 GitHub"

**RLVR 类比：**
- 音频逆向工程有明确的 verifiable reward（合成结果与目标的音频距离）
- 这使得 RL 训练路线在理论上成立（SynthRL 已验证最小情形）
- 但从单合成器扩展到完整工程，搜索空间爆炸式增长

### 当前状态

实验暂停。各零件分散存在但无人组装：
- SynthRL 证明了 RL 逆向路线可行（单合成器级别）
- 语义 audio tokenizer 在进化（EntangleCodec 2026）
- 音乐理解模型在做分类级听觉（MuQ 2025）
- 但"从音频还原完整工程状态"这个任务目前是空白

### 更深层的思考（2026-06-15 讨论）

**音频理解 ≈ 语言理解，但有关键差异：**

相似：
- 原始状态空间都巨大，人类只用了一部分主要通道（语言有语法/语义，音乐有 drum/bass/lead 分轨范式）
- 可以期望 scaling law 起作用——模型通过海量数据形成自己的高维表征
- 人类初学者同样面临"听不懂"困境，需要多年 try loop 建立直觉

关键差异：
1. **音频是 second class** — 文本自我描述，音频需要另一个模态（语言）来描述，而语言带宽远低于音频。人脑里两个表征物理共存但不完全重叠，强行把音频塞进语言空间一定有信息损失
2. **没有正确答案** — 音乐审美是被文化塑造的、主观的、历史性的。噪声从"不是音乐"变成一种音色（glitch），边界持续移动。不能定义固定的 reward function 说"好听"
3. **时间结构** — 音频本质上时间相关，同一个和弦在 buildup 和 drop 里意义完全不同。时间分辨率毫秒级，连续而非离散

**问题应拆成两层：**
- 工程逆向（音频→工程参数）— 有 ground truth（合成距离），可以 RL
- 审美判断（工程设计好不好）— 没有 ground truth，只能靠人的偏好（类似 RLHF）

**架构问题：**
- Transformer 不是为音频设计的，处理音频要么先 tokenize（有损）要么处理频谱图（计算量爆炸）
- State Space Models（Mamba 等）天然处理长序列连续信号，可能更 audio-native
- 但 Transformer 也不是为语言"设计"的，scaling law 让它 work 了

**音乐体系的边界远超想象：**
- 噪声音乐、无调性、实验音乐（敲钢琴板子）都是音乐
- 和声学远远覆盖不了这个范畴
- 理解音乐的模型必须能处理这种多样性，不能只针对调性音乐

### 当前状态

实验暂停，计划后续探索 audio-native 模型方向。各零件分散存在但无人组装：
- SynthRL 证明了 RL 逆向路线可行（单合成器级别）
- 语义 audio tokenizer 在进化（EntangleCodec 2026）
- 音乐理解模型在做分类级听觉（MuQ 2025）
- 但"从音频还原完整工程状态"这个任务目前是空白

### 先期技术调研（2026-06-15）

计算资源：H100 可用，4B 以下模型可训练。

#### 合成器参数估计（学界现状）

| 工作 | 方法 | 合成器 | 关键点 |
|------|------|--------|--------|
| InverSynth (2019) | 监督 CNN | 减法 | 开创性工作 |
| DDSP (Google, ICLR 2020) | 可微分 DSP | 加法（谐波+噪声） | 开创性框架。神经网络预测 DSP 参数而非直接生成波形，10-15min 数据即可训练，可参考可微分模块设计 |
| DDX7 (ISMIR 2022) | DDSP + TCN | FM (DX7) | 固定频率比+外部 pitch 规避 FM 频率不收敛问题，~400K 参数，CPU 实时 32ms，**FM 方向最实用** |
| DWTS (ICASSP 2022) | 可微分波表 | 波表 | 波表=可学习字典，10-20 个波表≈DDSP 100 谐波质量，推理快 12x，grid_sample 天然可微 |
| Sound2Synth (2022) | LSTM+VAE+CNN | FM (Dexed/DX7) | FM preset 估计 |
| NAS-FM (IJCAI 2023) | NAS + 进化搜索 | FM | 自动搜索 FM 拓扑（DDX7 用固定 DX7 algorithm），优于手工设计，支持音色插值 |
| DiffMoog (2024) | 可微分模块化合成器 | 模块化（减法+FM） | signal-chain loss（每级做损失）+ Wasserstein loss（缓解频率不收敛）。最全面但 FM 部分不收敛 |
| AST Sound Matching (2024) | Audio Spectrogram Transformer | Massive | 显著优于 MLP/CNN |
| PNP Loss (2024, IEEE TASLP) | 感知-神经-物理联合损失 | 通用 | 平衡感知相关性和计算效率 |
| SynthRL (IJCAI 2025) | RL + 音频距离 reward | 通用 | 跨域泛化优于监督学习。架构：mel→2D CNN→Transformer enc→Transformer dec (learnable queries + cross-attn)→参数 |
| Flow Matching (ISMIR 2025) | 条件流匹配 | 对称空间 | 处理参数对称性（如两个 osc 互换后音色不变） |
| Modulation Extraction (WASPAA 2025 Best Paper Candidate) | DDSP + 调制提取 | — | 能发现 LFO/envelope 调制结构 |

共同模式：全部是单合成器级别，数据集全部正向合成生成。

**FM 方向专项结论（2026-06-15 讨论）：**
- DDX7（固定频率比+外部 pitch）和 NAS-FM（自动搜索拓扑）成功的关键都是**回避频率的端到端优化**
- DiffMoog 试图硬刚 FM 频率参数，loss landscape 高度非凸（大量局部最小值），不收敛
- FM 在实际 EDM 制作中占比仅 ~10%，研究价值在于参数耦合是最难的 case，但不应是引擎的优先级

#### 可编程合成引擎

| 工具 | 特点 | 适用 |
|------|------|------|
| **torchsynth** | GPU 加速 PyTorch 模块化合成器，16200x 实时速度，自带 synth1B1 十亿样本数据集 | **大规模训练数据生成** |
| **DawDreamer** | Python JUCE wrapper，可加载 VST3/AU + Faust 代码，支持 Faust→JAX 可微分 | **最全面，可加载 Serum/Vital** |
| **Pedalboard** (Spotify) | Python 效果器库，支持 VST3 宿主，300x faster than pySoX | **效果器链** |
| **SynthAX** | JAX 模块化合成器，可微分 | 需要梯度回传时 |
| **SpiegeLib** | 自动合成器编程研究库，含评估框架 | 实验评估 |

建议组合：torchsynth (合成) + Pedalboard (效果器) + DawDreamer (VST 集成)

#### 音频 Tokenizer

| Tokenizer | 码率 | 特点 | 开源 |
|-----------|------|------|------|
| EnCodec (Meta 2022) | 1.5-24 kbps | RVQ 多 codebook，波形重建为主 | ✅ |
| DAC (Descript 2023) | 8-16 kbps | 改进 RVQGAN，16kbps 优于 EnCodec 24kbps | ✅ |
| WavTokenizer (2024) | 超低码率 | **单 codebook**，40-75 token/秒 | ✅ |
| **SemantiCodec** (2024) | 0.31-1.40 kbps | **双编码器**（AudioMAE 语义 + 声学），语义保留显著优于声学 codec | ✅ |
| **X-Codec** (AAAI 2025) | — | RVQ 前注入预训练语义特征，解决声学 codec 语义缺失 | ✅ |

关键发现：Codec-SUPERB 基准测试证实 SemantiCodec 在极低码率下语义信息显著优于所有声学 codec。**逆向工程需要语义而非波形保真，SemantiCodec / X-Codec 最相关。**

#### 音频表示 / Feature Extractor

| 模型 | 参数量 | 特点 |
|------|--------|------|
| **MERT** (2023) | 330M | BERT 式自监督，14 个 MIR 任务 SOTA，可直接做 feature extractor |
| **MuQ** (2025) | — | MERT 改进，流派/歌手/结构/乐器分析更强 |
| **CLAP** | — | 音频-文本对比学习（类 CLIP），零样本分类 |
| **Essentia** (MTG) | — | C++ 核心，内置 87 种流派分类（含电子音乐子类），预训练调性/节拍/情绪模型 |

#### Audio-native 模型架构

| 架构 | 特点 | 音频应用 |
|------|------|---------|
| **SaShiMi** (S4, ICML 2022) | 多分辨率 SSM，自回归波形生成 SOTA | 首个 S4 音频应用 |
| **Mamba** (2023) | 选择性 SSM，线性复杂度 | 理论适合但音频专用论文少 |
| 线性 RNN (2025) | 与 SSM 等价 | 长音乐样本自回归生成 |

SSM/Mamba 长序列效率优（线性 vs 二次），Transformer 短序列质量仍更强。

#### 音频距离度量（RL Reward 设计）

| 度量 | 层级 | 适用 |
|------|------|------|
| Mel-spectrogram L1/L2 | 低层声学 | 训练 loss，可微 |
| Multi-resolution STFT loss | 低层声学 | 多尺度频谱匹配 |
| **CDPAM** | 感知 | 和人类评分高相关，适合做 reward |
| **MERT/CLAP embedding 距离** | 高层语义 | 捕捉音乐级相似性 |
| FAD | 分布级 | 评估生成质量，不适合单样本 |

建议 reward：**低层 mel loss + 高层 MERT embedding 距离**，覆盖声学保真和语义相似。

#### EDM/DnB 音色来源分析（2026-06-15 讨论）

真实 EDM 制作中各合成方式的占比估算：

| 来源 | 占比 | 典型音色 | 可微分难度 |
|------|------|---------|-----------|
| 减法合成 | ~35% | Reese bass（失谐锯齿波）、supersaw、模拟 pad/lead、sub bass | 低 |
| 采样 | ~30% | 鼓组 one-shot、经典 break（Amen/Think）、人声、氛围 | N/A（检索问题） |
| 波表合成 | ~20% | Serum/Vital 类音色、growl/wobble、morphing 音色 | 低（DWTS 已解决） |
| FM 合成 | ~10% | 电钢琴、铃声、FM bass、金属质感 | 高（频率耦合不收敛） |
| 其他 | ~5% | 物理建模、granular | 暂不考虑 |

**关键认知：**
- FM 虽是学术难点但实际占比最小，减法 + 波表 + 采样覆盖 ~85%
- 效果器链对最终音色的贡献可达 60-70%（尤其 Neuro bass = 简单波形 + 重度 distortion/filter 处理）
- Reese bass 是减法合成不是 FM（两个失谐 saw + 滤波器），DnB 几乎每首都有
- 波表合成实现成本低（PyTorch `grid_sample` + 帧间插值即可微分，DWTS ICASSP 2022 已验证），覆盖面大

**波表可微分方案参考：**
- DWTS（Shan et al. ICASSP 2022）：波表 = 可学习字典，10-20 个波表达到 DDSP 加法合成器水平，推理快 12x
- 核心操作：`output(t) = Σ_k w_k(t) · wavetable_k(phase(t))`，线性插值天然可微

#### 采样库策略（2026-06-15 讨论）

**核心认知：采样不是合成问题，是检索问题。**

模型遇到采样类音频时，不需要从零合成，而是：
1. 识别"这是采样，不是合成的"
2. 从库中检索最接近的匹配（MERT/CLAP embedding + 最近邻）
3. 估计施加在采样上的变换参数（pitch shift、time stretch、效果器）

**采样库规模（渐进建设）：**
- 鼓 one-shots：~2000-3000 个精选（覆盖 95% 场景）
- 经典 break：~20-30 个（Amen/Think/Apache/Funky Drummer 等，几乎全覆盖）
- FX 采样（riser/impact/sweep）：~500 个（大部分可合成替代）
- 人声：不入库，只做检测标注
- 总计 ~5000-10000 个采样，几 GB 级别

**采样训练数据生成：** 库中每个采样随机施加变换（pitch shift / time stretch / 效果器链）→ 生成 (采样ID + 变换参数, 音频) pair。与合成数据混合训练。

**模型架构影响——需要分流：**
```
输入音频 → [分类器] → 合成的 / 采样的
                        ↓           ↓
                   参数估计      embedding → 库检索 + 变换参数估计
                 （核心研究）     （工程问题，相对已解决）
```

#### 起步方案

Phase 0 — 可微分合成引擎（headless DAW）：
- 纯 Python 或 torchsynth 基础上扩展
- 合成最小集：**subtractive + wavetable + FM** + sampler
- 效果器最小集（7-8 种）：**distortion**、LP/HP/BP filter、compressor/OTT、reverb、delay、chorus、EQ
- step sequencer + 多轨混音
- 确定性渲染，批量生成 (参数, 音频) pair

Phase 0.5 — 采样库基建：
- 收集精选采样包（免费/开源起步）
- 构建采样 embedding 索引（MERT/CLAP）
- 采样增强 pipeline（随机变换生成训练数据）

Phase 1 — 音频 tokenizer：
- 先复用 SemantiCodec 或 X-Codec 跑通
- 后续视需要训练自己的语义 tokenizer

Phase 2 — 单合成器逆向（复现 SynthRL）：
- 模型 ~100M，验证 pipeline
- Reward: mel loss + MERT distance

Phase 3 — 扩展到多组件（多音色、鼓组、效果器链、时间结构）+ 采样检索集成

Phase 4 — 引入真实音乐，RLHF 做审美层

#### 训练方法论

**可微分 vs RL：**
- 自研引擎可做可微分 → 梯度直接回传，不需要 RL，收敛更快
- DiffMoog 的 signal-chain loss（每一级做损失）+ Wasserstein loss（解决频率参数不收敛）是关键技术
- RL 留给后期对接真实 VST / 黑盒合成器的场景
- DDSP（Google Magenta）开源，可参考其可微分 DSP 模块设计

**Scaling Law（SODA, ICLR 2026）：**
- 音频 token 信息密度低于文本 → 最优数据量增速是模型大小增速的 1.6 倍 → **偏数据不偏参数**
- 从零训练优于从文本 LLM 热启动（音频和文本表征不重叠）
- 100M-1B 配大量合成数据可能比 4B 配少量数据更有效

**自监督预训练方案参考：**
- AudioMAE：mel spectrogram → patch 化 → mask 75% → Transformer 重建
- BEATs/OpenBEATs：迭代式声学 tokenizer + SSL 模型互蒸馏，完整开源
- MERT：BERT 式，RVQ-VAE + CQT 双 teacher 做伪标签，音乐专用

**训练 Pipeline：**
1. 搓可微分合成引擎（headless DAW）
2. 批量生成合成数据（偏数据策略）
3. AudioMAE/MERT 式自监督预训练 encoder（学音频表示）
4. 可微分梯度训练做参数逆向（signal-chain loss + Wasserstein loss）
5. 课程学习：单 osc → 多 osc → 加 FX → 多轨，渐进增加复杂度
6. Reward/Loss: 低层 mel loss + 高层 MERT embedding 距离
7. 后期 RL（SynthRL 方法）对接黑盒合成器 / 真实 VST
8. 引入真实音乐，RLHF 做审美层

**规模参考：**
- 单合成器逆向：~100M 起步
- 多组件逆向：~1B
- 计算资源：H100 可用，4B 以下可训练
- SynthRL 未开源代码，DiffMoog / DDSP / OpenBEATs 均有开源实现可参考

#### 选型结论

**模型架构：Mamba 为主 + 少量 Attention**
- Audio Mamba 在音频分类上赢 Transformer 3-5%，推理快 1.6x，小数据效率更高
- 混合架构（7 层 Mamba : 1 层 Attention）兼顾长序列效率和全局上下文
- SaShiMi 在音频生成上 MOS 2 倍于 WaveNet，参数量仅 1/3

**训练数据：合成 + 采样，按需生成**
- 合成数据：torchsynth GPU 上 16200x 实时速度，带 72 个参数标注，无限生成
- 采样数据：库中采样 + 随机变换（pitch/time/FX）生成 (采样ID + 变换参数, 音频) pair
- 两类数据混合训练，模型同时学会合成参数估计和采样识别/检索
- 学界惯例 ~100 万样本做单合成器逆向
- 按 SODA scaling law：偏数据不偏参数，100M-1B 配大量数据优于 4B 配少量
- 真实音频留给后期审美阶段

**语言对齐：**
- 最终模型需理解音乐行业术语（supersaw / sidechain / reese bass / complextro 等）
- 在 Phase 4 通过术语标注 + 对话式微调实现

#### 工期估算

计算资源：H100，1 人开发。

| Phase | 内容 | 估时 | 产出 |
|-------|------|------|------|
| 0 | 可微分合成引擎 | 4-6 周 | headless DAW：subtractive + wavetable + FM + sampler + 7-8 种效果器 + sequencer + 多轨混音，全部可微分 |
| 0.5 | 采样库基建 | 1-2 周 | 精选采样收集 + embedding 索引 + 增强 pipeline |
| 1 | 音频 tokenizer 集成 | 1-2 周 | SemantiCodec / X-Codec 接入 |
| 2a | 自监督预训练 | 2-3 周 | AudioMAE 式 encoder，100M 模型，~100 万合成样本 |
| 2b | 单合成器逆向 | 3-4 周 | signal-chain loss，验证参数还原精度 |
| 3 | 课程学习扩展 | 6-8 周 | 多 osc → FX → 多轨，模型扩到 500M-1B |
| 4 | 语言对齐 | 3-4 周 | 音乐术语理解，对话能力 |
| 5 | 真实音频 + 审美 | 开放 | RL 对接黑盒 VST，RLHF 审美，真实音乐 |

**Go/No-go 判断点：Phase 0-2b（~10-15 周 / 2.5-4 个月）**验证可微分引擎 + 单合成器逆向是否 work。
Phase 3-4 再加 9-12 周。起步到初步音频理解约 **半年**。实际大概率 ×1.5-2。

### 当前状态

项目在 Context 仓库孵化中，开工后分离为独立项目。

## 待解决问题

- ~~REAPER MCP Server 选型~~ — 已选 xDarkzx/Reaper-MCP
- ~~VST 音源选型~~ — Serum 为主力
- ~~音色预筛选可行性~~ — 不可行（MCP 读不到 preset）
- ~~渲染试听流程~~ — 不再需要
- ~~方向性问题~~ — 已决定：DAW 辅助路线废弃，转向 agent 音频理解研究
