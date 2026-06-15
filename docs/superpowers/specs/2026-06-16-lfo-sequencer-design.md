# LFO + Sequencer — Phase 0 批次 3 设计

## 概述

新增 LFO 调制模块（解决 neuro bass 缺少 wobble 的问题）和 32-step sequencer（让引擎能渲染多音符节奏/旋律）。多轨混音留给后续批次。

## LFO 模块

独立 `nn.Module`，生成时变调制信号叠加到目标参数上。

**参数（5 个 + 1 个路由）：**

| 参数 | 范围（归一化前） | 归一化方式 | 说明 |
|------|----------------|-----------|------|
| lfo_rate | 0.1 - 20 Hz | 对数 | LFO 速度 |
| lfo_depth | 0 - 1 | 线性 | 调制深度，0=无调制 |
| lfo_waveform | (batch, 4) | 连续混合 | sine/saw/square/tri |
| lfo_target | (batch, 4) | 连续混合 | 路由权重 [cutoff, pitch, dist_amount, pan(预留)] |
| lfo_phase | 0 - 1 | 线性 | 初始相位偏移 |

**调制方式：**

LFO 输出 `[-1, 1]` 范围的信号 `(batch, n_samples)`，在 SubtractiveSynth.forward 中按路由权重叠加到目标参数：

```
cutoff_mod = filter_cutoff + lfo_target[0] * lfo_out * depth * 0.3
pitch_mod = osc_pitch + lfo_target[1] * lfo_out * depth * 0.05  (±3.6 semitones)
dist_mod = dist_amount + lfo_target[2] * lfo_out * depth * 0.3
pan: 预留，本批次不接线
```

LFO 波形生成复用 AdditiveOscillator 的谐波系数方案（sine/saw/square/tri），但频率在 sub-audio 范围（0.1-20Hz），不需要 anti-aliasing。

## Step Sequencer

纯函数（非 nn.Module），在 synth 之上循环调用 synth 渲染每个音符，然后叠加到 output buffer。

### 网格参数（per-step，32 步）

| 参数 | shape | 范围 | 说明 |
|------|-------|------|------|
| seq_pitch | (batch, 32) | [0, 1] | 每步音高 |
| seq_velocity | (batch, 32) | [0, 1] | 力度，0=rest |
| seq_gate | (batch, 32) | [0, 1] | 音符长度占 step 比例 |
| seq_timing | (batch, 32) | [-0.5, 0.5] | micro-timing 偏移（groove/swing） |

### 全局参数

| 参数 | 范围 | 说明 |
|------|------|------|
| bpm | 100-200 | 速度 |
| n_bars | 固定 1 | 渲染 1 小节（32 步 = 4 拍） |

### 渲染逻辑

1. 计算 step 时长：`step_sec = 60 / bpm / 8`（32 步 = 4 拍 = 1 bar）
2. 总渲染长度：`total_samples = int(32 * step_sec * sample_rate)`
3. 对每个 step：
   - 若 `velocity[step] > 0.05`（活跃）：渲染一个音符片段
   - 音符 `osc_pitch` = `seq_pitch[step]`，其余合成参数共享
   - 音符 `master_gain` *= `seq_velocity[step]`
   - 音符时间位置 = `step * step_sec + seq_timing[step] * step_sec`
   - 音符长度 = `seq_gate[step] * step_sec`（决定 ADSR note_on_duration）
   - 将音符片段叠加到 output buffer 的正确位置
4. 所有活跃 step 的音符批量渲染（扩展 batch 维度并行处理），然后散布到时间轴上

### 关键设计

- 每个音符独立调用 SubtractiveSynth 渲染短片段
- 音符长度 = `gate * step_sec`，synth 的 ADSR `note_on_duration` 设为此值
- velocity 乘到 master_gain 上
- batch 化：所有 batch 的同一 step 并行渲染；同一 batch 的不同 step 也可以并行（需要扩展 batch 维度）

## 文件结构

```
src/loom/
├── lfo.py              # 新增
├── synth.py            # 修改 — LFO 调制
├── sequencer.py        # 新增
└── render.py           # 修改

tests/
├── test_lfo.py         # 新增
├── test_sequencer.py   # 新增
├── test_synth.py       # 修改
└── test_gradients.py   # 修改
```

## SubtractiveSynth 改动

新增 LFO 实例。forward 中计算 LFO 调制信号后，用调制后的参数传给各模块。

LFO 的时变调制与现有的滤波器包络调制叠加：
- 现有：filter_cutoff 被 ADSR 包络的 mean 偏移（静态）
- 新增：filter_cutoff 再被 LFO 信号偏移（时变）

pitch 调制需要特殊处理：LFO→pitch 是时变的，但当前振荡器（Additive/Wavetable/FM）接受的 pitch 是标量 `(batch,)`。解决方案：在 sequencer 层面，每个音符的 pitch 已经是按 step 设定的；LFO→pitch 调制在 synth 内部用 FM-like 方式实现（将 LFO 信号加到 oscillator 的相位增量上）。

## 参数增量

| 来源 | 新参数数 |
|------|---------|
| LFO | ~6（rate, depth, phase + waveform(4) + target(4)，实际 ~14，归一化后） |
| Sequencer | ~128（32 × 4 per-step） + bpm |

Synth 级别参数从 ~48 → ~62（+LFO）。Sequencer 参数是上层参数，不在 synth 内部。

## 测试策略

**LFO（4 tests）：**
- output shape `(batch, n_samples)`
- depth=0 输出全零
- rate 影响频率（FFT 验证）
- no NaN

**Sequencer（4 tests）：**
- 输出 shape 正确
- 全 velocity=0 输出静音
- 单 step 活跃在正确时间位置有音频
- 多步 pattern 能量分布匹配

**梯度测试：** LFO 参数加入 continuous_keys（depth=0 bypass）。Sequencer 不加梯度测试（上层调度器）。
