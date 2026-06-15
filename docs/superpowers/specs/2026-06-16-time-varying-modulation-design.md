# Time-Varying Modulation — LFO 重构设计

## 问题

LFO 调制目前用 `.mean(dim=1)` 把时变信号压成标量，导致 wobble/vibrato 等效果完全丢失。根本原因：所有模块都设计为接受标量参数。

## 解决方案

三个目标各自用正确的方案：

### 1. Filter Cutoff → 新增 SVFilter（chunk-based SVF）

**为什么不是修改 BiquadFilter：** Biquad IIR 的递归结构在系数变化时不稳定。SVF（State Variable Filter）是业界标准的可调制滤波器（Moog/Oberheim/Prophet 全用 SVF），状态变量是物理量（积分器输出），系数变化时天然稳定。

**实现：** Chunk-based SVF。将音频切成 64-sample chunk，每个 chunk 用该时刻的 LFO 值计算滤波系数。chunk 间携带状态（s1, s2）。

- 控制速率：44100 / 64 = 689 Hz，远高于 LFO 最大 20Hz
- 每个 chunk 内用 `torchaudio.functional.lfilter`（固定系数），chunk 间更新系数
- SVF 系数：`g = tan(π * cutoff_hz / sr)`，`R = 1 / (2 * Q)`
- 输出同时提供 LP/HP/BP（SVF 天然多输出），用 filter_type blend 选择

**保留 BiquadFilter：** 用于 EQ 等不需要时变调制的场景。

### 2. Pitch → 振荡器改为 per-sample 频率输入

**当前：** `phases = 2π * f0 * t`（f0 是标量）
**改为：** `phases = cumsum(2π * f(t) / sr)`（f(t) 是 per-sample 时变频率）

- `f(t) = f0 * (1 + lfo_pitch_signal)`，LFO pitch 信号是 `(batch, n_samples)`
- `torch.cumsum` 可微分、GPU 原生
- 相位 wrap：每 ~1000 samples 做 `% (2π)` 防止 float32 漂移
- 三个振荡器（Additive/Wavetable/FM）都需要改

**Additive 特殊处理：** 当前实现预计算 `freqs = f0 * harmonic_n`，然后 `sin(2π * freqs * t)`。改为 per-sample 后，需要对每个谐波做 `cumsum(2π * f(t) * n / sr)`。内存开销增加但计算模式不变。

### 3. Distortion Drive → per-sample tensor

**当前：** `drive.unsqueeze(1)` 标量广播
**改为：** 接受 `(batch, n_samples)` 的时变 drive

几乎零成本：`torch.tanh(signal * drive)` 对标量和 per-sample tensor 行为相同。只需在 synth.py 中构造 per-sample drive 信号而非标量。

### 4. synth.py LFO 集成改动

LFO 的 `lfo_signal (batch, n_samples)` 不再取 mean，直接传给各模块：

```python
# Per-sample modulated cutoff signal -> SVFilter
cutoff_signal = params["filter_cutoff"].unsqueeze(1) + lfo_target[:, 0:1] * lfo_signal * 0.3

# Per-sample modulated frequency -> oscillators  
freq_mod = 1.0 + lfo_target[:, 1:2] * lfo_signal * 0.05  # (batch, n_samples)
# Oscillators use f0 * freq_mod as per-sample frequency

# Per-sample modulated drive -> distortion
drive_signal = params["dist_amount"].unsqueeze(1) + lfo_target[:, 2:3] * lfo_signal * 0.3
```

## 文件改动

```
src/loom/
├── svfilter.py         # 新增 — chunk-based SVF
├── oscillators.py      # 修改 — per-sample frequency input
├── wavetable.py        # 修改 — per-sample frequency input
├── fm.py               # 修改 — per-sample frequency input
├── effects/distortion.py # 修改 — per-sample drive
├── synth.py            # 修改 — LFO 时变调制集成

tests/
├── test_svfilter.py    # 新增
├── test_oscillators.py # 修改 — 测试 per-sample freq
├── test_wavetable.py   # 修改
├── test_fm.py          # 修改
├── test_distortion.py  # 修改
├── test_synth.py       # 更新
```

## 兼容性

- 所有振荡器的 pitch 参数改为可选的 per-sample 频率调制 `freq_mod: Tensor | None`。当 `freq_mod=None` 时退化为原来的标量行为（向后兼容）。
- Distortion 的 amount 参数保持 `(batch,)` 接口不变，在 synth.py 层面构造 per-sample tensor。
- SVFilter 是新模块，不影响 BiquadFilter。
- 现有测试全部应通过（向后兼容的接口）。

## 测试策略

- **SVFilter**：output shape、LP 衰减高频、时变 cutoff 产生 sweep 效果（FFT 频谱随时间变化）、no NaN
- **振荡器 per-sample freq**：freq_mod=None 向后兼容、freq_mod 产生 vibrato（FFT 旁瓣增加）
- **Distortion per-sample drive**：向后兼容、时变 drive 产生幅度调制
- **Synth 集成**：LFO depth>0 + target=[1,0,0,0] 的输出与 depth=0 不同（wobble 测试）
