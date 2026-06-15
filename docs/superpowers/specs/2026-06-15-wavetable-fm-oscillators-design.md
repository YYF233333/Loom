# Wavetable + FM Oscillators — Phase 0 批次 2 设计

## 概述

为引擎新增两种振荡器：WavetableOscillator（波表合成，覆盖 ~20% EDM 音色）和 FMOscillator（FM 合成，覆盖 ~10%）。作为独立 nn.Module 与现有 AdditiveOscillator 平级，SubtractiveSynth 通过 `osc_type` 连续权重混合三者。

## 文件结构

```
src/loom/
├── wavetable.py        # 新增 WavetableOscillator
├── fm.py               # 新增 FMOscillator
├── synth.py            # 修改 — 三振荡器混合
└── render.py           # 修改 — 新参数

tests/
├── test_wavetable.py   # 新增
├── test_fm.py          # 新增
├── test_synth.py       # 修改
└── test_gradients.py   # 修改
```

## WavetableOscillator

波表 = 一组预定义单周期波形（帧），振荡器在帧之间插值实现音色 morph。

**算法：**
1. 相位累积：`phase[t] = cumsum(f0 / sr) % 1.0`（归一化相位 [0, 1)）
2. 帧内读取：`grid_sample` 从当前波表帧中按相位位置插值采样
3. 帧间 morph：`wt_position` [0, 1] 线性混合相邻帧

**内置波表：** 16 帧，从 saw 渐变到 square，数学生成：
- frame 0: pure saw（谐波系数 1/n）
- frame 15: pure square（奇次谐波 1/n）
- 中间帧：线性插值谐波系数

**参数（3 个连续）：**

| 参数 | 范围（归一化前） | 归一化方式 |
|------|----------------|-----------|
| wt_pitch | MIDI 24-96 | 线性 → [0,1] |
| wt_detune | ±100 cents | 线性，0.5 = 无偏移 |
| wt_position | 0 - 1 | 线性（帧位置） |

参考：DWTS (ICASSP 2022) 的 `grid_sample` 波表读取方案。

## FMOscillator

单 carrier + 单 modulator FM 合成，固定频率比方案（参考 DDX7）。

**算法：**
```
mod_phase = cumsum(2π * f0 * mod_ratio / sr)
carrier_phase = cumsum(2π * f0 * carrier_ratio / sr + mod_index * sin(mod_phase))
output = sin(carrier_phase)
```

相位用 `cumsum` + `% (2π)` 周期 wrap 避免 float32 漂移。

**参数（5 个连续）：**

| 参数 | 范围（归一化前） | 归一化方式 |
|------|----------------|-----------|
| fm_pitch | MIDI 24-96 | 线性 → [0,1] |
| fm_detune | ±100 cents | 线性，0.5 = 无偏移 |
| fm_carrier_ratio | 1 - 8 | 线性 |
| fm_mod_ratio | 1 - 8 | 线性 |
| fm_mod_index | 0 - 20 | 线性 |

**关键决策：**
- carrier_ratio 和 mod_ratio 用连续值（不量化为整数），训练时可微
- 用固定比率而非绝对频率，回避 FM 频率不收敛问题（DDX7、DiffMoog 经验）

## SubtractiveSynth 集成

新增 `osc_type` 参数 `(batch, 3)` — 三种振荡器的连续混合权重 `[additive, wavetable, fm]`：

```python
audio = (osc_type[:, 0:1] * additive_out
       + osc_type[:, 1:2] * wavetable_out
       + osc_type[:, 2:3] * fm_out)
```

训练时连续可微，推理时取 argmax。与 waveform blend 和 filter_type blend 设计一致。

各振荡器共享 pitch/detune 参数空间（MIDI 24-96, ±100 cents），但 WavetableOscillator 有 wt_position，FMOscillator 有 carrier_ratio/mod_ratio/mod_index。当某个振荡器的 osc_type 权重为 0 时，其特有参数的梯度仍流通（与效果器 mix=0 bypass 同理）。

## 参数汇总

新增 ~11 个参数（wt: 3, fm: 5, osc_type: 3-blend），总参数从 ~37 → ~48。

## 测试策略

**WavetableOscillator（4 tests）：**
- output shape (batch, n_samples)
- 频率正确性：FFT 峰值匹配目标频率
- position 影响音色：不同 wt_position 产生不同波形
- no NaN（极端参数）

**FMOscillator（4 tests）：**
- output shape (batch, n_samples)
- carrier_ratio=1, mod_index=0 退化为纯正弦
- mod_index 增大引入更多谐波
- no NaN（极端参数）

**梯度测试更新：** 新参数加入 continuous_keys 和 optimize_keys。参数恢复测试中新振荡器保持 bypass（osc_type = [1,0,0]），与效果器 bypass 策略一致。
