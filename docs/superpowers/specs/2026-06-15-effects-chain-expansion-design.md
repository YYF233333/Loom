# Effects Chain Expansion — Phase 0 扩展设计

## 概述

为现有 SubtractiveSynth 补全完整效果器链：在已有的 Distortion 基础上新增 Compressor、Chorus、Delay、Reverb (FDN)、EQ 共 5 个可微分效果器模块。总参数从 ~18 扩展到 ~37。

这是批次 1（效果器补全），后续批次 2（wavetable/FM/sampler）和批次 3（sequencer/多轨）各自独立 spec。

## 信号流

```
Oscillator → Filter (w/ env) → VCA (w/ env) → 效果器链 → Out

效果器链顺序（固定）：
Distortion → Compressor → Chorus → Delay → Reverb → EQ → Master Gain
```

顺序依据 EDM 制作惯例：distortion 塑形原始波形，compressor 控制动态，调制类在时域效果之前，EQ 最后做频率修正。

## 文件结构

```
src/loom/effects/
├── __init__.py
├── distortion.py      # 已有
├── compressor.py      # 新增
├── chorus.py          # 新增
├── delay.py           # 新增
├── reverb.py          # 新增
└── eq.py              # 新增

tests/
├── test_compressor.py # 新增
├── test_chorus.py     # 新增
├── test_delay.py      # 新增
├── test_reverb.py     # 新增
├── test_eq.py         # 新增
├── test_gradients.py  # 更新 — 扩展梯度检查
└── ...                # 已有测试不变
```

## 新增参数表

| 模块 | 参数 | 范围（归一化前） | 归一化方式 |
|------|------|----------------|-----------|
| Compressor | comp_threshold | -40dB - 0dB | 线性 |
| | comp_ratio | 1:1 - 20:1 | 对数 |
| | comp_attack | 0.1ms - 100ms | 对数 |
| | comp_release | 10ms - 1000ms | 对数 |
| | comp_makeup | 0dB - 30dB | 线性 |
| | comp_mix | 0 - 1 | 线性 (dry/wet) |
| Chorus | chorus_rate | 0.1Hz - 5Hz | 对数 |
| | chorus_depth | 0 - 1 | 线性 |
| | chorus_mix | 0 - 1 | 线性 |
| Delay | delay_time | 10ms - 500ms | 对数 |
| | delay_feedback | 0 - 0.9 | 线性 |
| | delay_mix | 0 - 1 | 线性 |
| Reverb (FDN) | reverb_room_size | 0 - 1 | 线性（缩放延迟线长度） |
| | reverb_decay | 0 - 1 | 线性（反馈增益） |
| | reverb_damping | 0 - 1 | 线性（高频衰减） |
| | reverb_mix | 0 - 1 | 线性 |
| EQ | eq_low_gain | -12dB - +12dB | 线性 |
| | eq_mid_gain | -12dB - +12dB | 线性 |
| | eq_high_gain | -12dB - +12dB | 线性 |

共 19 个新参数，总计 ~37 个参数。

## 各模块技术方案

### Compressor

可微分 feed-forward compressor：

1. RMS 能量检测：滑动窗口均值（窗口 ~1024 samples），用 `torch.nn.functional.avg_pool1d` 实现
2. Gain reduction：`gain = (rms / threshold) ^ (1/ratio - 1)` 当 `rms > threshold`，否则 1.0。用 `torch.where` + `torch.pow` 保持可微
3. Attack/release 平滑：一阶 IIR 平滑 gain 变化（`coeff = exp(-1 / (time * sample_rate))`），用 `torch.clamp` 确保数值稳定
4. Makeup gain 补偿 + dry/wet 混合

参考：DiffMoog compressor 的 gain reduction 曲线设计。

### Chorus

调制延迟线：

1. LFO 生成：`lfo = depth * sin(2π * rate * t)`，输出延迟调制量 [1ms, 20ms]
2. 分数延迟插值：将 1D 信号 reshape 为 `(batch, 1, 1, n_samples)`，用 `torch.nn.functional.grid_sample` 做亚采样精度的延迟读取
3. Wet = 延迟调制后的信号，output = `(1 - mix) * dry + mix * wet`

### Delay

可微分延迟线 + feedback：

1. 延迟时间 [10ms, 500ms]，对数映射
2. Feedback 上限 0.9 防止发散
3. 展开 feedback 循环 8 次：`output = input; for i in range(8): delayed = fractional_delay(output, delay_samples); output = output + feedback^i * delayed`
4. 分数延迟：同 Chorus 的 `grid_sample` 方案
5. dry/wet 混合

### Reverb (FDN)

Feedback Delay Network，4 条延迟线：

1. 延迟线长度（互质）：base lengths [1433, 1601, 1867, 2053] samples，乘以 `room_size` 缩放
2. 反馈矩阵：4×4 Householder 矩阵 `H = I - 2vv^T / (v^T v)`，v = [1,1,1,1]，正交且能量守恒
3. 每条延迟线末端加 one-pole lowpass：`y[n] = (1 - damping) * x[n] + damping * y[n-1]`，实现高频衰减
4. Decay 控制反馈增益：feedback_gain = `decay * 0.95`（上限防止发散）
5. 展开 N 次迭代（N = ceil(duration * sr / min_delay_length)，约 80-120 次）
6. dry/wet 混合

### EQ

3 段参数化 EQ，复用 BiquadFilter 的系数计算逻辑：

1. Low shelf @ 200Hz：boost/cut 低频
2. Mid peak @ 1kHz：boost/cut 中频（peaking EQ，Q 固定 = 1.0）
3. High shelf @ 5kHz：boost/cut 高频
4. 三个 biquad 串联，每段用 `torchaudio.functional.lfilter`
5. Shelf/peak 系数公式来自 Audio EQ Cookbook（同 BiquadFilter 的参考源）

## SubtractiveSynth 集成

`synth.py` 的 `forward` 方法扩展效果器链：

```python
audio = self.distortion(audio, params["dist_amount"], params["dist_mix"])
audio = self.compressor(audio, params["comp_threshold"], params["comp_ratio"],
                        params["comp_attack"], params["comp_release"],
                        params["comp_makeup"], params["comp_mix"])
audio = self.chorus(audio, params["chorus_rate"], params["chorus_depth"],
                    params["chorus_mix"])
audio = self.delay(audio, params["delay_time"], params["delay_feedback"],
                   params["delay_mix"])
audio = self.reverb(audio, params["reverb_room_size"], params["reverb_decay"],
                    params["reverb_damping"], params["reverb_mix"])
audio = self.eq(audio, params["eq_low_gain"], params["eq_mid_gain"],
                params["eq_high_gain"])
```

每个效果器 `mix=0` 时完全 bypass（梯度仍流通）。

## 测试策略

每个效果器 4 个测试：

| 测试 | 验证内容 |
|------|---------|
| output shape | `(batch, n_samples)` |
| bypass (mix=0) | 输出等于输入 |
| 物理特性 | 见下 |
| no NaN | 极端参数无 NaN |

物理特性验证：
- **Compressor**：大信号被压缩（输出 RMS < 输入 RMS when threshold < input RMS）
- **Chorus**：频谱旁瓣能量增加（调制引入频率扩展）
- **Delay**：自相关在延迟时间处有峰
- **Reverb**：输出尾部有能量衰减（最后 1/4 段非零但小于前 1/4 段）
- **EQ**：boost low_gain 后低频能量增加

梯度测试更新：`test_synth_has_gradients` 扩展覆盖所有新参数。参数恢复测试自动覆盖（`random_params` 包含新参数）。

## Go/No-go 判据

同 Phase 0 最小切片：参数恢复 loss 收敛比 < 0.5。如果新效果器导致收敛失败，逐个 bypass 排查。
