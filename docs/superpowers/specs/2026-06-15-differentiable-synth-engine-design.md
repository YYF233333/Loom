# Differentiable Synth Engine — Phase 0 最小切片设计

## 概述

从零搭建纯 PyTorch 可微分合成引擎，广泛参考 DiffMoog、DDSP、torchsynth 的成熟设计。最小切片目标：subtractive synth + distortion 效果器 + 单轨渲染，跑通"参数 → 音频 → 梯度回传 → 参数恢复"全链路。

性能语言策略：现阶段纯 Python/PyTorch（GPU tensor 操作已是 CUDA kernel，Python 只是调度层）。模块接口保持干净，未来有 CPU 实时推理需求时可逐模块替换为 Rust/C++。

## 音频参数

- 采样率：44100 Hz
- 渲染长度：4 秒（176400 samples）
- Tensor shape：`(batch, samples)`

## 项目结构

```
loom/
├── pyproject.toml              # uv + pyproject.toml 管理
├── src/
│   └── loom/
│       ├── __init__.py
│       ├── core.py             # 全局常量、SynthModule 基类
│       ├── oscillators.py      # anti-aliased saw/square/sine/tri
│       ├── envelope.py         # ADSR 包络
│       ├── filters.py          # 可微分双二阶滤波器 (LP/HP/BP)
│       ├── amplifier.py        # VCA
│       ├── effects/
│       │   ├── __init__.py
│       │   └── distortion.py   # waveshaper distortion
│       ├── synth.py            # SubtractiveSynth 组装
│       └── render.py           # 渲染入口：参数字典 → 音频 tensor
├── tests/
│   ├── test_oscillators.py
│   ├── test_filters.py
│   ├── test_envelope.py
│   ├── test_effects.py
│   ├── test_synth.py
│   └── test_gradients.py       # gradcheck + 参数恢复回归测试
└── scripts/
    └── param_recovery.py       # 端到端验证脚本（输出收敛曲线）
```

## 参考来源映射

| 模块 | 主要参考 | 具体内容 |
|------|---------|---------|
| oscillators | DDSP (Google Magenta) | harmonic oscillator、anti-aliasing via bandlimiting |
| envelope | torchsynth | ADSR 的可微分实现，避免 hard gate |
| filters | DiffMoog | 可微分 biquad、signal-chain loss 设计 |
| distortion | DiffMoog | waveshaper 函数选型（tanh、soft-clip） |
| 整体架构 | DDSP + DiffMoog | 模块化信号流、每级可算 loss |

不直接使用 DiffMoog 的原因：已废弃（无维护者响应）、PyTorch 1.13.1（落后三个大版本）、缺少 wavetable/sampler/主要效果器（只有 tremolo）、合成引擎与训练 pipeline 深度耦合。

## 信号流

```
参数字典 (pitch, amp, filter_cutoff, filter_Q, adsr, distortion_amount, ...)
    │
    ▼
┌──────────┐   ┌──────┐   ┌────────┐   ┌──────┐   ┌────────────┐   ┌─────┐
│Oscillator│──▶│ ADSR │──▶│ Filter │──▶│ VCA  │──▶│ Distortion │──▶│ Out │
│(saw/sq/  │   │ (amp │   │(biquad │   │      │   │(waveshaper)│   │     │
│ sin/tri) │   │  env)│   │LP/HP/BP│   │      │   │            │   │     │
└──────────┘   └──────┘   └────────┘   └──────┘   └────────────┘   └─────┘
                              ▲
                          ┌──────┐
                          │ ADSR │
                          │(filter│
                          │  env) │
                          └──────┘
```

### 设计决策

1. **参数全部归一化到 [0, 1]**。模块内部负责反归一化到物理单位。频率类用对数映射，增益用 dB 映射。归一化是训练收敛的基本前提（DiffMoog、SynthRL）。

2. **两个 ADSR 包络** — amplitude envelope 控制 VCA，filter envelope 调制 cutoff。经典 Moog/Prophet 架构。

3. **效果器在合成器之后**，最小切片只有 distortion，顺序固定。

4. **batch 化设计**。所有 tensor shape 为 `(batch, samples)`，一次渲染多组参数。

5. **离散参数用连续混合**。waveform（4 种波形加权和）和 filter_type（LP/HP/BP 加权和）训练时可微，推理时取 argmax。DiffMoog 验证过此方案比 Gumbel-softmax 更稳定。

6. **核心基类 `SynthModule(nn.Module)`**，统一接口 `forward(signal_or_none, params) → audio_tensor`，方便 signal-chain loss。

## 参数空间

SubtractiveSynth 完整参数表（~18 个连续参数 + 2 个离散选择）：

| 参数 | 范围（归一化前） | 归一化方式 |
|------|----------------|-----------|
| osc_waveform | {saw, square, sine, tri} | 连续混合（4 权重） |
| osc_pitch | MIDI 24-96 (C1-C7) | 线性 → [0,1] |
| osc_detune | ±100 cents | 线性 → [0,1] |
| amp_attack | 1ms - 2s | 对数 |
| amp_decay | 1ms - 2s | 对数 |
| amp_sustain | 0 - 1 | 线性 |
| amp_release | 1ms - 4s | 对数 |
| filter_type | {LP, HP, BP} | 连续混合（3 权重） |
| filter_cutoff | 20Hz - 20kHz | 对数 |
| filter_Q | 0.5 - 20 | 对数 |
| filt_env_attack | 1ms - 2s | 对数 |
| filt_env_decay | 1ms - 2s | 对数 |
| filt_env_sustain | 0 - 1 | 线性 |
| filt_env_release | 1ms - 4s | 对数 |
| filt_env_amount | -1 - 1 | 线性 |
| dist_amount | 0 - 1 | 线性（0 = bypass） |
| dist_mix | 0 - 1 | 线性（dry/wet） |
| master_gain | -60dB - 0dB | dB → 线性 |

## 测试策略

### 第一层：单元测试（pytest）

每个模块独立验证物理正确性：

- **oscillator**：FFT 峰值频率匹配、幅度范围正确
- **filter**：白噪声通过后频谱衰减符合预期（-12dB/oct for 2-pole biquad）
- **ADSR**：attack/decay/sustain/release 各阶段形状和时长
- **distortion**：soft-clip 行为、谐波引入

### 第二层：可微分性验证

- **gradcheck**：`torch.autograd.gradcheck` 验证每个模块的梯度数值正确
- **参数恢复测试**（端到端 go/no-go 判据）：
  1. 随机采样参数 → 渲染"目标"音频
  2. 随机初始化参数 → 渲染"预测"音频
  3. mel-spectrogram L1 loss + 梯度下降
  4. 断言：N 步后参数误差 < 阈值

`scripts/param_recovery.py` 是参数恢复测试的可视化版本，输出收敛曲线和参数对比。

### Go/No-go 判据

参数恢复测试通过 = Phase 0 最小切片完成。如果单合成器自己的参数都恢复不了，后续一切无意义。

## 最小切片之后的扩展路径

1. +compressor 效果器
2. +wavetable oscillator（`grid_sample` + 帧间插值，参考 DWTS ICASSP 2022）
3. +更多效果器（reverb、delay、chorus、EQ）
4. +FM oscillator
5. +sampler（embedding 检索 + 变换参数估计）
6. +sequencer + 多轨混音
