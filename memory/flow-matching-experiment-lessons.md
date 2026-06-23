---
name: flow-matching-experiment-lessons
description: Flow Matching training experiment results, root causes found, viable directions for GPU training
metadata:
  type: project
---

# Flow Matching 实验总结 (2026-06-23)

## 管线状态

完整的 Flow Matching 训练+推理管线已实现并通过 CPU 验证：

```
src/loom/flow/
├── frontend.py      # CQT / Gammatone / Mel / MultiResCQT 前端
├── conditioner.py   # FreqEncoder(no-pool) → Transformer → MultiQueryPool
├── tokenizer.py     # 97 params ↔ 16 per-group tokens
├── dit.py           # DiT backbone (AdaLN / Cross-Attn / Direct-Cond)
├── flow_matching.py # Rectified flow loss + ODE samplers
├── stage_mask.py    # Per-stage fixed/varying parameter masks
├── train.py         # Simulation-free training loop
└── inference.py     # ODE + gradient refinement
```

**Why:** 回归管线失败——参数不可辨识（多组参数→同一音频），MSE 强迫模型在等价解之间取平均。
**How to apply:** GPU 上直接运行 `python -m loom.flow.train --frontend cqt --dit-blocks 4 --n-layers 4 --pool-size 20000 --epochs 300`。推理用 `scripts/render_demo_audio.py`。

## 核心发现

### 1. 固定参数污染（最关键的 bug）⭐⭐⭐⭐⭐

**问题**：Flow matching 的 N(0,I) 噪声对固定参数（如 stage 1 中 dist_mix=0.0）生成 velocity target ~0.0±1.0。模型输出 ~0.5，导致 synth 渲染完全错误——distortion/reverb 等 FX 被错误激活。

**修法**：
- `stage_mask.py`：定义每个 stage 的固定参数值
- 训练时 mask loss：只对变化参数计算 Huber loss
- 推理时覆写：ODE 后用 `apply_stage_fix()` 将固定参数覆写成正确值

**效果**：Spectral loss 从 2.35 → 0.86（3× 改善），音频从"完全错"变成"大体对"。

**相关文件**：`src/loom/flow/stage_mask.py`

### 2. 频率轴下采样摧毁音高信息 ⭐⭐⭐⭐

**问题**：旧 CNN2DStem 做 4 次 freq-stride=2，128 bin → 8 bin。Bass 频段（20-200Hz）压缩到 0.5 bin，conditioner 无法分辨基频。

**修法**：`FreqEncoder`——Conv1d 沿频率轴，不做任何 stride。配合 CQT (192 bins, 24 bins/oct, fmin=C1)，bass 保留 ~10 bin。

**相关文件**：`src/loom/flow/conditioner.py` 中的 `FreqEncoder`

### 3. 状态保持

遗留关键 bug：Flow Network 不能可靠学习音高（pitch Δ~0.3，预测值集中在 0.5 附近），因为条件机制不够强。见下方"待解决"。

## 架构演进路径（已验证有效 → 待验证）

### ✅ 已验证有效的改进（按效果排序）

| 改进 | 效果 | 文件 |
|------|------|------|
| Stage mask（固定参数覆写）| Spectral loss 3× ↓ | stage_mask.py |
| CQT 替代 Gammatone（192 bin, 24/oct）| Bass 分辨率 ↑ | frontend.py |
| FreqEncoder 不做频率下采样 | 频率信息完整保留 | conditioner.py |
| MultiResCQT（3 时间尺度）| Spectral 从 1.16→0.90 | frontend.py |
| In-distribution noise（shuffle batch 做 x_0）| Flow loss 6× ↓，训练快 3× | dit.py compute_loss |
| Direct conditioning（audio_cond 拼到 token 输入）| 理论上最强 | dit.py DiTBackbone |

### ⚠️ 待解决：Pitch/Filter Cutoff 的 Conditioning 不足

**现象**：模型对所有目标音高输出 ~0.5（训练分布均值）。C2 bass (0.15) 和 C7 bright (0.85) 的预测差仅 0.08，而 ODE 随机噪声就有 0.07。

**根因**：AdaLN 条件机制太弱——它只调制 LayerNorm 的 scale/shift，适合粗粒度条件（"生成猫 vs 狗"），不适合精确参数值。

**未完成的尝试**：Direct conditioning（`data_flow_dc`，训练因 bug 中断）——将 audio_cond 直接拼接到每个 DiT token 输入，让音频信息从一开始就参与 self-attention。理论上能解决，但未完成训练验证。

**其他可能方案**：
- 增大 audio latent tokens（4→16），让 DiT 有更多音频信息
- 在 conditioner 中显式拼接 pitch 检测特征（自相关/CREPE）
- 增加 pitch 参数的 loss 权重（×10）

### ❌ 无效的尝试

| 尝试 | 原因 |
|------|------|
| 纯 AdaLN 条件（原始 DiT）| 条件太弱，模型退化为无条件生成 |
| Cross-Attention DiT（param token attend to audio latents）| 初始化不当，spectral loss 反而退化 |
| LEAF learnable frontend | 学界结论：不比 mel 好多少（EfficientLEAF 2022） |

## GPU 训练时直接使用

默认命令行（推荐）：

```bash
python -m loom.flow.train \
  --frontend cqt --n-bins 192 \
  --dit-blocks 6 --d-model 256 --d-cond 512 \
  --n-layers 4 --n-queries 4 \
  --pool-size 20000 --batch-size 256 \
  --epochs 300 --lr 3e-4 \
  --stage 1 --audio-duration 1.0 \
  --data-dir data_flow_v1
```

关键：必须用 `--stage` 指定正确的课程阶段（0=纯osc, 1=+filter, 2=+env, 3=+mild FX, 99=全开）。

**Why:** 训练的模型必须知道哪些参数是固定的，否则固定参数会被 ODE 推向错误值。
**How to apply:** 始终用 `--stage` 参数，不要用默认值 99（除非在 stage 3+ 的后期训练）。
