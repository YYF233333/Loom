# Encoder v2: Architecture Redesign & Training Pipeline

## Problem

Current encoder (ParamEncoder, 1M params) plateaus at:
- S0 (pitch+waveform): val 0.139 (converges)
- S1 (+filter): val 0.227 (slow, plateaus)
- S2 (+envelope): val 0.454 (barely learns)
- S3 (+mild FX): val 0.699 (barely learns)

Root causes identified by comparison with SynthRL, DiffMoog, AST Sound Matching, DDSP:

1. **1D mel treatment**: Conv1d(128, d, 3) treats frequency bins as channels, loses freq-axis structure
2. **Mean pooling**: destroys temporal information critical for envelope parameters
3. **No attention**: pure SSM, no global context capture
4. **No audio-domain loss**: param MSE is the only loss; spectral loss OOMs at useful batch sizes
5. **Uniform parameter weighting**: osc_pitch gradient ~10000x larger than envelope params

## Approach: Two Architectures, One Training Pipeline

### Architecture A: SynthRL/DETR-style

Directly follows SynthRL (IJCAI 2025) — the current SOTA for synth parameter estimation.

```
Input: mel spectrogram (batch, 128, T)  [T ≈ 173 for 1s @ 44.1kHz]

2D CNN Stem:
  reshape → (batch, 1, 128, T)
  Conv2d(1, 32, 3×3, padding=1) + BN + GELU
  Conv2d(32, 64, 3×3, stride=2, padding=1) + BN + GELU   → (batch, 64, 64, T/2)
  Conv2d(64, 128, 3×3, stride=2, padding=1) + BN + GELU  → (batch, 128, 32, T/4)
  Conv2d(128, d_model, 3×3, stride=2, padding=1) + BN + GELU → (batch, d, 16, T/8)
  flatten spatial → (batch, 16*T/8, d_model)

Positional Encoding:
  Learnable 2D positional embedding (freq_pos + time_pos)

Transformer Encoder:
  6 layers, d_model=256, nhead=8, ff_dim=1024, dropout=0.1

Transformer Decoder:
  N_QUERY learnable query tokens (one per parameter group)
  2 layers cross-attention into encoder output
  Each query attends to the full encoded representation

Parameter Heads (per group):
  osc_head:    Linear(d_model, n_osc_params)     → sigmoid
  filter_head: Linear(d_model, n_filter_params)   → sigmoid
  env_head:    Linear(d_model, n_env_params)       → sigmoid
  fx_head:     Linear(d_model, n_fx_params)        → sigmoid
  cat_heads:   Linear(d_model, n_classes) per cat  → softmax
  routing_head: Linear(d_model, 36)                → raw logits
```

Parameter groups and query allocation:
- 2 queries → oscillator params (pitch, detune, waveform, osc_type, wt, fm)
- 2 queries → filter params (cutoff, Q, type, mix, filter env)
- 2 queries → amplitude envelope (ADSR)
- 2 queries → effects (dist, comp, chorus, delay, reverb, eq)
- 1 query  → global (master_gain, lfo)
- 1 query  → routing (fx_routing logits)
Total: 10 queries

Estimated params: ~5M (d_model=256, 6+2 transformer layers)

### Architecture B: Hybrid Mamba + Attention

Keeps SSM backbone, adds the missing pieces: 2D stem, attention layers, attention pooling.

```
Input: mel spectrogram (batch, 128, T)

2D CNN Stem: (same as Architecture A)
  → (batch, seq_len, d_model)

Positional Encoding: learnable

Backbone: 8 blocks total
  [Mamba, Mamba, Mamba, Mamba, Mamba, Attention, Mamba, Attention]
  Mamba: d_model=256, d_state=64, d_conv=4, expand=2
  Attention: standard multi-head (nhead=8, ff_dim=1024)

Attention Pooling: (same DETR-style decoder as A)
  10 learnable queries, 2-layer cross-attention decoder

Parameter Heads: (same as Architecture A)
```

Estimated params: ~4M

### Shared Training Pipeline

#### Loss Function

```
L_total = L_param + alpha * L_spectral

L_param  = MSE(pred_continuous, target_continuous)    (per-group weighted)
         + 0.5 * CE(pred_categorical, target_categorical)

L_spectral = L_mr_stft(pred_audio, target_audio)      (end-to-end)
           + L_signal_chain(pred_intermediates, target_intermediates)
```

Per-group MSE weighting (based on observed gradient magnitudes):
- osc params (pitch, detune): weight 1.0
- filter params (cutoff, Q): weight 2.0
- envelope params (all ADSR): weight 3.0
- fx params: weight 1.0

alpha is set by GradNorm (existing EMA-based gradient balancing, already implemented).

#### Gradient Accumulation for Spectral Loss

The spectral loss requires running the differentiable synth forward pass (with grad), which OOMs at batch > 64 on V100.

Solution: decouple batch sizes.

```python
# param_loss: full batch (256), fast
pred = model(mel_batch)  # batch=256
loss_p = param_loss(pred, target)

# spectral loss: sub-batches (32), accumulated
loss_s = 0
for i in range(0, 256, 32):
    pred_sub = pred[i:i+32]
    # ... run synth, compute spectral + signal-chain loss
    loss_s += sub_loss / n_accumulations

loss = loss_p + alpha * loss_s
```

#### Optimizer

AdamW with weight decay 0.01. Per-group learning rates:
- stem (CNN): lr * 0.3
- backbone (transformer/mamba): lr * 1.0
- decoder + heads: lr * 1.0

Base lr: 3e-4 (standard for transformers). CosineAnnealingLR with eta_min=3e-5.

#### Curriculum Learning

Same 4-stage system (already implemented), with:
- stage-patience=30 for early advance
- lr reset on stage change
- stage_epochs=200 max per stage

#### Data

- Pool rotation: 20K samples per pool, 5 epochs per pool
- Musical sampling distributions (beta-distributed, correlated FX budget) — already implemented

## Comparison Experiment Protocol

Both architectures trained with identical:
- Training pipeline (same loss, optimizer, curriculum, data distribution)
- Hyperparameters where applicable (d_model=256, same lr schedule)
- Hardware (V100 32GB)
- 800 epochs total with curriculum

Metrics:
- val param_loss per stage (primary)
- val spectral_loss per stage (secondary, when spectral is enabled)
- wall-clock time per epoch
- peak VRAM usage

## Files to Change

| File | Change |
|------|--------|
| `src/loom/training/encoder.py` | Rewrite: two encoder classes (TransformerEncoder, HybridEncoder) + shared decoder/heads |
| `src/loom/training/train.py` | Gradient accumulation for spectral loss; AdamW with param groups; --arch flag |
| `src/loom/training/losses.py` | Per-group param loss weighting |
| `src/loom/training/dataset.py` | No change (param vector format unchanged) |
| `src/loom/synth.py` | No change (return_intermediates already done) |
| `src/loom/render.py` | No change (sampling distributions already done) |

## Out of Scope

- MERT/CLAP embedding distance (Phase 2+ work, needs pretrained model download)
- Wasserstein spectral loss (implement later if frequency params still don't converge)
- Multi-oscillator / parameter symmetry (Phase 3)
- RL training (Phase 5, for black-box VST)
