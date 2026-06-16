# Sinkhorn Effects Chain Routing

## Problem

The current effects chain has a hardcoded order: Distortion -> Compressor -> Chorus -> Delay -> Reverb -> EQ. In real production, effect ordering is a creative choice (e.g., distortion before vs after filter produces fundamentally different timbres). A fixed chain limits the engine's ability to reverse-engineer audio with non-standard routing.

## Design

Replace the fixed effects chain with N=6 slots. Each slot can host any of the 6 effects. A 6x6 routing matrix, normalized via Sinkhorn iteration, determines which effect runs in which slot. Each effect appears exactly once (permutation constraint). Bypass is handled orthogonally by each effect's existing `mix` parameter.

### Architecture

```
VCA output
    |
    v
 [Slot 0] -- Sinkhorn row 0 weights --> weighted sum of all 6 effects
    |
    v
 [Slot 1] -- Sinkhorn row 1 weights --> weighted sum of all 6 effects
    |
    ...
    v
 [Slot 5] -- Sinkhorn row 5 weights --> weighted sum of all 6 effects
    |
    v
 Output
```

### Sinkhorn Normalization

Input: 6x6 logit matrix L (from params), temperature tau.

```python
S = L / tau
for _ in range(sinkhorn_iters):
    S = S - logsumexp(S, dim=-1, keepdim=True)   # row normalize (log domain)
    S = S - logsumexp(S, dim=-2, keepdim=True)   # col normalize (log domain)
P = softmax(S, dim=-1)  # final row-stochastic, approximately doubly-stochastic
```

Properties:
- Each row sums to ~1 (each slot selects one effect)
- Each column sums to ~1 (each effect used exactly once)
- tau high -> uniform exploration; tau low -> sharp permutation
- Fully differentiable

### Slot Execution

Each slot receives the output of the previous slot and runs ALL 6 effects on it, then blends by routing weights:

```python
for slot_idx in range(6):
    outputs = [effect_fn(audio, params) for effect_fn in effects]
    audio = sum(P[batch, slot_idx, fx_idx] * outputs[fx_idx] for fx_idx in range(6))
```

Training: 6 slots x 6 effects = 36 effect evaluations per forward pass.
Inference: argmax each row -> 6 evaluations (same as current).

### Temperature Annealing

During training, tau follows a schedule:
- Warmup phase: tau = tau_max (e.g., 5.0) - soft blending, explore orderings
- Anneal phase: tau decays exponentially toward tau_min (e.g., 0.1)
- tau is NOT a learned parameter; it follows a fixed schedule per epoch

### Effect Wrappers

Each effect has a different signature. A uniform wrapper normalizes them:

```python
class EffectSlot:
    effects = [distortion, compressor, chorus, delay, reverb, eq]

    def run_effect(self, idx, audio, params):
        # Dispatch to the correct effect with correct params
        ...
        return processed_audio
```

Special handling for Distortion: the LFO-modulated per-sample drive is pre-computed before the slot chain and passed via params, same as current.

Special handling for EQ: no `mix` param. Bypass state is all gains = 0.5 (0dB). The routing matrix handles ordering; the EQ parameters handle bypass.

### Parameter Interface

New parameter: `fx_routing` - (batch, 36) flattened logit matrix, reshaped to (batch, 6, 6) internally.

In `dataset.py`:
- Add 36 entries to CONTINUOUS_KEYS for the routing logits
- Random init: sample from N(0, 1) to start with roughly uniform routing
- `vector_to_params` / `params_to_vector` updated accordingly

In `render.py`:
- `random_params()` generates random routing logits

### Files Changed

| File | Change |
|------|--------|
| `src/loom/effects/chain.py` | **NEW** - EffectsChain module with Sinkhorn routing |
| `src/loom/synth.py` | Replace 6 inline effect calls with single EffectsChain call |
| `src/loom/render.py` | Add `fx_routing` to random_params |
| `src/loom/training/dataset.py` | Add routing logits to CONTINUOUS_KEYS |
| `tests/test_effects_chain.py` | **NEW** - Sinkhorn convergence, permutation property, gradient flow |
| `tests/test_gradients.py` | Add fx_routing to gradient checks |
| `tests/test_preset_waveforms.py` | Add fx_routing to _transparent_params (zeros = uniform, bypassed via mix=0) |
| `tests/test_preset_synth.py` | Add fx_routing to _base_params |

### Backward Compatibility

- `fx_routing` is optional in params dict via `params.get("fx_routing")`
- When absent, falls back to the current fixed order (identity permutation equivalent)
- All existing tests pass without modification (they don't supply fx_routing)

### Risks

1. **36 extra effect evals during training**: ~6x compute for effects portion. Effects are lightweight vs oscillator+filter, so total overhead estimated at ~30-50%.
2. **Gradient noise during soft blending**: mitigated by temperature annealing.
3. **Dataset regeneration required**: old datasets don't have routing params. Training script already supports `--regenerate`.
