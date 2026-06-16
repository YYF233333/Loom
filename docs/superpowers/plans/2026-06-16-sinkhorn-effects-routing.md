# Sinkhorn Effects Chain Routing — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the fixed effects chain with learnable Sinkhorn-routed slots so the model can discover optimal effect ordering during training.

**Architecture:** 6 slots, each runs all 6 effects weighted by a doubly-stochastic matrix from Sinkhorn normalization. Temperature annealing sharpens to a hard permutation at convergence. Backward compatible — missing `fx_routing` param falls back to the current fixed order.

**Tech Stack:** PyTorch, existing effect modules (Distortion, Compressor, Chorus, Delay, Reverb, EQ)

---

### Task 1: Sinkhorn normalization function

**Files:**
- Create: `src/loom/effects/chain.py`
- Test: `tests/test_effects_chain.py`

- [ ] **Step 1: Write failing test for Sinkhorn properties**

```python
# tests/test_effects_chain.py
import torch
import pytest


class TestSinkhornNormalize:
    def test_output_is_doubly_stochastic(self):
        from loom.effects.chain import sinkhorn_normalize
        logits = torch.randn(2, 6, 6)
        P = sinkhorn_normalize(logits, tau=1.0, iters=20)
        # Rows sum to ~1
        row_sums = P.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3)
        # Columns sum to ~1
        col_sums = P.sum(dim=-2)
        assert torch.allclose(col_sums, torch.ones_like(col_sums), atol=1e-3)

    def test_low_temperature_approaches_permutation(self):
        from loom.effects.chain import sinkhorn_normalize
        logits = torch.randn(1, 6, 6)
        P = sinkhorn_normalize(logits, tau=0.01, iters=50)
        # Each row should have one dominant entry
        maxvals = P.max(dim=-1).values
        assert (maxvals > 0.95).all()

    def test_high_temperature_is_uniform(self):
        from loom.effects.chain import sinkhorn_normalize
        logits = torch.zeros(1, 6, 6)
        P = sinkhorn_normalize(logits, tau=10.0, iters=20)
        expected = torch.full_like(P, 1.0 / 6)
        assert torch.allclose(P, expected, atol=0.05)

    def test_gradients_flow(self):
        from loom.effects.chain import sinkhorn_normalize
        logits = torch.randn(1, 6, 6, requires_grad=True)
        P = sinkhorn_normalize(logits, tau=1.0, iters=10)
        P.sum().backward()
        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_effects_chain.py::TestSinkhornNormalize -v`
Expected: ImportError — `loom.effects.chain` does not exist

- [ ] **Step 3: Implement sinkhorn_normalize**

```python
# src/loom/effects/chain.py
import torch
import torch.nn as nn


def sinkhorn_normalize(
    logits: torch.Tensor, tau: float = 1.0, iters: int = 10,
) -> torch.Tensor:
    """Sinkhorn normalization: logits (batch, N, N) -> doubly-stochastic matrix.

    Args:
        logits: Raw routing logits.
        tau: Temperature. High = uniform, low = sharp permutation.
        iters: Number of alternating row/col normalization steps.
    """
    s = logits / max(tau, 1e-6)
    for _ in range(iters):
        s = s - torch.logsumexp(s, dim=-1, keepdim=True)
        s = s - torch.logsumexp(s, dim=-2, keepdim=True)
    return torch.softmax(s, dim=-1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_effects_chain.py::TestSinkhornNormalize -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```
git add src/loom/effects/chain.py tests/test_effects_chain.py
git commit -m "feat: add sinkhorn_normalize function"
```

---

### Task 2: EffectsChain module

**Files:**
- Modify: `src/loom/effects/chain.py`
- Test: `tests/test_effects_chain.py`

- [ ] **Step 1: Write failing test for EffectsChain forward pass**

```python
# append to tests/test_effects_chain.py
from loom.effects.chain import EffectsChain
from loom.core import SAMPLE_RATE


def _make_fx_params(batch=1, device="cpu"):
    """Minimal effect params with all effects bypassed (mix=0)."""
    return {
        "dist_drive": torch.zeros(batch, device=device),
        "dist_mix": torch.zeros(batch, device=device),
        "comp_threshold": torch.full((batch,), 0.5, device=device),
        "comp_ratio": torch.full((batch,), 0.3, device=device),
        "comp_attack": torch.full((batch,), 0.5, device=device),
        "comp_release": torch.full((batch,), 0.5, device=device),
        "comp_makeup": torch.zeros(batch, device=device),
        "comp_mix": torch.zeros(batch, device=device),
        "chorus_rate": torch.full((batch,), 0.5, device=device),
        "chorus_depth": torch.full((batch,), 0.5, device=device),
        "chorus_mix": torch.zeros(batch, device=device),
        "delay_time": torch.full((batch,), 0.5, device=device),
        "delay_feedback": torch.full((batch,), 0.3, device=device),
        "delay_mix": torch.zeros(batch, device=device),
        "reverb_room_size": torch.full((batch,), 0.5, device=device),
        "reverb_decay": torch.full((batch,), 0.5, device=device),
        "reverb_damping": torch.full((batch,), 0.3, device=device),
        "reverb_mix": torch.zeros(batch, device=device),
        "eq_low_gain": torch.full((batch,), 0.5, device=device),
        "eq_mid_gain": torch.full((batch,), 0.5, device=device),
        "eq_high_gain": torch.full((batch,), 0.5, device=device),
    }


class TestEffectsChain:
    def setup_method(self):
        self.n_samples = 4410
        self.chain = EffectsChain(SAMPLE_RATE, self.n_samples)

    def test_output_shape(self):
        audio = torch.randn(2, self.n_samples)
        params = _make_fx_params(batch=2)
        routing = torch.zeros(2, 6, 6)
        out = self.chain(audio, params, routing_logits=routing, tau=1.0)
        assert out.shape == audio.shape

    def test_bypassed_is_passthrough(self):
        """All effects at mix=0 -> output ≈ input regardless of routing."""
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        routing = torch.randn(1, 6, 6)
        out = self.chain(audio, params, routing_logits=routing, tau=1.0)
        assert torch.allclose(out, audio, atol=1e-4)

    def test_no_routing_uses_default_order(self):
        """Missing routing_logits -> identity order (backward compat)."""
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        out = self.chain(audio, params)
        assert out.shape == audio.shape

    def test_gradient_through_routing(self):
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        params["dist_mix"] = torch.tensor([0.5])
        routing = torch.randn(1, 6, 6, requires_grad=True)
        out = self.chain(audio, params, routing_logits=routing, tau=1.0)
        out.pow(2).mean().backward()
        assert routing.grad is not None
        assert not torch.isnan(routing.grad).any()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_effects_chain.py::TestEffectsChain -v`
Expected: ImportError or AttributeError — EffectsChain not defined

- [ ] **Step 3: Implement EffectsChain**

```python
# append to src/loom/effects/chain.py
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ

N_EFFECTS = 6
# Canonical order indices: dist=0, comp=1, chorus=2, delay=3, reverb=4, eq=5
FX_DIST, FX_COMP, FX_CHORUS, FX_DELAY, FX_REVERB, FX_EQ = range(N_EFFECTS)


class EffectsChain(nn.Module):
    """Sinkhorn-routed effects chain with learnable ordering.

    6 slots, each runs all 6 effects weighted by a doubly-stochastic
    routing matrix. Falls back to fixed canonical order when no
    routing_logits are provided.
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.distortion = Distortion()
        self.compressor = Compressor()
        self.chorus = Chorus(sample_rate, n_samples)
        self.delay = Delay(sample_rate, n_samples)
        self.reverb = Reverb(sample_rate, n_samples)
        self.eq = EQ(sample_rate)

    def _run_effect(self, idx: int, audio: torch.Tensor,
                    params: dict) -> torch.Tensor:
        if idx == FX_DIST:
            return self.distortion(audio, params["dist_drive"], params["dist_mix"])
        elif idx == FX_COMP:
            return self.compressor(
                audio, params["comp_threshold"], params["comp_ratio"],
                params["comp_attack"], params["comp_release"],
                params["comp_makeup"], params["comp_mix"],
            )
        elif idx == FX_CHORUS:
            return self.chorus(
                audio, params["chorus_rate"], params["chorus_depth"],
                params["chorus_mix"],
            )
        elif idx == FX_DELAY:
            return self.delay(
                audio, params["delay_time"], params["delay_feedback"],
                params["delay_mix"],
            )
        elif idx == FX_REVERB:
            return self.reverb(
                audio, params["reverb_room_size"], params["reverb_decay"],
                params["reverb_damping"], params["reverb_mix"],
            )
        elif idx == FX_EQ:
            return self.eq(
                audio, params["eq_low_gain"], params["eq_mid_gain"],
                params["eq_high_gain"],
            )

    def forward(
        self, audio: torch.Tensor, params: dict,
        routing_logits: torch.Tensor | None = None,
        tau: float = 1.0,
    ) -> torch.Tensor:
        """Apply effects chain.

        Args:
            audio: (batch, n_samples)
            params: Dict of effect parameters.
            routing_logits: (batch, 6, 6) or None. None = fixed canonical order.
            tau: Sinkhorn temperature.
        """
        if routing_logits is None:
            # Fixed canonical order: dist -> comp -> chorus -> delay -> reverb -> eq
            for fx_idx in range(N_EFFECTS):
                audio = self._run_effect(fx_idx, audio, params)
            return audio

        P = sinkhorn_normalize(routing_logits, tau=tau)

        for slot_idx in range(N_EFFECTS):
            weights = P[:, slot_idx, :]  # (batch, N_EFFECTS)
            outputs = []
            for fx_idx in range(N_EFFECTS):
                outputs.append(self._run_effect(fx_idx, audio, params))
            stacked = torch.stack(outputs, dim=1)  # (batch, N_EFFECTS, n_samples)
            audio = (weights.unsqueeze(-1) * stacked).sum(dim=1)

        return audio
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_effects_chain.py -v`
Expected: 8 passed (4 Sinkhorn + 4 EffectsChain)

- [ ] **Step 5: Commit**

```
git add src/loom/effects/chain.py tests/test_effects_chain.py
git commit -m "feat: add EffectsChain module with Sinkhorn routing"
```

---

### Task 3: Integrate EffectsChain into SubtractiveSynth

**Files:**
- Modify: `src/loom/synth.py`

- [ ] **Step 1: Run existing tests as baseline**

Run: `pytest tests/ -q`
Expected: 231 passed

- [ ] **Step 2: Refactor synth.py to use EffectsChain**

Replace the 6 individual effect calls in `SubtractiveSynth.__init__` and `forward` with an `EffectsChain` instance.

In `__init__`, replace:
```python
        self.distortion = Distortion()
        self.compressor = Compressor()
        self.chorus = Chorus(sample_rate, n_samples)
        self.delay = Delay(sample_rate, n_samples)
        self.reverb = Reverb(sample_rate, n_samples)
        self.eq = EQ(sample_rate)
```
with:
```python
        from loom.effects.chain import EffectsChain
        self.effects_chain = EffectsChain(sample_rate, n_samples)
```

In `forward`, replace everything from the distortion call through the EQ call (lines 103-120) with:
```python
        # Effects chain (Sinkhorn-routed when fx_routing provided)
        dist_lfo = lfo_target[:, 2:3] * lfo_signal * 0.3
        dist_drive = (params["dist_amount"].unsqueeze(1) + dist_lfo).clamp(0.0, 1.0)

        fx_params = {
            "dist_drive": dist_drive,
            "dist_mix": params["dist_mix"],
            "comp_threshold": params["comp_threshold"],
            "comp_ratio": params["comp_ratio"],
            "comp_attack": params["comp_attack"],
            "comp_release": params["comp_release"],
            "comp_makeup": params["comp_makeup"],
            "comp_mix": params["comp_mix"],
            "chorus_rate": params["chorus_rate"],
            "chorus_depth": params["chorus_depth"],
            "chorus_mix": params["chorus_mix"],
            "delay_time": params["delay_time"],
            "delay_feedback": params["delay_feedback"],
            "delay_mix": params["delay_mix"],
            "reverb_room_size": params["reverb_room_size"],
            "reverb_decay": params["reverb_decay"],
            "reverb_damping": params["reverb_damping"],
            "reverb_mix": params["reverb_mix"],
            "eq_low_gain": params["eq_low_gain"],
            "eq_mid_gain": params["eq_mid_gain"],
            "eq_high_gain": params["eq_high_gain"],
        }

        routing = params.get("fx_routing")
        tau = params.get("fx_routing_tau", 1.0)
        if isinstance(tau, torch.Tensor):
            tau = tau.item()
        audio = self.effects_chain(audio, fx_params, routing_logits=routing, tau=tau)
```

Also remove the now-unused direct imports at the top of synth.py:
```python
# Remove these lines:
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ
```

- [ ] **Step 3: Run existing tests to verify backward compatibility**

Run: `pytest tests/ -q`
Expected: 231 passed — all existing tests pass without supplying fx_routing

- [ ] **Step 4: Commit**

```
git add src/loom/synth.py
git commit -m "refactor: replace inline effects with EffectsChain in synth"
```

---

### Task 4: Update parameter generation and training dataset

**Files:**
- Modify: `src/loom/render.py`
- Modify: `src/loom/training/dataset.py`

- [ ] **Step 1: Add fx_routing to render.py random_params**

After the `"filter_mix"` line, add:
```python
        "fx_routing": torch.randn(batch, 6, 6, device=device),
```

- [ ] **Step 2: Add routing logits to dataset.py**

In `CONTINUOUS_KEYS`, after `"lfo_phase"`, add:
```python
    # fx_routing logits (6x6 = 36 values)
    *[f"fx_routing_{i}_{j}" for i in range(6) for j in range(6)],
```

Update `params_to_vector` — after the categorical keys loop, add:
```python
    if "fx_routing" in params:
        batch = parts[0].shape[0]
        parts.append(params["fx_routing"].reshape(batch, 36))
```

Update `vector_to_params` — after the categorical keys loop, add:
```python
    if idx < vector.shape[1]:
        params["fx_routing"] = vector[:, idx:idx + 36].reshape(-1, 6, 6)
        idx += 36
```

- [ ] **Step 3: Run existing tests**

Run: `pytest tests/ -q`
Expected: all pass

- [ ] **Step 4: Commit**

```
git add src/loom/render.py src/loom/training/dataset.py
git commit -m "feat: add fx_routing to parameter generation and dataset"
```

---

### Task 5: Update gradient and preset tests

**Files:**
- Modify: `tests/test_gradients.py`
- Modify: `tests/test_preset_waveforms.py`
- Modify: `tests/test_preset_synth.py`

- [ ] **Step 1: Add fx_routing to gradient tests**

In `test_gradients.py`, in `test_synth_has_gradients`:
- Add `"fx_routing"` to `blend_keys` list (since it's a matrix, not a scalar)

In `test_parameter_recovery_converges`:
- Add to `target_params`:
```python
            "fx_routing": torch.zeros(1, 6, 6, device=DEVICE),
```
- Add `"fx_routing"` to `bypass_keys` set

- [ ] **Step 2: Add fx_routing to preset test base params**

In `tests/test_preset_waveforms.py` `_transparent_params()`, add after `"filter_mix"`:
```python
        "fx_routing": torch.zeros(1, 6, 6),
```

In `tests/test_preset_synth.py` `_base_params()`, add after `"filter_mix"`:
```python
        "fx_routing": torch.zeros(1, 6, 6),
```

Note: zeros logits + Sinkhorn = uniform routing. With all effects at mix=0, any routing produces the same output, so existing assertions hold.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: all pass (231 + 8 new effects_chain tests)

- [ ] **Step 4: Commit**

```
git add tests/test_gradients.py tests/test_preset_waveforms.py tests/test_preset_synth.py
git commit -m "test: add fx_routing to gradient and preset tests"
```

---

### Task 6: Add routing-specific integration tests

**Files:**
- Modify: `tests/test_effects_chain.py`

- [ ] **Step 1: Write tests for routing affecting output**

```python
# append to tests/test_effects_chain.py
class TestRoutingBehavior:
    def setup_method(self):
        self.n_samples = 44100
        self.chain = EffectsChain(SAMPLE_RATE, self.n_samples)

    def test_different_routing_different_output(self):
        """Two different hard routings with active effects produce different audio."""
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        params["dist_mix"] = torch.tensor([0.8])
        params["dist_drive"] = torch.full((1,), 0.5)
        params["reverb_mix"] = torch.tensor([0.5])

        # Route A: distortion first (slot 0), reverb last (slot 5)
        logits_a = torch.eye(6).unsqueeze(0) * 100  # identity = canonical order
        # Route B: reverb first (slot 0), distortion in slot 4
        logits_b = torch.zeros(1, 6, 6)
        perm_b = [4, 1, 2, 3, 0, 5]  # swap dist(0) and reverb(4)
        for slot, fx in enumerate(perm_b):
            logits_b[0, slot, fx] = 100.0

        out_a = self.chain(audio.clone(), params, routing_logits=logits_a, tau=0.01)
        out_b = self.chain(audio.clone(), params, routing_logits=logits_b, tau=0.01)

        diff = (out_a - out_b).abs().max().item()
        assert diff > 0.01, f"Routing should change output, max diff={diff}"

    def test_identity_routing_matches_canonical(self):
        """Identity permutation routing should match no-routing (canonical) output."""
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        params["dist_mix"] = torch.tensor([0.3])
        params["dist_drive"] = torch.full((1,), 0.3)

        out_none = self.chain(audio.clone(), params, routing_logits=None)
        logits_id = torch.eye(6).unsqueeze(0) * 100
        out_id = self.chain(audio.clone(), params, routing_logits=logits_id, tau=0.01)

        assert torch.allclose(out_none, out_id, atol=1e-3), (
            f"Identity routing should match canonical. Max diff: "
            f"{(out_none - out_id).abs().max().item()}"
        )
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_effects_chain.py::TestRoutingBehavior -v`
Expected: 2 passed

- [ ] **Step 3: Write test for audio output saving**

```python
# append to TestRoutingBehavior
    def test_save_routed_audio(self):
        """Save audio with non-canonical routing for manual inspection."""
        import os
        from tests.conftest import save_test_wav
        audio = torch.randn(1, self.n_samples)
        params = _make_fx_params(batch=1)
        params["dist_mix"] = torch.tensor([0.7])
        params["dist_drive"] = torch.full((1,), 0.6)
        params["reverb_mix"] = torch.tensor([0.4])

        # Reverb before distortion
        logits = torch.zeros(1, 6, 6)
        perm = [4, 1, 2, 3, 0, 5]  # reverb->comp->chorus->delay->dist->eq
        for slot, fx in enumerate(perm):
            logits[0, slot, fx] = 100.0

        out = self.chain(audio, params, routing_logits=logits, tau=0.01)
        path = save_test_wav(out[0].detach(), "fx_routing_reverb_before_dist")
        assert os.path.exists(path)
```

- [ ] **Step 4: Run full test suite**

Run: `pytest tests/ -q`
Expected: all pass

- [ ] **Step 5: Commit**

```
git add tests/test_effects_chain.py
git commit -m "test: add routing behavior integration tests"
```

---

### Task 7: Add temperature annealing helper for training

**Files:**
- Modify: `src/loom/effects/chain.py`
- Modify: `src/loom/training/train.py`

- [ ] **Step 1: Write test for annealing schedule**

```python
# append to tests/test_effects_chain.py
class TestAnnealSchedule:
    def test_linear_anneal(self):
        from loom.effects.chain import routing_temperature
        assert routing_temperature(0, 100) == pytest.approx(5.0, abs=0.1)
        assert routing_temperature(99, 100) == pytest.approx(0.1, abs=0.05)

    def test_midpoint(self):
        from loom.effects.chain import routing_temperature
        mid = routing_temperature(50, 100)
        assert 0.1 < mid < 5.0
```

- [ ] **Step 2: Implement routing_temperature**

```python
# append to src/loom/effects/chain.py
import math

def routing_temperature(
    epoch: int, total_epochs: int,
    tau_max: float = 5.0, tau_min: float = 0.1,
) -> float:
    """Exponential temperature decay for Sinkhorn routing."""
    progress = min(epoch / max(total_epochs - 1, 1), 1.0)
    return tau_max * math.exp(progress * math.log(tau_min / tau_max))
```

- [ ] **Step 3: Integrate into train.py**

In the training loop, before `pred_audio = synth(...)`, compute tau and inject:
```python
            from loom.effects.chain import routing_temperature
            tau = routing_temperature(epoch, args.epochs)
            # Inject tau into params for the synth forward pass
            clamped["fx_routing_tau"] = torch.tensor(tau)
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_effects_chain.py::TestAnnealSchedule -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```
git add src/loom/effects/chain.py src/loom/training/train.py tests/test_effects_chain.py
git commit -m "feat: add temperature annealing for routing and integrate in training"
```
