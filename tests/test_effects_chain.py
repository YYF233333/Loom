# tests/test_effects_chain.py
import torch
import pytest
from loom.effects.chain import sinkhorn_normalize, EffectsChain, N_EFFECTS
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fx_params(batch=1, device="cpu"):
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


def _sine(batch=1, n_samples=N_SAMPLES, device="cpu"):
    t = torch.arange(n_samples, dtype=torch.float32, device=device) / SAMPLE_RATE
    wave = 0.5 * torch.sin(2 * 3.14159265 * 440.0 * t)
    return wave.unsqueeze(0).expand(batch, -1).contiguous()


# ---------------------------------------------------------------------------
# TestSinkhornNormalize
# ---------------------------------------------------------------------------

class TestSinkhornNormalize:
    def test_doubly_stochastic(self):
        """Output rows and columns should each sum to 1."""
        torch.manual_seed(0)
        logits = torch.randn(2, N_EFFECTS, N_EFFECTS)
        P = sinkhorn_normalize(logits, tau=1.0, iters=20)
        # Row sums
        assert torch.allclose(P.sum(dim=-1), torch.ones(2, N_EFFECTS), atol=1e-4), \
            f"Row sums: {P.sum(dim=-1)}"
        # Column sums
        assert torch.allclose(P.sum(dim=-2), torch.ones(2, N_EFFECTS), atol=1e-4), \
            f"Col sums: {P.sum(dim=-2)}"

    def test_low_temperature_permutation(self):
        """Very low tau -> one dominant entry per row/col (near-permutation)."""
        torch.manual_seed(1)
        logits = torch.randn(1, N_EFFECTS, N_EFFECTS) * 10.0
        P = sinkhorn_normalize(logits, tau=0.01, iters=30)
        # Each row should have a max close to 1
        row_max = P.max(dim=-1).values
        assert (row_max > 0.9).all(), f"Low-tau row maxima: {row_max}"

    def test_high_temperature_uniform(self):
        """Very high tau -> each entry close to 1/N."""
        logits = torch.zeros(1, N_EFFECTS, N_EFFECTS)
        P = sinkhorn_normalize(logits, tau=100.0, iters=20)
        expected = 1.0 / N_EFFECTS
        assert torch.allclose(P, torch.full_like(P, expected), atol=1e-3), \
            f"High-tau values: {P}"

    def test_gradient_flow(self):
        """Gradients must flow back through sinkhorn_normalize."""
        logits = torch.randn(1, N_EFFECTS, N_EFFECTS, requires_grad=True)
        P = sinkhorn_normalize(logits, tau=1.0, iters=10)
        loss = P.sum()
        loss.backward()
        assert logits.grad is not None, "No gradient at logits"
        assert logits.grad.abs().max() > 0, "Gradient is all zeros"


# ---------------------------------------------------------------------------
# TestEffectsChain
# ---------------------------------------------------------------------------

class TestEffectsChain:
    def setup_method(self):
        self.chain = EffectsChain(SAMPLE_RATE, N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        """Output shape must match input shape."""
        audio = _sine(batch=2, device=str(DEVICE))
        params = _make_fx_params(batch=2, device=str(DEVICE))
        out = self.chain(audio, params)
        assert out.shape == audio.shape, f"Expected {audio.shape}, got {out.shape}"

    def test_bypassed_passthrough(self):
        """All mix=0 (and EQ at 0.5 = 0dB) -> output ≈ input."""
        audio = _sine(batch=1, device=str(DEVICE))
        params = _make_fx_params(batch=1, device=str(DEVICE))
        out = self.chain(audio, params)
        assert torch.allclose(out, audio, atol=1e-5), \
            f"Max diff: {(out - audio).abs().max().item()}"

    def test_no_routing_default_order(self):
        """routing_logits=None uses sequential default order without error."""
        audio = _sine(batch=1, device=str(DEVICE))
        params = _make_fx_params(batch=1, device=str(DEVICE))
        out_default = self.chain(audio, params, routing_logits=None)
        assert out_default.shape == audio.shape

    def test_gradient_through_routing(self):
        """Gradients flow through the Sinkhorn routing path."""
        audio = _sine(batch=1, device=str(DEVICE))
        params = _make_fx_params(batch=1, device=str(DEVICE))
        logits = torch.randn(1, N_EFFECTS, N_EFFECTS, device=str(DEVICE), requires_grad=True)
        out = self.chain(audio, params, routing_logits=logits, tau=1.0)
        loss = out.sum()
        loss.backward()
        assert logits.grad is not None, "No gradient at routing_logits"
        assert logits.grad.abs().max() > 0, "Routing gradient is all zeros"
