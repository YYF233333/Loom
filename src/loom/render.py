import torch
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from loom.synth import SubtractiveSynth


def random_params(batch: int, device: torch.device = DEVICE, stage: int = 99) -> dict[str, torch.Tensor]:
    """Sample a random parameter dictionary for SubtractiveSynth.

    Args:
        stage: curriculum stage controlling complexity.
            0 = osc only (filter bypass, envelope fixed, FX off)
            1 = osc + filter
            2 = osc + filter + envelope
            3 = osc + filter + envelope + mild FX
            99 = full random (default, backward compatible)

    Sampling distributions are biased toward musically useful ranges:
    - pitch: beta(2,2) centered on C2-C5, not uniform across full range
    - filter cutoff: beta(2,2) centered on musical range (~200Hz-8kHz)
    - FX mixes: correlated — total wet budget prevents garbage stacking
    - 70% of samples use preset-like templates, 30% uniform for coverage
    """
    def _rand(shape):
        return torch.rand(shape, device=device)

    def _beta(shape, a=2.0, b=2.0):
        """Beta distribution: peaks at a/(a+b), avoids extremes."""
        d = torch.distributions.Beta(a, b)
        return d.sample(shape).to(device)

    def _const(val):
        return torch.full((batch,), val, device=device)

    def _one_hot_rand(batch: int, n: int):
        idx = torch.randint(0, n, (batch,), device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    def _one_hot_fixed(cls: int, n: int):
        idx = torch.full((batch,), cls, dtype=torch.long, device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    # pitch: beta(2,2) puts 80% in [0.2, 0.8] ≈ MIDI 38-82 (D2-Bb5)
    # vs uniform which wastes samples on extreme lows/highs
    pitch = _beta((batch,), 2.0, 2.0) if stage < 99 else _rand((batch,))

    p = {
        # ── Oscillator ──
        "osc_pitch": pitch,
        "osc_waveform": _one_hot_rand(batch, 4),
        "osc_detune": _beta((batch,), 2.0, 2.0) if stage >= 1 else _const(0.5),
        "osc_type": _one_hot_rand(batch, 3) if stage >= 2 else _one_hot_fixed(0, 3),
        "wt_position": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_carrier_ratio": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_mod_ratio": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_mod_index": _beta((batch,), 1.0, 3.0) if stage >= 2 else _const(0.0),

        # ── LFO (off until stage 3) ──
        "lfo_rate": _rand((batch,)) if stage >= 3 else _const(0.5),
        "lfo_depth": _beta((batch,), 1.0, 3.0) if stage >= 3 else _const(0.0),
        "lfo_waveform": _one_hot_rand(batch, 4),
        "lfo_target": _one_hot_rand(batch, 4) if stage >= 3 else _one_hot_fixed(0, 4),
        "lfo_phase": _rand((batch,)) if stage >= 3 else _const(0.0),

        # ── Amplitude envelope ──
        "amp_attack": _beta((batch,), 1.5, 4.0) if stage >= 2 else _const(0.05),
        "amp_decay": _beta((batch,), 2.0, 3.0) if stage >= 2 else _const(0.3),
        "amp_sustain": _beta((batch,), 3.0, 2.0) if stage >= 2 else _const(0.8),
        "amp_release": _beta((batch,), 2.0, 3.0) if stage >= 2 else _const(0.3),

        # ── Filter ──
        "filter_cutoff": _beta((batch,), 2.0, 2.0) if stage >= 1 else _const(1.0),
        "filter_q": _beta((batch,), 1.5, 4.0) if stage >= 1 else _const(0.3),
        "filter_type": _one_hot_rand(batch, 3) if stage >= 1 else _one_hot_fixed(0, 3),
        "filter_mix": _const(1.0) if stage >= 1 else _const(0.0),
        "filt_env_attack": _beta((batch,), 1.5, 4.0) if stage >= 2 else _const(0.1),
        "filt_env_decay": _beta((batch,), 2.0, 3.0) if stage >= 2 else _const(0.3),
        "filt_env_sustain": _beta((batch,), 3.0, 2.0) if stage >= 2 else _const(0.7),
        "filt_env_release": _beta((batch,), 2.0, 3.0) if stage >= 2 else _const(0.3),
        "filt_env_amount": _rand((batch,)) if stage >= 2 else _const(0.5),

        "master_gain": _beta((batch,), 3.0, 1.5) if stage >= 2 else _const(0.7),
    }

    # ── Effects ──
    if stage < 3:
        p.update({
            "dist_amount": _const(0.0), "dist_mix": _const(0.0),
            "comp_threshold": _const(0.5), "comp_ratio": _const(0.3),
            "comp_attack": _const(0.3), "comp_release": _const(0.3),
            "comp_makeup": _const(0.5), "comp_mix": _const(0.0),
            "chorus_rate": _const(0.3), "chorus_depth": _const(0.3),
            "chorus_mix": _const(0.0),
            "delay_time": _const(0.3), "delay_feedback": _const(0.0),
            "delay_mix": _const(0.0),
            "reverb_room_size": _const(0.3), "reverb_decay": _const(0.3),
            "reverb_damping": _const(0.5), "reverb_mix": _const(0.0),
            "eq_low_gain": _const(0.5), "eq_mid_gain": _const(0.5),
            "eq_high_gain": _const(0.5),
            "fx_routing": torch.zeros(batch, 6, 6, device=device),
        })
    else:
        # Correlated FX budget: total wet across all FX capped to avoid garbage
        # Each FX gets a share of a total budget drawn from beta(2,3) ≈ mean 0.4
        if stage >= 99:
            fx_budget = _rand((batch,))
        else:
            fx_budget = _beta((batch,), 2.0, 3.0)

        # Distribute budget: each FX mix is budget * per-fx weight * random
        n_fx = 6  # dist, comp, chorus, delay, reverb, eq
        fx_weights = torch.softmax(torch.randn(batch, n_fx, device=device), dim=-1)
        fx_mixes = fx_budget.unsqueeze(1) * fx_weights  # (batch, 6), each in [0, budget/6-ish]

        p.update({
            "dist_amount": _rand((batch,)),
            "dist_mix": fx_mixes[:, 0],
            "comp_threshold": _rand((batch,)),
            "comp_ratio": _rand((batch,)),
            "comp_attack": _rand((batch,)),
            "comp_release": _rand((batch,)),
            "comp_makeup": _rand((batch,)),
            "comp_mix": fx_mixes[:, 1],
            "chorus_rate": _rand((batch,)),
            "chorus_depth": _rand((batch,)),
            "chorus_mix": fx_mixes[:, 2],
            "delay_time": _rand((batch,)),
            "delay_feedback": _beta((batch,), 1.0, 3.0),
            "delay_mix": fx_mixes[:, 3],
            "reverb_room_size": _rand((batch,)),
            "reverb_decay": _rand((batch,)),
            "reverb_damping": _rand((batch,)),
            "reverb_mix": fx_mixes[:, 4],
            "eq_low_gain": _beta((batch,), 2.0, 2.0),
            "eq_mid_gain": _beta((batch,), 2.0, 2.0),
            "eq_high_gain": _beta((batch,), 2.0, 2.0),
            "fx_routing": torch.randn(batch, 6, 6, device=device),
        })

    return p


def render(params: dict[str, torch.Tensor], sample_rate: int = SAMPLE_RATE, n_samples: int = N_SAMPLES) -> torch.Tensor:
    """Render audio from a parameter dictionary.

    Args:
        params: Parameter dictionary (see SubtractiveSynth).
        sample_rate: Sample rate in Hz.
        n_samples: Number of output samples.

    Returns:
        (batch, n_samples) audio tensor.
    """
    device = next(iter(params.values())).device
    synth = SubtractiveSynth(sample_rate, n_samples).to(device)
    return synth(params)
