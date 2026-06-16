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
            3 = osc + filter + envelope + mild FX (mix biased toward 0)
            99 = full random (default, backward compatible)
    """
    def _rand(shape):
        return torch.rand(shape, device=device)

    def _const(val):
        return torch.full((batch,), val, device=device)

    def _one_hot_rand(batch: int, n: int):
        idx = torch.randint(0, n, (batch,), device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    def _one_hot_fixed(cls: int, n: int):
        idx = torch.full((batch,), cls, dtype=torch.long, device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    p = {
        # ── Oscillator (always random) ──
        "osc_pitch": _rand((batch,)),
        "osc_waveform": _one_hot_rand(batch, 4),
        "osc_detune": _rand((batch,)) if stage >= 1 else _const(0.5),
        "osc_type": _one_hot_rand(batch, 3) if stage >= 2 else _one_hot_fixed(0, 3),
        "wt_position": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_carrier_ratio": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_mod_ratio": _rand((batch,)) if stage >= 2 else _const(0.5),
        "fm_mod_index": _rand((batch,)) if stage >= 2 else _const(0.0),

        # ── LFO (off until stage 3) ──
        "lfo_rate": _rand((batch,)) if stage >= 3 else _const(0.5),
        "lfo_depth": _rand((batch,)) if stage >= 3 else _const(0.0),
        "lfo_waveform": _one_hot_rand(batch, 4),
        "lfo_target": _one_hot_rand(batch, 4) if stage >= 3 else _one_hot_fixed(0, 4),
        "lfo_phase": _rand((batch,)) if stage >= 3 else _const(0.0),

        # ── Amplitude envelope ──
        "amp_attack": _rand((batch,)) if stage >= 2 else _const(0.05),
        "amp_decay": _rand((batch,)) if stage >= 2 else _const(0.3),
        "amp_sustain": _rand((batch,)) if stage >= 2 else _const(0.8),
        "amp_release": _rand((batch,)) if stage >= 2 else _const(0.3),

        # ── Filter ──
        "filter_cutoff": _rand((batch,)) if stage >= 1 else _const(1.0),
        "filter_q": _rand((batch,)) if stage >= 1 else _const(0.3),
        "filter_type": _one_hot_rand(batch, 3) if stage >= 1 else _one_hot_fixed(0, 3),
        "filter_mix": _const(1.0) if stage >= 1 else _const(0.0),
        "filt_env_attack": _rand((batch,)) if stage >= 2 else _const(0.1),
        "filt_env_decay": _rand((batch,)) if stage >= 2 else _const(0.3),
        "filt_env_sustain": _rand((batch,)) if stage >= 2 else _const(0.7),
        "filt_env_release": _rand((batch,)) if stage >= 2 else _const(0.3),
        "filt_env_amount": _rand((batch,)) if stage >= 2 else _const(0.5),

        "master_gain": _rand((batch,)) if stage >= 2 else _const(0.7),
    }

    # ── Effects: off until stage 3, mild at stage 3, full at 99 ──
    fx_mix_fn = _rand if stage >= 99 else (lambda s: _rand(s).pow(3)) if stage >= 3 else (lambda s: _const(0.0))

    p.update({
        "dist_amount": _rand((batch,)) if stage >= 3 else _const(0.0),
        "dist_mix": fx_mix_fn((batch,)),
        "comp_threshold": _rand((batch,)) if stage >= 3 else _const(0.5),
        "comp_ratio": _rand((batch,)) if stage >= 3 else _const(0.3),
        "comp_attack": _rand((batch,)) if stage >= 3 else _const(0.3),
        "comp_release": _rand((batch,)) if stage >= 3 else _const(0.3),
        "comp_makeup": _rand((batch,)) if stage >= 3 else _const(0.5),
        "comp_mix": fx_mix_fn((batch,)),
        "chorus_rate": _rand((batch,)) if stage >= 3 else _const(0.3),
        "chorus_depth": _rand((batch,)) if stage >= 3 else _const(0.3),
        "chorus_mix": fx_mix_fn((batch,)),
        "delay_time": _rand((batch,)) if stage >= 3 else _const(0.3),
        "delay_feedback": _rand((batch,)) if stage >= 3 else _const(0.0),
        "delay_mix": fx_mix_fn((batch,)),
        "reverb_room_size": _rand((batch,)) if stage >= 3 else _const(0.3),
        "reverb_decay": _rand((batch,)) if stage >= 3 else _const(0.3),
        "reverb_damping": _rand((batch,)) if stage >= 3 else _const(0.5),
        "reverb_mix": fx_mix_fn((batch,)),
        "eq_low_gain": _rand((batch,)) if stage >= 3 else _const(0.5),
        "eq_mid_gain": _rand((batch,)) if stage >= 3 else _const(0.5),
        "eq_high_gain": _rand((batch,)) if stage >= 3 else _const(0.5),
        "fx_routing": torch.randn(batch, 6, 6, device=device) if stage >= 3 else torch.zeros(batch, 6, 6, device=device),
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
