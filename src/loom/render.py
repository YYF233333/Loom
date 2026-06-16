import torch
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from loom.synth import SubtractiveSynth


def random_params(batch: int, device: torch.device = DEVICE) -> dict[str, torch.Tensor]:
    """Sample a random parameter dictionary for SubtractiveSynth."""
    def _rand(shape):
        return torch.rand(shape, device=device)

    def _one_hot_rand(batch: int, n: int):
        idx = torch.randint(0, n, (batch,), device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    return {
        "osc_pitch": _rand((batch,)),
        "osc_waveform": _one_hot_rand(batch, 4),
        "osc_detune": _rand((batch,)),
        "osc_type": _one_hot_rand(batch, 3),
        "wt_position": _rand((batch,)),
        "fm_carrier_ratio": _rand((batch,)),
        "fm_mod_ratio": _rand((batch,)),
        "fm_mod_index": _rand((batch,)),
        "lfo_rate": _rand((batch,)),
        "lfo_depth": _rand((batch,)),
        "lfo_waveform": _one_hot_rand(batch, 4),
        "lfo_target": torch.zeros(batch, 4, device=device),
        "lfo_phase": _rand((batch,)),
        "amp_attack": _rand((batch,)),
        "amp_decay": _rand((batch,)),
        "amp_sustain": _rand((batch,)),
        "amp_release": _rand((batch,)),
        "filter_cutoff": _rand((batch,)),
        "filter_q": _rand((batch,)),
        "filter_type": _one_hot_rand(batch, 3),
        "filt_env_attack": _rand((batch,)),
        "filt_env_decay": _rand((batch,)),
        "filt_env_sustain": _rand((batch,)),
        "filt_env_release": _rand((batch,)),
        "filt_env_amount": _rand((batch,)),
        "filter_mix": _rand((batch,)),
        "dist_amount": _rand((batch,)),
        "dist_mix": _rand((batch,)),
        "master_gain": _rand((batch,)),
        "comp_threshold": _rand((batch,)),
        "comp_ratio": _rand((batch,)),
        "comp_attack": _rand((batch,)),
        "comp_release": _rand((batch,)),
        "comp_makeup": _rand((batch,)),
        "comp_mix": _rand((batch,)),
        "chorus_rate": _rand((batch,)),
        "chorus_depth": _rand((batch,)),
        "chorus_mix": _rand((batch,)),
        "delay_time": _rand((batch,)),
        "delay_feedback": _rand((batch,)),
        "delay_mix": _rand((batch,)),
        "reverb_room_size": _rand((batch,)),
        "reverb_decay": _rand((batch,)),
        "reverb_damping": _rand((batch,)),
        "reverb_mix": _rand((batch,)),
        "eq_low_gain": _rand((batch,)),
        "eq_mid_gain": _rand((batch,)),
        "eq_high_gain": _rand((batch,)),
        "fx_routing": torch.randn(batch, 6, 6, device=device),
    }


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
