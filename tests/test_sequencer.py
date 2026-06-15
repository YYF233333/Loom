import torch
import pytest
from loom.sequencer import render_sequence
from loom.core import SAMPLE_RATE, DEVICE


def _make_synth_params(device=DEVICE):
    """Shared synth params for all notes."""
    return {
        "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
        "osc_detune": torch.full((1,), 0.5, device=device),
        "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=device),
        "wt_position": torch.full((1,), 0.5, device=device),
        "fm_carrier_ratio": torch.full((1,), 0.0, device=device),
        "fm_mod_ratio": torch.full((1,), 0.0, device=device),
        "fm_mod_index": torch.full((1,), 0.0, device=device),
        "lfo_rate": torch.full((1,), 0.5, device=device),
        "lfo_depth": torch.full((1,), 0.0, device=device),
        "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "lfo_target": torch.zeros(1, 4, device=device),
        "lfo_phase": torch.full((1,), 0.0, device=device),
        "amp_attack": torch.full((1,), 0.1, device=device),
        "amp_decay": torch.full((1,), 0.3, device=device),
        "amp_sustain": torch.full((1,), 0.7, device=device),
        "amp_release": torch.full((1,), 0.2, device=device),
        "filter_cutoff": torch.full((1,), 0.6, device=device),
        "filter_q": torch.full((1,), 0.3, device=device),
        "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=device),
        "filt_env_attack": torch.full((1,), 0.2, device=device),
        "filt_env_decay": torch.full((1,), 0.3, device=device),
        "filt_env_sustain": torch.full((1,), 0.5, device=device),
        "filt_env_release": torch.full((1,), 0.3, device=device),
        "filt_env_amount": torch.full((1,), 0.6, device=device),
        "dist_amount": torch.full((1,), 0.0, device=device),
        "dist_mix": torch.full((1,), 0.0, device=device),
        "master_gain": torch.full((1,), 0.8, device=device),
        "comp_threshold": torch.full((1,), 0.5, device=device),
        "comp_ratio": torch.full((1,), 0.3, device=device),
        "comp_attack": torch.full((1,), 0.5, device=device),
        "comp_release": torch.full((1,), 0.5, device=device),
        "comp_makeup": torch.full((1,), 0.0, device=device),
        "comp_mix": torch.full((1,), 0.0, device=device),
        "chorus_rate": torch.full((1,), 0.5, device=device),
        "chorus_depth": torch.full((1,), 0.5, device=device),
        "chorus_mix": torch.full((1,), 0.0, device=device),
        "delay_time": torch.full((1,), 0.5, device=device),
        "delay_feedback": torch.full((1,), 0.3, device=device),
        "delay_mix": torch.full((1,), 0.0, device=device),
        "reverb_room_size": torch.full((1,), 0.5, device=device),
        "reverb_decay": torch.full((1,), 0.5, device=device),
        "reverb_damping": torch.full((1,), 0.3, device=device),
        "reverb_mix": torch.full((1,), 0.0, device=device),
        "eq_low_gain": torch.full((1,), 0.5, device=device),
        "eq_mid_gain": torch.full((1,), 0.5, device=device),
        "eq_high_gain": torch.full((1,), 0.5, device=device),
    }


class TestSequencer:
    def test_output_shape(self):
        synth_params = _make_synth_params()
        bpm = 170.0
        step_sec = 60.0 / bpm / 8.0
        total_samples = int(32 * step_sec * SAMPLE_RATE)

        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.full((1, 32), 0.8, device=DEVICE)
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=bpm, sample_rate=SAMPLE_RATE,
        )
        assert out.shape[0] == 1
        assert abs(out.shape[1] - total_samples) < SAMPLE_RATE

    def test_silence_when_all_velocity_zero(self):
        synth_params = _make_synth_params()
        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.zeros(1, 32, device=DEVICE)
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        assert out.abs().max().item() < 0.01

    def test_single_step_has_audio_at_correct_position(self):
        synth_params = _make_synth_params()
        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.zeros(1, 32, device=DEVICE)
        seq_velocity[0, 16] = 0.8
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        total = out.shape[1]
        first_half_energy = out[0, :total // 2].pow(2).mean()
        second_half_energy = out[0, total // 2:].pow(2).mean()
        assert second_half_energy > first_half_energy * 5

    def test_no_nan(self):
        synth_params = _make_synth_params()
        seq_pitch = torch.rand(1, 32, device=DEVICE)
        seq_velocity = torch.rand(1, 32, device=DEVICE)
        seq_gate = torch.rand(1, 32, device=DEVICE)
        seq_timing = (torch.rand(1, 32, device=DEVICE) - 0.5)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        assert not torch.isnan(out).any()
