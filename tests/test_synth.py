import torch
import pytest
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestSubtractiveSynth:
    def setup_method(self):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _make_params(self, batch: int = 1) -> dict[str, torch.Tensor]:
        return {
            "osc_pitch": torch.full((batch,), 0.5, device=DEVICE),
            "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch, device=DEVICE),
            "osc_detune": torch.full((batch,), 0.5, device=DEVICE),
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]] * batch, device=DEVICE),
            "wt_position": torch.full((batch,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((batch,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((batch,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((batch,), 0.0, device=DEVICE),
            "lfo_rate": torch.full((batch,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((batch,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch, device=DEVICE),
            "lfo_target": torch.zeros(batch, 4, device=DEVICE),
            "lfo_phase": torch.full((batch,), 0.0, device=DEVICE),
            "amp_attack": torch.full((batch,), 0.3, device=DEVICE),
            "amp_decay": torch.full((batch,), 0.3, device=DEVICE),
            "amp_sustain": torch.full((batch,), 0.7, device=DEVICE),
            "amp_release": torch.full((batch,), 0.3, device=DEVICE),
            "filter_cutoff": torch.full((batch,), 0.5, device=DEVICE),
            "filter_q": torch.full((batch,), 0.5, device=DEVICE),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]] * batch, device=DEVICE),
            "filt_env_attack": torch.full((batch,), 0.3, device=DEVICE),
            "filt_env_decay": torch.full((batch,), 0.3, device=DEVICE),
            "filt_env_sustain": torch.full((batch,), 0.5, device=DEVICE),
            "filt_env_release": torch.full((batch,), 0.3, device=DEVICE),
            "filt_env_amount": torch.full((batch,), 0.5, device=DEVICE),
            "dist_amount": torch.full((batch,), 0.3, device=DEVICE),
            "dist_mix": torch.full((batch,), 0.5, device=DEVICE),
            "master_gain": torch.full((batch,), 0.8, device=DEVICE),
            "comp_threshold": torch.full((batch,), 0.5, device=DEVICE),
            "comp_ratio": torch.full((batch,), 0.3, device=DEVICE),
            "comp_attack": torch.full((batch,), 0.5, device=DEVICE),
            "comp_release": torch.full((batch,), 0.5, device=DEVICE),
            "comp_makeup": torch.full((batch,), 0.0, device=DEVICE),
            "comp_mix": torch.full((batch,), 0.0, device=DEVICE),
            "chorus_rate": torch.full((batch,), 0.5, device=DEVICE),
            "chorus_depth": torch.full((batch,), 0.5, device=DEVICE),
            "chorus_mix": torch.full((batch,), 0.0, device=DEVICE),
            "delay_time": torch.full((batch,), 0.5, device=DEVICE),
            "delay_feedback": torch.full((batch,), 0.3, device=DEVICE),
            "delay_mix": torch.full((batch,), 0.0, device=DEVICE),
            "reverb_room_size": torch.full((batch,), 0.5, device=DEVICE),
            "reverb_decay": torch.full((batch,), 0.5, device=DEVICE),
            "reverb_damping": torch.full((batch,), 0.3, device=DEVICE),
            "reverb_mix": torch.full((batch,), 0.0, device=DEVICE),
            "eq_low_gain": torch.full((batch,), 0.5, device=DEVICE),
            "eq_mid_gain": torch.full((batch,), 0.5, device=DEVICE),
            "eq_high_gain": torch.full((batch,), 0.5, device=DEVICE),
        }

    def test_output_shape(self):
        params = self._make_params(batch=4)
        audio = self.synth(params)
        assert audio.shape == (4, N_SAMPLES)

    def test_produces_audio(self):
        """Output should not be silence."""
        params = self._make_params()
        audio = self.synth(params)
        assert audio.abs().max().item() > 0.001

    def test_no_nan(self):
        params = self._make_params()
        audio = self.synth(params)
        assert not torch.isnan(audio).any()

    def test_different_params_different_audio(self):
        """Different parameters should produce different audio."""
        params_a = self._make_params()
        params_b = self._make_params()
        params_b["osc_pitch"] = torch.tensor([0.8], device=DEVICE)
        audio_a = self.synth(params_a)
        audio_b = self.synth(params_b)
        assert not torch.allclose(audio_a, audio_b)

    def test_batch_consistency(self):
        """Batched rendering should match individual rendering."""
        params_single = self._make_params(batch=1)
        params_batch = self._make_params(batch=3)
        audio_single = self.synth(params_single)
        audio_batch = self.synth(params_batch)
        assert torch.allclose(audio_single[0], audio_batch[0], atol=1e-5)
