import torch
import pytest
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestSubtractiveSynth:
    def setup_method(self):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        )

    def _make_params(self, batch: int = 1) -> dict[str, torch.Tensor]:
        return {
            "osc_pitch": torch.full((batch,), 0.5),
            "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch),
            "osc_detune": torch.full((batch,), 0.5),
            "amp_attack": torch.full((batch,), 0.3),
            "amp_decay": torch.full((batch,), 0.3),
            "amp_sustain": torch.full((batch,), 0.7),
            "amp_release": torch.full((batch,), 0.3),
            "filter_cutoff": torch.full((batch,), 0.5),
            "filter_q": torch.full((batch,), 0.5),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]] * batch),
            "filt_env_attack": torch.full((batch,), 0.3),
            "filt_env_decay": torch.full((batch,), 0.3),
            "filt_env_sustain": torch.full((batch,), 0.5),
            "filt_env_release": torch.full((batch,), 0.3),
            "filt_env_amount": torch.full((batch,), 0.5),
            "dist_amount": torch.full((batch,), 0.3),
            "dist_mix": torch.full((batch,), 0.5),
            "master_gain": torch.full((batch,), 0.8),
            "comp_threshold": torch.full((batch,), 0.5),
            "comp_ratio": torch.full((batch,), 0.3),
            "comp_attack": torch.full((batch,), 0.5),
            "comp_release": torch.full((batch,), 0.5),
            "comp_makeup": torch.full((batch,), 0.0),
            "comp_mix": torch.full((batch,), 0.0),
            "chorus_rate": torch.full((batch,), 0.5),
            "chorus_depth": torch.full((batch,), 0.5),
            "chorus_mix": torch.full((batch,), 0.0),
            "delay_time": torch.full((batch,), 0.5),
            "delay_feedback": torch.full((batch,), 0.3),
            "delay_mix": torch.full((batch,), 0.0),
            "reverb_room_size": torch.full((batch,), 0.5),
            "reverb_decay": torch.full((batch,), 0.5),
            "reverb_damping": torch.full((batch,), 0.3),
            "reverb_mix": torch.full((batch,), 0.0),
            "eq_low_gain": torch.full((batch,), 0.5),
            "eq_mid_gain": torch.full((batch,), 0.5),
            "eq_high_gain": torch.full((batch,), 0.5),
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
        params_b["osc_pitch"] = torch.tensor([0.8])
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
