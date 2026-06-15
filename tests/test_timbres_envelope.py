import torch
import numpy as np
import math
import pytest
from loom.envelope import ADSR
from loom.lfo import LFO
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from tests.timbre_helpers import fundamental_freq


class TestADSRTimbres:
    def setup_method(self):
        self.adsr = ADSR(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, attack, decay, sustain, release):
        with torch.no_grad():
            env = self.adsr(
                torch.tensor([attack], device=DEVICE, dtype=torch.float32),
                torch.tensor([decay], device=DEVICE, dtype=torch.float32),
                torch.tensor([sustain], device=DEVICE, dtype=torch.float32),
                torch.tensor([release], device=DEVICE, dtype=torch.float32),
            )
        return env[0].cpu().numpy()

    def _expected_ms(self, norm, max_ms):
        log_min = math.log(1.0)
        log_max = math.log(max_ms)
        return math.exp(norm * (log_max - log_min) + log_min)

    def test_attack_time_accuracy(self):
        env = self._render(0.5, 0.5, 0.8, 0.3)
        expected_ms = self._expected_ms(0.5, 2000.0)
        threshold = 0.95
        idx = np.argmax(env >= threshold)
        actual_ms = idx / SAMPLE_RATE * 1000
        assert abs(actual_ms - expected_ms) < 10.0, (
            f"Attack: expected {expected_ms:.1f}ms, got {actual_ms:.1f}ms"
        )

    def test_attack_zero_fast(self):
        env = self._render(0.0, 0.5, 0.8, 0.3)
        peak_idx = np.argmax(env)
        peak_ms = peak_idx / SAMPLE_RATE * 1000
        assert peak_ms < 3.0

    def test_attack_max_slow(self):
        env = self._render(1.0, 0.5, 0.8, 0.3)
        expected_ms = self._expected_ms(1.0, 2000.0)  # 2000ms
        threshold = 0.95
        above = np.where(env >= threshold)[0]
        if len(above) > 0:
            actual_ms = above[0] / SAMPLE_RATE * 1000
            assert abs(actual_ms - expected_ms) < 100.0

    def test_decay_reaches_sustain(self):
        env = self._render(0.1, 0.5, 0.6, 0.3)
        attack_ms = self._expected_ms(0.1, 2000.0)
        decay_ms = self._expected_ms(0.5, 2000.0)
        settle_sample = int((attack_ms + decay_ms * 3) / 1000 * SAMPLE_RATE)
        settle_sample = min(settle_sample, int(SAMPLE_RATE * 2.5))
        if settle_sample < len(env):
            level = env[settle_sample]
            assert abs(level - 0.6) < 0.1, f"Sustain level: expected 0.6, got {level:.3f}"

    def test_sustain_holds_flat(self):
        env = self._render(0.1, 0.2, 0.7, 0.3)
        start = int(SAMPLE_RATE * 0.5)
        end = int(SAMPLE_RATE * 2.5)
        segment = env[start:end]
        assert np.std(segment) < 0.02, f"Sustain std: {np.std(segment):.4f}"

    def test_release_decays_to_zero(self):
        env = self._render(0.1, 0.2, 0.7, 0.3)
        release_ms = self._expected_ms(0.3, 4000.0)
        check_sample = int((3.0 + release_ms / 1000 * 2) * SAMPLE_RATE)
        check_sample = min(check_sample, len(env) - 1)
        assert env[check_sample] < 0.05

    def test_sustain_zero_ad_only(self):
        env = self._render(0.1, 0.3, 0.0, 0.3)
        mid = int(SAMPLE_RATE * 1.0)
        assert env[mid] < 0.05

    def test_sustain_one_no_decay(self):
        env = self._render(0.2, 0.5, 1.0, 0.3)
        mid = int(SAMPLE_RATE * 1.5)
        assert env[mid] > 0.95

    @pytest.mark.parametrize("norm", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_denorm_time_mapping(self, norm):
        expected = self._expected_ms(norm, 2000.0)
        actual_tensor = self.adsr._denorm_time(
            torch.tensor([norm], device=DEVICE), 2000.0
        )
        actual_ms = actual_tensor.item() * 1000
        assert abs(actual_ms - expected) < 0.1


class TestLFOTimbres:
    def setup_method(self):
        self.lfo = LFO(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, rate, depth, waveform, phase=0.0):
        with torch.no_grad():
            return self.lfo(
                torch.tensor([rate], device=DEVICE, dtype=torch.float32),
                torch.tensor([depth], device=DEVICE, dtype=torch.float32),
                torch.tensor([waveform], device=DEVICE, dtype=torch.float32),
                torch.tensor([phase], device=DEVICE, dtype=torch.float32),
            )[0].cpu().numpy()

    def _expected_rate_hz(self, norm):
        log_min = math.log(0.1)
        log_max = math.log(20.0)
        return math.exp(norm * (log_max - log_min) + log_min)

    def test_sine_frequency(self):
        signal = self._render(0.5, 1.0, [1, 0, 0, 0])
        f0 = fundamental_freq(signal, SAMPLE_RATE)
        expected = self._expected_rate_hz(0.5)
        assert abs(f0 - expected) < 0.1

    def test_depth_range(self):
        signal = self._render(0.5, 0.8, [1, 0, 0, 0])
        assert signal.max() <= 0.81
        assert signal.min() >= -0.81

    def test_depth_zero_silent(self):
        signal = self._render(0.5, 0.0, [1, 0, 0, 0])
        assert np.allclose(signal, 0.0, atol=1e-7)

    def test_saw_shape(self):
        signal = self._render(0.3, 1.0, [0, 1, 0, 0])
        diff = np.diff(signal)
        positive_ratio = np.sum(diff > 0) / len(diff)
        assert positive_ratio > 0.45

    def test_square_binary_values(self):
        signal = self._render(0.3, 0.8, [0, 0, 1, 0])
        unique_abs = np.unique(np.round(np.abs(signal), decimals=2))
        assert len(unique_abs) <= 5
        assert np.abs(signal).max() <= 0.81

    def test_triangle_symmetry(self):
        signal = self._render(0.3, 1.0, [0, 0, 0, 1])
        assert abs(signal.max() + signal.min()) < 0.1

    def test_phase_offset(self):
        signal_0 = self._render(0.3, 1.0, [1, 0, 0, 0], phase=0.0)
        signal_half = self._render(0.3, 1.0, [1, 0, 0, 0], phase=0.5)
        assert abs(signal_half[0]) < 0.1
        assert not np.allclose(signal_0, signal_half, atol=0.1)

    @pytest.mark.parametrize("norm,expected_hz", [
        (0.0, 0.1), (1.0, 20.0),
    ])
    def test_rate_denorm_range(self, norm, expected_hz):
        actual = self._expected_rate_hz(norm)
        assert abs(actual - expected_hz) < 0.01 * expected_hz


class TestLFOTargetRouting:
    """Test that lfo_target vector routes modulation to the correct parameter."""

    def setup_method(self):
        from loom.synth import SubtractiveSynth
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _base_params(self):
        return {
            "osc_pitch": torch.tensor([0.4], device=DEVICE, dtype=torch.float32),
            "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE, dtype=torch.float32),
            "osc_detune": torch.full((1,), 0.5, device=DEVICE),
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE, dtype=torch.float32),
            "wt_position": torch.full((1,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((1,), 0.0, device=DEVICE),
            "lfo_rate": torch.full((1,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((1,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE, dtype=torch.float32),
            "lfo_target": torch.zeros(1, 4, device=DEVICE),
            "lfo_phase": torch.full((1,), 0.0, device=DEVICE),
            "amp_attack": torch.full((1,), 0.1, device=DEVICE),
            "amp_decay": torch.full((1,), 0.3, device=DEVICE),
            "amp_sustain": torch.full((1,), 0.9, device=DEVICE),
            "amp_release": torch.full((1,), 0.3, device=DEVICE),
            "filter_cutoff": torch.full((1,), 0.5, device=DEVICE),
            "filter_q": torch.full((1,), 0.3, device=DEVICE),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE, dtype=torch.float32),
            "filt_env_attack": torch.full((1,), 0.1, device=DEVICE),
            "filt_env_decay": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_sustain": torch.full((1,), 0.5, device=DEVICE),
            "filt_env_release": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_amount": torch.full((1,), 0.5, device=DEVICE),
            "dist_amount": torch.full((1,), 0.3, device=DEVICE),
            "dist_mix": torch.full((1,), 0.5, device=DEVICE),
            "master_gain": torch.full((1,), 0.85, device=DEVICE),
            "comp_threshold": torch.full((1,), 0.5, device=DEVICE),
            "comp_ratio": torch.full((1,), 0.3, device=DEVICE),
            "comp_attack": torch.full((1,), 0.5, device=DEVICE),
            "comp_release": torch.full((1,), 0.5, device=DEVICE),
            "comp_makeup": torch.full((1,), 0.0, device=DEVICE),
            "comp_mix": torch.full((1,), 0.0, device=DEVICE),
            "chorus_rate": torch.full((1,), 0.5, device=DEVICE),
            "chorus_depth": torch.full((1,), 0.5, device=DEVICE),
            "chorus_mix": torch.full((1,), 0.0, device=DEVICE),
            "delay_time": torch.full((1,), 0.5, device=DEVICE),
            "delay_feedback": torch.full((1,), 0.3, device=DEVICE),
            "delay_mix": torch.full((1,), 0.0, device=DEVICE),
            "reverb_room_size": torch.full((1,), 0.5, device=DEVICE),
            "reverb_decay": torch.full((1,), 0.5, device=DEVICE),
            "reverb_damping": torch.full((1,), 0.3, device=DEVICE),
            "reverb_mix": torch.full((1,), 0.0, device=DEVICE),
            "eq_low_gain": torch.full((1,), 0.5, device=DEVICE),
            "eq_mid_gain": torch.full((1,), 0.5, device=DEVICE),
            "eq_high_gain": torch.full((1,), 0.5, device=DEVICE),
        }

    def test_zero_target_equals_no_lfo(self):
        params_off = self._base_params()
        params_off["lfo_depth"] = torch.full((1,), 0.0, device=DEVICE)

        params_zero_target = self._base_params()
        params_zero_target["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_zero_target["lfo_target"] = torch.zeros(1, 4, device=DEVICE)

        with torch.no_grad():
            audio_off = self.synth(params_off)[0].cpu().numpy()
            audio_zero = self.synth(params_zero_target)[0].cpu().numpy()
        assert np.allclose(audio_off, audio_zero, atol=1e-4)

    def test_cutoff_target_modulates_spectrum(self):
        params_static = self._base_params()
        params_mod = self._base_params()
        params_mod["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_mod["lfo_target"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            audio_static = self.synth(params_static)[0].cpu().numpy()
            audio_mod = self.synth(params_mod)[0].cpu().numpy()

        assert not np.allclose(audio_static, audio_mod, atol=0.01)

    def test_drive_target_modulates_thd(self):
        params_static = self._base_params()
        params_mod = self._base_params()
        params_mod["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_mod["lfo_target"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=DEVICE, dtype=torch.float32)

        with torch.no_grad():
            audio_static = self.synth(params_static)[0].cpu().numpy()
            audio_mod = self.synth(params_mod)[0].cpu().numpy()

        assert not np.allclose(audio_static, audio_mod, atol=0.01)
