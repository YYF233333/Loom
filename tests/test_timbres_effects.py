# tests/test_timbres_effects.py
import torch
import numpy as np
import pytest
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from tests.timbre_helpers import (
    thd, spectral_centroid, rms_envelope, fundamental_freq,
)


def _sine_tensor(freq=440.0, n_samples=N_SAMPLES, amplitude=0.5):
    t = torch.arange(n_samples, dtype=torch.float32) / SAMPLE_RATE
    return (amplitude * torch.sin(2 * 3.14159265 * freq * t)).unsqueeze(0).to(DEVICE)


class TestDistortionTimbres:
    def setup_method(self):
        self.dist = Distortion()

    def test_mix_zero_bypass(self):
        signal = _sine_tensor()
        out = self.dist(signal, torch.tensor([0.5], device=DEVICE), torch.tensor([0.0], device=DEVICE))
        assert torch.allclose(out, signal, atol=1e-6)

    def test_drive_zero_mix_zero_passthrough(self):
        signal = _sine_tensor()
        out = self.dist(signal, torch.tensor([0.0], device=DEVICE), torch.tensor([0.0], device=DEVICE))
        assert torch.allclose(out, signal, atol=1e-6)

    def test_soft_clip_adds_harmonics(self):
        signal = _sine_tensor()
        out = self.dist(signal, torch.tensor([0.5], device=DEVICE), torch.tensor([1.0], device=DEVICE))
        audio = out[0].detach().cpu().numpy()
        assert thd(audio, SAMPLE_RATE, 440.0) > 0.05

    def test_heavy_clip_near_square(self):
        signal = _sine_tensor(amplitude=0.8)
        out = self.dist(signal, torch.tensor([0.9], device=DEVICE), torch.tensor([1.0], device=DEVICE))
        audio = out[0].detach().cpu().numpy()
        assert np.abs(audio).max() < 1.05
        assert thd(audio, SAMPLE_RATE, 440.0) > 0.3

    def test_drive_monotonic_thd(self):
        signal = _sine_tensor()
        thd_values = []
        for drive in [0.2, 0.5, 0.8]:
            out = self.dist(signal, torch.tensor([drive], device=DEVICE), torch.tensor([1.0], device=DEVICE))
            thd_values.append(thd(out[0].detach().cpu().numpy(), SAMPLE_RATE, 440.0))
        assert thd_values[0] < thd_values[1] < thd_values[2]


class TestCompressorTimbres:
    def setup_method(self):
        self.comp = Compressor()

    def _make_dynamic_signal(self):
        t = torch.arange(N_SAMPLES, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        env = 0.3 + 0.7 * torch.abs(torch.sin(2 * 3.14159 * 2.0 * t))
        return (env * torch.sin(2 * 3.14159 * 440.0 * t)).unsqueeze(0)

    def test_mix_zero_bypass(self):
        signal = _sine_tensor()
        out = self.comp(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_reduces_dynamic_range(self):
        signal = self._make_dynamic_signal()
        out = self.comp(
            signal,
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.6], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
            torch.tensor([1.0], device=DEVICE),
        )
        env_in = rms_envelope(signal[0].cpu().numpy(), hop=2048)
        env_out = rms_envelope(out[0].detach().cpu().numpy(), hop=2048)
        dr_in = env_in.max() / (env_in.min() + 1e-8)
        dr_out = env_out.max() / (env_out.min() + 1e-8)
        assert dr_out < dr_in

    def test_higher_ratio_more_compression(self):
        signal = self._make_dynamic_signal()
        drs = []
        for ratio in [0.2, 0.8]:
            out = self.comp(
                signal,
                torch.tensor([0.3], device=DEVICE),
                torch.tensor([ratio], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([0.0], device=DEVICE),
                torch.tensor([1.0], device=DEVICE),
            )
            env = rms_envelope(out[0].detach().cpu().numpy(), hop=2048)
            drs.append(env.max() / (env.min() + 1e-8))
        assert drs[1] < drs[0]

    def test_makeup_gain_increases_rms(self):
        signal = _sine_tensor(amplitude=0.3)
        out = self.comp(
            signal,
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([1.0], device=DEVICE),
        )
        rms_in = np.sqrt(np.mean(signal[0].cpu().numpy() ** 2))
        rms_out = np.sqrt(np.mean(out[0].detach().cpu().numpy() ** 2))
        assert rms_out > rms_in

    def test_no_spectral_coloring(self):
        signal = _sine_tensor()
        out = self.comp(
            signal,
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
            torch.tensor([1.0], device=DEVICE),
        )
        sc_in = spectral_centroid(signal[0].cpu().numpy(), SAMPLE_RATE)
        sc_out = spectral_centroid(out[0].detach().cpu().numpy(), SAMPLE_RATE)
        assert abs(sc_out - sc_in) / sc_in < 0.10


class TestChorusTimbres:
    def setup_method(self):
        self.chorus = Chorus(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_mix_zero_bypass(self):
        signal = _sine_tensor()
        out = self.chorus(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-5)

    def test_preserves_fundamental(self):
        signal = _sine_tensor()
        out = self.chorus(
            signal,
            torch.tensor([0.4], device=DEVICE),
            torch.tensor([0.6], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
        )
        f_in = fundamental_freq(signal[0].cpu().numpy(), SAMPLE_RATE)
        f_out = fundamental_freq(out[0].detach().cpu().numpy(), SAMPLE_RATE)
        assert abs(f_out - f_in) < 2.0


class TestDelayTimbres:
    def setup_method(self):
        self.delay = Delay(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def _impulse(self):
        sig = torch.zeros(1, N_SAMPLES, device=DEVICE)
        # Short burst at the start
        burst_len = int(SAMPLE_RATE * 0.01)
        sig[0, :burst_len] = torch.sin(
            2 * 3.14159 * 1000.0 * torch.arange(burst_len, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        )
        return sig

    def test_mix_zero_bypass(self):
        signal = _sine_tensor()
        out = self.delay(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-5)

    def test_echo_at_correct_position(self):
        import math
        signal = self._impulse()
        delay_norm = 0.5
        log_min = math.log(10.0)
        log_max = math.log(500.0)
        delay_ms = math.exp(delay_norm * (log_max - log_min) + log_min)
        delay_samples = int(delay_ms / 1000.0 * SAMPLE_RATE)

        out = self.delay(
            signal,
            torch.tensor([delay_norm], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.8], device=DEVICE),
        )
        audio = out[0].detach().cpu().numpy()
        env = rms_envelope(audio, hop=256)
        # Find peaks in envelope
        hop_samples = 256
        expected_peak_frame = delay_samples // hop_samples
        search_start = max(0, expected_peak_frame - 5)
        search_end = min(len(env), expected_peak_frame + 5)
        echo_region = env[search_start:search_end]
        assert echo_region.max() > env[:search_start].min() * 2.0

    def test_feedback_multiple_echoes(self):
        signal = self._impulse()
        out = self.delay(
            signal,
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.6], device=DEVICE),
            torch.tensor([0.8], device=DEVICE),
        )
        env = rms_envelope(out[0].detach().cpu().numpy(), hop=256)
        # Count peaks above noise floor
        threshold = env.max() * 0.05
        peaks = []
        for i in range(1, len(env) - 1):
            if env[i] > threshold and env[i] > env[i - 1] and env[i] > env[i + 1]:
                peaks.append(i)
        assert len(peaks) >= 2, f"Expected 2+ echo peaks, found {len(peaks)}"

    def test_zero_feedback_single_echo(self):
        signal = self._impulse()
        out = self.delay(
            signal,
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
            torch.tensor([0.8], device=DEVICE),
        )
        env = rms_envelope(out[0].detach().cpu().numpy(), hop=256)
        threshold = env.max() * 0.1
        peaks = []
        for i in range(1, len(env) - 1):
            if env[i] > threshold and env[i] > env[i - 1] and env[i] > env[i + 1]:
                peaks.append(i)
        assert len(peaks) <= 2


class TestReverbTimbres:
    def setup_method(self):
        self.reverb = Reverb(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def _impulse(self):
        sig = torch.zeros(1, N_SAMPLES, device=DEVICE)
        burst_len = int(SAMPLE_RATE * 0.05)
        sig[0, :burst_len] = torch.sin(
            2 * 3.14159 * 440.0 * torch.arange(burst_len, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        )
        return sig

    def test_mix_zero_bypass(self):
        signal = _sine_tensor()
        out = self.reverb(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.0], device=DEVICE),
        )
        assert torch.allclose(out, signal, atol=1e-5)

    def test_adds_tail_energy(self):
        signal = self._impulse()
        out = self.reverb(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.3], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
        )
        audio = out[0].detach().cpu().numpy()
        tail = audio[int(SAMPLE_RATE * 0.5):]
        assert np.sqrt(np.mean(tail ** 2)) > 1e-4

    def test_larger_room_longer_tail(self):
        signal = self._impulse()
        tails = []
        for room in [0.3, 0.8]:
            out = self.reverb(
                signal,
                torch.tensor([room], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([0.3], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
            )
            audio = out[0].detach().cpu().numpy()
            # Bounded window avoids circular wrapping artifacts from freq-domain FDN
            tail_rms = np.sqrt(np.mean(audio[int(SAMPLE_RATE * 1.5):int(SAMPLE_RATE * 2.5)] ** 2))
            tails.append(tail_rms)
        assert tails[1] > tails[0]

    def test_damping_reduces_high_freq(self):
        signal = self._impulse()
        mid_low_ratios = []
        for damp in [0.2, 0.8]:
            out = self.reverb(
                signal,
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([damp], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
            )
            tail = out[0].detach().cpu().numpy()[int(SAMPLE_RATE * 0.5):]
            fft = np.abs(np.fft.rfft(tail))
            freqs = np.fft.rfftfreq(len(tail), 1.0 / SAMPLE_RATE)
            low_energy = np.sum(fft[freqs < 500] ** 2)
            mid_energy = np.sum(fft[(freqs >= 500) & (freqs < 5000)] ** 2)
            mid_low_ratios.append(mid_energy / (low_energy + 1e-20))
        assert mid_low_ratios[1] < mid_low_ratios[0], (
            f"High damping should reduce mid/low energy ratio: "
            f"damp=0.2 ratio={mid_low_ratios[0]:.6f}, damp=0.8 ratio={mid_low_ratios[1]:.6f}"
        )


class TestEQTimbres:
    def setup_method(self):
        self.eq = EQ(sample_rate=SAMPLE_RATE).to(DEVICE)

    def _white_noise(self):
        torch.manual_seed(42)
        return torch.randn(1, N_SAMPLES, device=DEVICE)

    def test_neutral_passthrough(self):
        signal = self._white_noise()
        out = self.eq(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
        )
        diff_db = 20 * np.log10(
            (out - signal).abs().max().item() / signal.abs().max().item() + 1e-10
        )
        assert diff_db < -35

    def _band_energy(self, audio, low_hz, high_hz):
        fft = np.abs(np.fft.rfft(audio))
        freqs = np.fft.rfftfreq(len(audio), 1.0 / SAMPLE_RATE)
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        return np.sum(fft[mask] ** 2)

    def test_low_boost(self):
        signal = self._white_noise()
        out = self.eq(
            signal,
            torch.tensor([0.8], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
        )
        e_in = self._band_energy(signal[0].cpu().numpy(), 20, 300)
        e_out = self._band_energy(out[0].detach().cpu().numpy(), 20, 300)
        gain_db = 10 * np.log10(e_out / (e_in + 1e-10))
        assert gain_db > 3.0

    def test_high_cut(self):
        signal = self._white_noise()
        out = self.eq(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.2], device=DEVICE),
        )
        e_in = self._band_energy(signal[0].cpu().numpy(), 5000, 20000)
        e_out = self._band_energy(out[0].detach().cpu().numpy(), 5000, 20000)
        gain_db = 10 * np.log10(e_out / (e_in + 1e-10))
        assert gain_db < -3.0

    def test_band_independence(self):
        signal = self._white_noise()
        out = self.eq(
            signal,
            torch.tensor([0.5], device=DEVICE),
            torch.tensor([0.8], device=DEVICE),
            torch.tensor([0.5], device=DEVICE),
        )
        e_low_in = self._band_energy(signal[0].cpu().numpy(), 20, 150)
        e_low_out = self._band_energy(out[0].detach().cpu().numpy(), 20, 150)
        e_high_in = self._band_energy(signal[0].cpu().numpy(), 8000, 20000)
        e_high_out = self._band_energy(out[0].detach().cpu().numpy(), 8000, 20000)
        low_change = abs(10 * np.log10(e_low_out / (e_low_in + 1e-10)))
        high_change = abs(10 * np.log10(e_high_out / (e_high_in + 1e-10)))
        assert low_change < 1.5
        assert high_change < 1.5
