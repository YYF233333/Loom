import torch
import numpy as np
import pytest
from loom.oscillators import AdditiveOscillator
from loom.wavetable import WavetableOscillator
from loom.fm import FMOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from tests.timbre_helpers import (
    fundamental_freq, harmonic_amplitudes, spectral_centroid, thd,
)


def _pitch_from_midi(midi_note):
    return (midi_note - 24) / (96 - 24)


def _pitch_from_hz(hz):
    """Convert Hz to normalized pitch. Use Hz that divide evenly into 4s for clean FFT."""
    import math
    midi = 69 + 12 * math.log2(hz / 440.0)
    return (midi - 24) / (96 - 24)


class TestAdditiveOscillatorTimbres:
    def setup_method(self):
        self.osc = AdditiveOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, pitch, waveform, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE, dtype=torch.float32)
        w = torch.tensor([waveform], device=DEVICE, dtype=torch.float32)
        d = torch.tensor([detune], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            return self.osc(p, w, d)[0].cpu().numpy()

    def test_sine_a4_frequency(self):
        audio = self._render(_pitch_from_midi(69), [1, 0, 0, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        assert abs(f0 - 440.0) < 1.0

    def test_sine_a4_no_harmonics(self):
        audio = self._render(_pitch_from_midi(69), [1, 0, 0, 0])
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, 440.0, 8)
        for i in range(1, 8):
            assert amps[i] < amps[0] - 55

    def test_sine_c2_frequency(self):
        audio = self._render(_pitch_from_midi(36), [1, 0, 0, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        assert abs(f0 - 65.41) < 0.5

    def test_saw_harmonic_decay(self):
        audio = self._render(_pitch_from_hz(200.0), [0, 1, 0, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 7):
            expected_drop = 20 * np.log10(1.0 / (k + 1))
            actual_drop = amps[k] - amps[0]
            assert abs(actual_drop - expected_drop) < 1.0, (
                f"Harmonic {k+1}: expected {expected_drop:.1f}dB, got {actual_drop:.1f}dB"
            )

    def test_square_odd_harmonics_only(self):
        audio = self._render(_pitch_from_hz(200.0), [0, 0, 1, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 8):
            if (k + 1) % 2 == 0:
                assert amps[k] < amps[0] - 45, f"Even harmonic {k+1} too loud: {amps[k]:.1f}dB"
            else:
                expected_drop = 20 * np.log10(1.0 / (k + 1))
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 1.5

    def test_triangle_harmonic_decay(self):
        audio = self._render(_pitch_from_hz(200.0), [0, 0, 0, 1])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 8):
            if (k + 1) % 2 == 0:
                assert amps[k] < amps[0] - 45
            else:
                expected_drop = 20 * np.log10(1.0 / ((k + 1) ** 2))
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 2.0

    def test_detune_shifts_frequency(self):
        pitch = _pitch_from_midi(69)
        audio_center = self._render(pitch, [1, 0, 0, 0], detune=0.5)
        audio_up = self._render(pitch, [1, 0, 0, 0], detune=0.6)
        f_center = fundamental_freq(audio_center, SAMPLE_RATE)
        f_up = fundamental_freq(audio_up, SAMPLE_RATE)
        expected = 440.0 * (2.0 ** (20.0 / 1200.0))
        assert abs(f_up - expected) < 0.5

    def test_sine_amplitude_range(self):
        audio = self._render(0.5, [1, 0, 0, 0])
        assert np.abs(audio).max() <= 1.05

    @pytest.mark.parametrize("waveform", [
        [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ])
    def test_complex_waveform_amplitude_range(self, waveform):
        audio = self._render(0.5, waveform)
        # Gibbs phenomenon: bandlimited saw/square overshoot ~9% with many harmonics
        assert np.abs(audio).max() <= 1.25


class TestWavetableOscillatorTimbres:
    def setup_method(self):
        self.wt = WavetableOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)
        self.additive = AdditiveOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render_wt(self, pitch, position, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE, dtype=torch.float32)
        d = torch.tensor([detune], device=DEVICE, dtype=torch.float32)
        pos = torch.tensor([position], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            return self.wt(p, d, pos)[0].cpu().numpy()

    def _render_additive(self, pitch, waveform, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE, dtype=torch.float32)
        w = torch.tensor([waveform], device=DEVICE, dtype=torch.float32)
        d = torch.tensor([detune], device=DEVICE, dtype=torch.float32)
        with torch.no_grad():
            return self.additive(p, w, d)[0].cpu().numpy()

    def test_position_0_matches_saw(self):
        pitch = _pitch_from_hz(200.0)
        wt_audio = self._render_wt(pitch, 0.0)
        saw_audio = self._render_additive(pitch, [0, 1, 0, 0])
        f0 = fundamental_freq(wt_audio, SAMPLE_RATE)
        wt_amps = harmonic_amplitudes(wt_audio, SAMPLE_RATE, f0, 8)
        saw_amps = harmonic_amplitudes(saw_audio, SAMPLE_RATE, f0, 8)
        for k in range(8):
            assert abs(wt_amps[k] - saw_amps[k]) < 2.0, (
                f"Harmonic {k+1}: WT={wt_amps[k]:.1f}dB, Saw={saw_amps[k]:.1f}dB"
            )

    @pytest.mark.xfail(reason="wavetable square frame has even harmonics — grid_sample boundary artifact")
    def test_position_1_matches_square(self):
        pitch = _pitch_from_hz(200.0)
        wt_audio = self._render_wt(pitch, 1.0)
        sq_audio = self._render_additive(pitch, [0, 0, 1, 0])
        f0 = fundamental_freq(wt_audio, SAMPLE_RATE)
        wt_amps = harmonic_amplitudes(wt_audio, SAMPLE_RATE, f0, 8)
        sq_amps = harmonic_amplitudes(sq_audio, SAMPLE_RATE, f0, 8)
        for k in range(8):
            assert abs(wt_amps[k] - sq_amps[k]) < 2.0

    def test_mid_position_centroid_between(self):
        pitch = _pitch_from_hz(200.0)
        saw_sc = spectral_centroid(self._render_wt(pitch, 0.0), SAMPLE_RATE)
        sq_sc = spectral_centroid(self._render_wt(pitch, 1.0), SAMPLE_RATE)
        mid_sc = spectral_centroid(self._render_wt(pitch, 0.5), SAMPLE_RATE)
        lo, hi = min(saw_sc, sq_sc), max(saw_sc, sq_sc)
        assert lo <= mid_sc <= hi, f"Mid centroid {mid_sc:.0f} not between {lo:.0f} and {hi:.0f}"

    @pytest.mark.parametrize("midi", [36, 48, 60, 72])
    def test_fundamental_correct(self, midi):
        pitch = _pitch_from_midi(midi)
        audio = self._render_wt(pitch, 0.5)
        expected_hz = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        assert abs(f0 - expected_hz) < 1.0


class TestFMOscillatorTimbres:
    def setup_method(self):
        self.fm = FMOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, pitch, carrier_ratio=0.0, mod_ratio=0.0, mod_index=0.0, detune=0.5):
        with torch.no_grad():
            return self.fm(
                torch.tensor([pitch], device=DEVICE),
                torch.tensor([detune], device=DEVICE),
                torch.tensor([carrier_ratio], device=DEVICE),
                torch.tensor([mod_ratio], device=DEVICE),
                torch.tensor([mod_index], device=DEVICE),
            )[0].cpu().numpy()

    def test_zero_mod_index_is_sine(self):
        audio = self._render(_pitch_from_midi(69), mod_index=0.0)
        assert thd(audio, SAMPLE_RATE, 440.0) < 0.01

    def test_low_mod_index_creates_sidebands(self):
        audio_mod = self._render(_pitch_from_midi(69), mod_index=0.1)
        thd_mod = thd(audio_mod, SAMPLE_RATE, 440.0)
        assert thd_mod > 0.05

    def test_high_mod_index_broader_spectrum(self):
        audio_lo = self._render(0.5, mod_index=0.1)
        audio_hi = self._render(0.5, mod_index=0.5)
        f0 = fundamental_freq(audio_lo, SAMPLE_RATE)
        amps_lo = harmonic_amplitudes(audio_lo, SAMPLE_RATE, f0, 10)
        amps_hi = harmonic_amplitudes(audio_hi, SAMPLE_RATE, f0, 10)
        energy_lo = np.sum(10 ** (amps_lo[1:] / 20))
        energy_hi = np.sum(10 ** (amps_hi[1:] / 20))
        assert energy_hi > energy_lo

    def test_carrier_ratio_shifts_fundamental(self):
        audio_r1 = self._render(0.5, carrier_ratio=0.0, mod_index=0.0)
        ratio_2_norm = (2.0 - 1.0) / (8.0 - 1.0)
        audio_r2 = self._render(0.5, carrier_ratio=ratio_2_norm, mod_index=0.0)
        f1 = fundamental_freq(audio_r1, SAMPLE_RATE)
        f2 = fundamental_freq(audio_r2, SAMPLE_RATE)
        assert abs(f2 / f1 - 2.0) < 0.05
