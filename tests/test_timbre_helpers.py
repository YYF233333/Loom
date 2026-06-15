import numpy as np
import pytest
from tests.timbre_helpers import (
    fundamental_freq,
    harmonic_amplitudes,
    spectral_centroid,
    spectral_rolloff,
    rms_envelope,
    thd,
    envelope_shape,
    mel_spectrogram_distance,
    freq_response,
)

SR = 44100


def _sine(freq, duration=1.0, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    return np.sin(2 * np.pi * freq * t)


def _saw(freq, n_harmonics=50, duration=1.0, sr=SR):
    t = np.arange(int(sr * duration)) / sr
    signal = np.zeros_like(t)
    for k in range(1, n_harmonics + 1):
        signal += ((-1) ** (k + 1)) * np.sin(2 * np.pi * freq * k * t) / k
    return signal * (2.0 / np.pi)


class TestFundamentalFreq:
    def test_440hz_sine(self):
        audio = _sine(440.0)
        assert abs(fundamental_freq(audio, SR) - 440.0) < 1.0

    def test_100hz_sine(self):
        audio = _sine(100.0)
        assert abs(fundamental_freq(audio, SR) - 100.0) < 1.0

    def test_1000hz_sine(self):
        audio = _sine(1000.0)
        assert abs(fundamental_freq(audio, SR) - 1000.0) < 1.0


class TestHarmonicAmplitudes:
    def test_sine_single_harmonic(self):
        audio = _sine(440.0)
        amps = harmonic_amplitudes(audio, SR, 440.0, 4)
        assert amps[0] > -3.0
        for i in range(1, 4):
            assert amps[i] < -50.0

    def test_saw_decay(self):
        audio = _saw(200.0)
        amps = harmonic_amplitudes(audio, SR, 200.0, 8)
        for k in range(1, 7):
            expected_drop = 20 * np.log10(1.0 / (k + 1)) - 20 * np.log10(1.0)
            actual_drop = amps[k] - amps[0]
            assert abs(actual_drop - expected_drop) < 1.5


class TestSpectralCentroid:
    def test_sine_centroid_at_fundamental(self):
        audio = _sine(1000.0)
        sc = spectral_centroid(audio, SR)
        assert abs(sc - 1000.0) < 50.0

    def test_saw_higher_than_sine(self):
        sine = _sine(440.0)
        saw = _saw(440.0)
        assert spectral_centroid(saw, SR) > spectral_centroid(sine, SR)


class TestSpectralRolloff:
    def test_sine_rolloff_near_fundamental(self):
        audio = _sine(440.0)
        ro = spectral_rolloff(audio, SR, 0.85)
        assert ro < 1000.0

    def test_saw_rolloff_higher(self):
        sine = _sine(440.0)
        saw = _saw(440.0)
        assert spectral_rolloff(saw, SR, 0.85) > spectral_rolloff(sine, SR, 0.85)


class TestRmsEnvelope:
    def test_constant_signal(self):
        audio = np.ones(SR)
        env = rms_envelope(audio, hop=512)
        assert np.std(env) < 0.05
        assert abs(np.mean(env) - 1.0) < 0.1

    def test_decaying_signal(self):
        t = np.arange(SR) / SR
        audio = np.sin(2 * np.pi * 440 * t) * np.exp(-3.0 * t)
        env = rms_envelope(audio, hop=512)
        assert env[0] > env[-1]


class TestTHD:
    def test_pure_sine_low_thd(self):
        audio = _sine(440.0)
        assert thd(audio, SR, 440.0) < 0.01

    def test_clipped_sine_high_thd(self):
        audio = np.clip(_sine(440.0) * 2.0, -1.0, 1.0)
        assert thd(audio, SR, 440.0) > 0.1


class TestEnvelopeShape:
    def test_attack_decay_sustain_release(self):
        n = SR * 2
        env = np.zeros(n)
        attack_samples = int(0.05 * SR)
        decay_samples = int(0.1 * SR)
        sustain_end = int(1.0 * SR)
        release_samples = int(0.2 * SR)
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        env[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.6, decay_samples)
        env[attack_samples + decay_samples:sustain_end] = 0.6
        rel_end = min(sustain_end + release_samples, n)
        env[sustain_end:rel_end] = np.linspace(0.6, 0, rel_end - sustain_end)

        audio = env * np.sin(2 * np.pi * 440 * np.arange(n) / SR)
        attack_ms, peak, sustain_level, release_ms = envelope_shape(audio, SR)
        assert abs(attack_ms - 50.0) < 20.0
        assert abs(peak - 1.0) < 0.15
        assert abs(sustain_level - 0.6) < 0.15


class TestMelSpectrogramDistance:
    def test_identical_signals(self):
        audio = _sine(440.0)
        assert mel_spectrogram_distance(audio, audio, SR) < 1e-6

    def test_different_signals(self):
        a = _sine(440.0)
        b = _sine(880.0)
        assert mel_spectrogram_distance(a, b, SR) > 0.01


class TestFreqResponse:
    def test_passthrough(self):
        def identity(x):
            return x
        freqs, mag_db = freq_response(identity, SR)
        assert np.abs(mag_db[1:len(mag_db) // 2]).max() < 1.0
