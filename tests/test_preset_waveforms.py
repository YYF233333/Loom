"""Integration tests: engine waveforms vs external reference audio (mp3).

Each test class renders a basic waveform with the engine and compares
it against a reference mp3 generated externally. Validates fundamental
frequency, harmonic structure, and overall timbral similarity via
mel-spectrogram distance.

Reference files use an octave convention where "C4" = MIDI 48 (130.81 Hz)
and "A2" = MIDI 33 (55 Hz).
"""
import torch
import numpy as np
import os
import pytest
import soundfile as sf

from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, DEVICE
from tests.conftest import REFERENCE_DIR, save_test_wav
from tests.timbre_helpers import (
    fundamental_freq, harmonic_amplitudes, spectral_centroid,
    thd, mel_spectrogram_distance,
)

N_SAMPLES = SAMPLE_RATE * 4


def _load_reference(filename):
    path = os.path.join(REFERENCE_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"Reference not found: {path}")
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    data = data[:SAMPLE_RATE * 4].astype(np.float32)
    assert sr == SAMPLE_RATE, f"Sample rate mismatch: {sr} != {SAMPLE_RATE}"
    return data


def _normalize(audio):
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    return audio / peak


def _midi_to_pitch(midi):
    return (midi - 24.0) / (96.0 - 24.0)


def _transparent_params():
    """Synth params with no effects, open filter, flat envelope."""
    return {
        "osc_pitch": torch.tensor([0.5]),
        "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "osc_detune": torch.tensor([0.5]),
        "osc_type": torch.tensor([[1.0, 0.0, 0.0]]),
        "wt_position": torch.tensor([0.5]),
        "fm_carrier_ratio": torch.tensor([0.0]),
        "fm_mod_ratio": torch.tensor([0.0]),
        "fm_mod_index": torch.tensor([0.0]),
        "lfo_rate": torch.tensor([0.5]),
        "lfo_depth": torch.tensor([0.0]),
        "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        "lfo_target": torch.zeros(1, 4),
        "lfo_phase": torch.tensor([0.0]),
        "amp_attack": torch.tensor([0.0]),
        "amp_decay": torch.tensor([0.3]),
        "amp_sustain": torch.tensor([1.0]),
        "amp_release": torch.tensor([1.0]),
        "filter_cutoff": torch.tensor([1.0]),
        "filter_q": torch.tensor([0.0]),
        "filter_type": torch.tensor([[1.0, 0.0, 0.0]]),
        "filt_env_attack": torch.tensor([0.2]),
        "filt_env_decay": torch.tensor([0.4]),
        "filt_env_sustain": torch.tensor([0.3]),
        "filt_env_release": torch.tensor([0.3]),
        "filt_env_amount": torch.tensor([0.5]),
        "filter_mix": torch.tensor([1.0]),
        "dist_amount": torch.tensor([0.0]),
        "dist_mix": torch.tensor([0.0]),
        "master_gain": torch.tensor([0.85]),
        "comp_threshold": torch.tensor([0.5]),
        "comp_ratio": torch.tensor([0.3]),
        "comp_attack": torch.tensor([0.5]),
        "comp_release": torch.tensor([0.5]),
        "comp_makeup": torch.tensor([0.0]),
        "comp_mix": torch.tensor([0.0]),
        "chorus_rate": torch.tensor([0.5]),
        "chorus_depth": torch.tensor([0.5]),
        "chorus_mix": torch.tensor([0.0]),
        "delay_time": torch.tensor([0.5]),
        "delay_feedback": torch.tensor([0.3]),
        "delay_mix": torch.tensor([0.0]),
        "reverb_room_size": torch.tensor([0.5]),
        "reverb_decay": torch.tensor([0.5]),
        "reverb_damping": torch.tensor([0.3]),
        "reverb_mix": torch.tensor([0.0]),
        "eq_low_gain": torch.tensor([0.5]),
        "eq_mid_gain": torch.tensor([0.5]),
        "eq_high_gain": torch.tensor([0.5]),
    }


class WaveformTestBase:
    """Base for waveform reference comparison tests.

    Rendered audio is saved to tests/output/ for manual inspection.
    """

    REFERENCE_FILE = None
    EXPECTED_F0 = None
    PARAMS_OVERRIDE = {}

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES,
            note_on_duration=5.0,
        ).to(DEVICE)
        self._audio = None
        self._ref = None

    @property
    def audio_np(self):
        if self._audio is None:
            params = _transparent_params()
            params.update(self.PARAMS_OVERRIDE)
            params = {k: v.to(DEVICE) for k, v in params.items()}
            with torch.no_grad():
                out = self.synth(params)
            self._audio = out[0].cpu().numpy()
        return self._audio

    @property
    def ref_np(self):
        if self._ref is None:
            self._ref = _load_reference(self.REFERENCE_FILE)
        return self._ref

    def _steady_slice(self, audio, start_sec=0.5, end_sec=3.0):
        """Extract the sustained portion, avoiding attack/release artifacts."""
        s = int(start_sec * SAMPLE_RATE)
        e = int(end_sec * SAMPLE_RATE)
        return audio[s:e]

    def test_save_audio(self):
        """Save rendered audio for manual inspection."""
        name = self.REFERENCE_FILE.replace(".mp3", "").replace(".", "_")
        path = save_test_wav(self.audio_np, f"waveform_{name}")
        assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Sine wave at C4 (MIDI 48, 130.81 Hz)
# ---------------------------------------------------------------------------

class TestSineC4(WaveformTestBase):
    REFERENCE_FILE = "sin_C4.mp3"
    EXPECTED_F0 = 130.81
    PARAMS_OVERRIDE = {
        "osc_pitch": torch.tensor([_midi_to_pitch(48)]),
        "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.audio_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 2.0, (
            f"Engine f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_reference_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.ref_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 2.0, (
            f"Reference f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_low_thd(self):
        """Pure sine should have very low total harmonic distortion."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        distortion = thd(steady, SAMPLE_RATE, f0)
        assert distortion < 0.05, f"THD={distortion:.4f}, expected < 0.05"

    def test_fundamental_power_concentration(self):
        """Fundamental should carry most of the signal power."""
        steady = self._steady_slice(self.audio_np)
        fft = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(len(steady), 1.0 / SAMPLE_RATE)
        power = fft ** 2
        fund_power = power[(freqs >= 120) & (freqs <= 145)].sum()
        total_power = power.sum()
        ratio = fund_power / total_power
        assert ratio > 0.95, (
            f"Fundamental power ratio={ratio:.4f}, expected > 0.95"
        )

    def test_mel_distance_to_reference(self):
        engine = _normalize(self._steady_slice(self.audio_np))
        ref = _normalize(self._steady_slice(self.ref_np))
        dist = mel_spectrogram_distance(engine, ref, SAMPLE_RATE)
        assert dist < 5.0, f"Mel-spec distance={dist:.4f}, expected < 5.0"


# ---------------------------------------------------------------------------
# Sawtooth wave at C4 (MIDI 48, 130.81 Hz)
# ---------------------------------------------------------------------------

class TestSawC4(WaveformTestBase):
    REFERENCE_FILE = "saw_C4.mp3"
    EXPECTED_F0 = 130.81
    PARAMS_OVERRIDE = {
        "osc_pitch": torch.tensor([_midi_to_pitch(48)]),
        "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
    }

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.audio_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 2.0, (
            f"Engine f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_harmonic_decay(self):
        """Sawtooth harmonics decay as 1/n → ~6dB per octave."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 8)
        for k in range(1, 7):
            expected_drop = 20 * np.log10(1.0 / (k + 1))
            actual_drop = amps[k] - amps[0]
            assert abs(actual_drop - expected_drop) < 3.0, (
                f"Harmonic {k+1}: expected {expected_drop:.1f}dB, "
                f"got {actual_drop:.1f}dB"
            )

    def test_reference_harmonic_decay(self):
        """Reference saw should also show 1/n decay."""
        steady = self._steady_slice(self.ref_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 6)
        for k in range(1, 5):
            expected_drop = 20 * np.log10(1.0 / (k + 1))
            actual_drop = amps[k] - amps[0]
            assert abs(actual_drop - expected_drop) < 5.0, (
                f"Reference harmonic {k+1}: expected {expected_drop:.1f}dB, "
                f"got {actual_drop:.1f}dB"
            )

    def test_high_thd(self):
        """Sawtooth should have significant harmonic content."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        distortion = thd(steady, SAMPLE_RATE, f0)
        assert distortion > 0.3, f"THD={distortion:.4f}, expected > 0.3"

    def test_spectral_centroid_above_f0(self):
        """Rich harmonics push the centroid well above the fundamental."""
        steady = self._steady_slice(self.audio_np)
        sc = spectral_centroid(steady, SAMPLE_RATE)
        assert sc > self.EXPECTED_F0 * 2, (
            f"Spectral centroid={sc:.1f}Hz, expected > {self.EXPECTED_F0*2:.1f}Hz"
        )

    def test_mel_distance_to_reference(self):
        engine = _normalize(self._steady_slice(self.audio_np))
        ref = _normalize(self._steady_slice(self.ref_np))
        dist = mel_spectrogram_distance(engine, ref, SAMPLE_RATE)
        assert dist < 1.5, f"Mel-spec distance={dist:.4f}, expected < 1.5"


# ---------------------------------------------------------------------------
# Square wave at C4 (MIDI 48, 130.81 Hz)
# ---------------------------------------------------------------------------

class TestSquareC4(WaveformTestBase):
    REFERENCE_FILE = "square_C4.mp3"
    EXPECTED_F0 = 130.81
    PARAMS_OVERRIDE = {
        "osc_pitch": torch.tensor([_midi_to_pitch(48)]),
        "osc_waveform": torch.tensor([[0.0, 0.0, 1.0, 0.0]]),
    }

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.audio_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 2.0, (
            f"Engine f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_odd_harmonics_dominant(self):
        """Square wave: odd harmonics (3, 5, 7) present, even (2, 4, 6) suppressed."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 8)
        for k in range(1, 7):
            harmonic_num = k + 1
            if harmonic_num % 2 == 0:
                assert amps[k] < amps[0] - 30, (
                    f"Even harmonic {harmonic_num}: {amps[k]:.1f}dB should be "
                    f"< {amps[0]-30:.1f}dB (fundamental - 30dB)"
                )
            else:
                expected_drop = 20 * np.log10(1.0 / harmonic_num)
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 3.0, (
                    f"Odd harmonic {harmonic_num}: expected {expected_drop:.1f}dB, "
                    f"got {actual_drop:.1f}dB"
                )

    def test_reference_odd_harmonics(self):
        """Reference square should show odd-harmonic dominance."""
        steady = self._steady_slice(self.ref_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 6)
        for k in range(1, 5):
            harmonic_num = k + 1
            if harmonic_num % 2 == 0:
                assert amps[k] < amps[0] - 20, (
                    f"Reference even harmonic {harmonic_num}: "
                    f"{amps[k]:.1f}dB should be < {amps[0]-20:.1f}dB"
                )

    def test_mel_distance_to_reference(self):
        engine = _normalize(self._steady_slice(self.audio_np))
        ref = _normalize(self._steady_slice(self.ref_np))
        dist = mel_spectrogram_distance(engine, ref, SAMPLE_RATE)
        assert dist < 1.5, f"Mel-spec distance={dist:.4f}, expected < 1.5"


# ---------------------------------------------------------------------------
# Triangle wave at C4 (MIDI 48, 130.81 Hz)
# ---------------------------------------------------------------------------

class TestTriangleC4(WaveformTestBase):
    REFERENCE_FILE = "triangle_C4.mp3"
    EXPECTED_F0 = 130.81
    PARAMS_OVERRIDE = {
        "osc_pitch": torch.tensor([_midi_to_pitch(48)]),
        "osc_waveform": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
    }

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.audio_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 2.0, (
            f"Engine f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_odd_harmonics_only(self):
        """Triangle: odd harmonics decay as 1/n^2, even harmonics suppressed."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 8)
        for k in range(1, 7):
            harmonic_num = k + 1
            if harmonic_num % 2 == 0:
                assert amps[k] < amps[0] - 30, (
                    f"Even harmonic {harmonic_num}: {amps[k]:.1f}dB should be "
                    f"< {amps[0]-30:.1f}dB"
                )
            else:
                expected_drop = 20 * np.log10(1.0 / harmonic_num ** 2)
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 3.0, (
                    f"Odd harmonic {harmonic_num}: expected {expected_drop:.1f}dB, "
                    f"got {actual_drop:.1f}dB"
                )

    def test_reference_odd_harmonics(self):
        """Reference triangle should show 1/n^2 odd-harmonic decay."""
        steady = self._steady_slice(self.ref_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        amps = harmonic_amplitudes(steady, SAMPLE_RATE, f0, 6)
        for k in range(1, 5):
            harmonic_num = k + 1
            if harmonic_num % 2 == 0:
                assert amps[k] < amps[0] - 20, (
                    f"Reference even harmonic {harmonic_num}: "
                    f"{amps[k]:.1f}dB should be < {amps[0]-20:.1f}dB"
                )

    def test_lower_thd_than_saw(self):
        """Triangle has weaker harmonics than saw (1/n^2 vs 1/n)."""
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        distortion = thd(steady, SAMPLE_RATE, f0)
        assert distortion < 0.15, f"THD={distortion:.4f}, expected < 0.15"

    def test_mel_distance_to_reference(self):
        engine = _normalize(self._steady_slice(self.audio_np))
        ref = _normalize(self._steady_slice(self.ref_np))
        dist = mel_spectrogram_distance(engine, ref, SAMPLE_RATE)
        assert dist < 3.0, f"Mel-spec distance={dist:.4f}, expected < 3.0"


# ---------------------------------------------------------------------------
# Sine wave at A2 (MIDI 33, 55.0 Hz) — subbass
# ---------------------------------------------------------------------------

class TestSineA2SubBass(WaveformTestBase):
    REFERENCE_FILE = "sin_A2.mp3"
    EXPECTED_F0 = 55.0
    PARAMS_OVERRIDE = {
        "osc_pitch": torch.tensor([_midi_to_pitch(33)]),
        "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    }

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.audio_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 1.5, (
            f"Engine f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_reference_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady_slice(self.ref_np), SAMPLE_RATE)
        assert abs(f0 - self.EXPECTED_F0) < 1.5, (
            f"Reference f0={f0:.2f}Hz, expected ~{self.EXPECTED_F0}Hz"
        )

    def test_low_thd(self):
        steady = self._steady_slice(self.audio_np)
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        distortion = thd(steady, SAMPLE_RATE, f0)
        assert distortion < 0.05, f"THD={distortion:.4f}, expected < 0.05"

    def test_fundamental_power_concentration(self):
        """Subbass fundamental should carry most of the signal power."""
        steady = self._steady_slice(self.audio_np)
        fft = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(len(steady), 1.0 / SAMPLE_RATE)
        power = fft ** 2
        fund_power = power[(freqs >= 50) & (freqs <= 62)].sum()
        total_power = power.sum()
        ratio = fund_power / total_power
        assert ratio > 0.95, (
            f"Fundamental power ratio={ratio:.4f}, expected > 0.95"
        )

    def test_mel_distance_to_reference(self):
        engine = _normalize(self._steady_slice(self.audio_np))
        ref = _normalize(self._steady_slice(self.ref_np))
        dist = mel_spectrogram_distance(engine, ref, SAMPLE_RATE)
        assert dist < 5.0, f"Mel-spec distance={dist:.4f}, expected < 5.0"
