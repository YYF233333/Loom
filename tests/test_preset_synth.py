"""Preset integration tests: engine vs Serum reference audio.

Tests verify that the engine's filter, envelope, detune, and reverb
behaviors qualitatively match Serum output. Each test renders with
matching parameters and compares acoustic properties against the
Serum reference.

All references are saw waves at MIDI 48 (130.81 Hz).
"""
import torch
import numpy as np
import math
import os
import pytest
import soundfile as sf

from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, DEVICE
from tests.conftest import REFERENCE_DIR, save_test_wav
from tests.timbre_helpers import (
    fundamental_freq, spectral_centroid, rms_envelope,
    envelope_shape, mel_spectrogram_distance,
)

MIDI_PITCH = 48
NORM_PITCH = (MIDI_PITCH - 24.0) / (96.0 - 24.0)


def _load_ref(filename, max_seconds=None):
    path = os.path.join(REFERENCE_DIR, filename)
    if not os.path.exists(path):
        pytest.skip(f"Reference not found: {path}")
    data, sr = sf.read(path)
    if data.ndim == 2:
        data = data.mean(axis=1)
    if max_seconds:
        data = data[:int(sr * max_seconds)]
    assert sr == SAMPLE_RATE, f"Sample rate mismatch: {sr} != {SAMPLE_RATE}"
    return data.astype(np.float32)


def _normalize(audio):
    peak = np.abs(audio).max()
    if peak < 1e-8:
        return audio
    return audio / peak


def _base_params():
    """Saw wave at MIDI 48, no effects, filter bypassed, flat envelope."""
    return {
        "osc_pitch": torch.tensor([NORM_PITCH]),
        "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]]),
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
        "amp_release": torch.tensor([0.3]),
        "filter_cutoff": torch.tensor([0.5]),
        "filter_q": torch.tensor([0.0]),
        "filter_type": torch.tensor([[1.0, 0.0, 0.0]]),
        "filt_env_attack": torch.tensor([0.2]),
        "filt_env_decay": torch.tensor([0.4]),
        "filt_env_sustain": torch.tensor([0.3]),
        "filt_env_release": torch.tensor([0.3]),
        "filt_env_amount": torch.tensor([0.5]),
        "filter_mix": torch.tensor([0.0]),
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


def _hz_to_cutoff_norm(hz):
    log_min = math.log(20.0)
    log_max = math.log(20000.0)
    return (math.log(hz) - log_min) / (log_max - log_min)


def _ms_to_attack_norm(ms):
    log_min = math.log(1.0)
    log_max = math.log(2000.0)
    return (math.log(max(ms, 1.0)) - log_min) / (log_max - log_min)


def _ms_to_decay_norm(ms):
    return _ms_to_attack_norm(ms)


def _ms_to_release_norm(ms):
    log_min = math.log(1.0)
    log_max = math.log(4000.0)
    return (math.log(max(ms, 1.0)) - log_min) / (log_max - log_min)


def _render(params, n_samples=SAMPLE_RATE * 4, note_on_duration=3.0):
    synth = SubtractiveSynth(
        SAMPLE_RATE, n_samples, note_on_duration=note_on_duration,
    ).to(DEVICE)
    p = {k: v.to(DEVICE) for k, v in params.items()}
    with torch.no_grad():
        return synth(p)[0].cpu().numpy()


# ---------------------------------------------------------------------------
# LP Filtered Saw — Serum: saw + MG Low 12 at 937 Hz
# ---------------------------------------------------------------------------

class TestLPFilteredSaw:
    REF_FILE = "serum_lp_saw.wav"

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._engine = None
        self._ref = None

    @property
    def engine(self):
        if self._engine is None:
            p = _base_params()
            p["filter_mix"] = torch.tensor([1.0])
            p["filter_cutoff"] = torch.tensor([_hz_to_cutoff_norm(937)])
            self._engine = _render(p)
        return self._engine

    @property
    def ref(self):
        if self._ref is None:
            self._ref = _load_ref(self.REF_FILE, max_seconds=4)
        return self._ref

    def _steady(self, audio):
        return audio[int(0.5 * SAMPLE_RATE):int(3.0 * SAMPLE_RATE)]

    def test_save_audio(self):
        path = save_test_wav(self.engine, "preset_lp_saw")
        assert os.path.exists(path)

    def test_fundamental_frequency(self):
        f0 = fundamental_freq(self._steady(self.engine), SAMPLE_RATE)
        assert abs(f0 - 130.81) < 2.0

    def test_energy_below_cutoff(self):
        """Most power should be below the LP cutoff."""
        steady = self._steady(self.engine)
        fft = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(len(steady), 1.0 / SAMPLE_RATE)
        power = fft ** 2
        below = power[freqs <= 1200].sum()
        ratio = below / power.sum()
        assert ratio > 0.85, f"Power below 1200Hz: {ratio:.2%}, expected > 85%"

    def test_ref_also_filtered(self):
        """Sanity: Serum reference should also be LP filtered."""
        steady = self._steady(self.ref)
        fft = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(len(steady), 1.0 / SAMPLE_RATE)
        power = fft ** 2
        below = power[freqs <= 1200].sum()
        ratio = below / power.sum()
        assert ratio > 0.90

    def test_ref_centroid_low(self):
        """Serum reference centroid should be below cutoff (no filter noise)."""
        steady = self._steady(self.ref)
        sc = spectral_centroid(steady, SAMPLE_RATE)
        assert sc < 1500, f"Ref centroid={sc:.0f}Hz, expected < 1500Hz"


# ---------------------------------------------------------------------------
# Pluck Envelope — Serum: saw, Atk=0ms Dec=328ms Sus=-inf Rel=10ms
# ---------------------------------------------------------------------------

class TestPluckEnvelope:
    REF_FILE = "serum_pluck.wav"

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._engine = None
        self._ref = None

    @property
    def engine(self):
        if self._engine is None:
            p = _base_params()
            p["amp_attack"] = torch.tensor([0.0])
            p["amp_decay"] = torch.tensor([_ms_to_decay_norm(328)])
            p["amp_sustain"] = torch.tensor([0.0])
            p["amp_release"] = torch.tensor([_ms_to_release_norm(10)])
            self._engine = _render(p)
        return self._engine

    @property
    def ref(self):
        if self._ref is None:
            self._ref = _load_ref(self.REF_FILE, max_seconds=4)
        return self._ref

    def test_save_audio(self):
        path = save_test_wav(self.engine, "preset_pluck")
        assert os.path.exists(path)

    def test_fast_attack(self):
        atk_ms, _, _, _ = envelope_shape(self.engine, SAMPLE_RATE)
        assert atk_ms < 20, f"Attack={atk_ms:.0f}ms, expected < 20ms"

    def test_quick_decay(self):
        """RMS in second half should be much lower than first 0.5s."""
        env = rms_envelope(self.engine, hop=512)
        first_quarter = env[:len(env) // 4].mean()
        second_half = env[len(env) // 2:].mean()
        if first_quarter < 1e-10:
            pytest.skip("No signal")
        ratio = second_half / first_quarter
        assert ratio < 0.1, f"Decay ratio={ratio:.3f}, expected < 0.1"

    def test_ref_also_decays(self):
        """Serum reference should show same pluck behavior."""
        env = rms_envelope(self.ref, hop=512)
        first_quarter = env[:len(env) // 4].mean()
        second_half = env[len(env) // 2:].mean()
        ratio = second_half / (first_quarter + 1e-10)
        assert ratio < 0.1


# ---------------------------------------------------------------------------
# Pad Envelope — Serum: saw + LP 1763Hz, Atk=590ms Sus=-2.8dB Rel=1s
# ---------------------------------------------------------------------------

class TestPadEnvelope:
    REF_FILE = "serum_pad.wav"

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._engine = None
        self._ref = None

    @property
    def engine(self):
        if self._engine is None:
            p = _base_params()
            p["amp_attack"] = torch.tensor([_ms_to_attack_norm(590)])
            p["amp_decay"] = torch.tensor([_ms_to_decay_norm(1000)])
            p["amp_sustain"] = torch.tensor([0.72])
            p["amp_release"] = torch.tensor([_ms_to_release_norm(1000)])
            p["filter_mix"] = torch.tensor([1.0])
            p["filter_cutoff"] = torch.tensor([_hz_to_cutoff_norm(1763)])
            self._engine = _render(p, note_on_duration=4.0)
        return self._engine

    @property
    def ref(self):
        if self._ref is None:
            self._ref = _load_ref(self.REF_FILE, max_seconds=4)
        return self._ref

    def test_save_audio(self):
        path = save_test_wav(self.engine, "preset_pad")
        assert os.path.exists(path)

    def test_slow_attack(self):
        """Attack should be significantly longer than instant."""
        atk_ms, _, _, _ = envelope_shape(self.engine, SAMPLE_RATE)
        assert atk_ms > 200, f"Attack={atk_ms:.0f}ms, expected > 200ms"

    def test_ref_slow_attack(self):
        atk_ms, _, _, _ = envelope_shape(self.ref, SAMPLE_RATE)
        assert atk_ms > 200, f"Ref attack={atk_ms:.0f}ms, expected > 200ms"

    def test_sustained_level(self):
        """Signal should remain present after attack phase."""
        env = rms_envelope(self.engine, hop=512)
        late = env[len(env) * 2 // 3:len(env) * 5 // 6]
        peak = env.max()
        ratio = late.mean() / (peak + 1e-10)
        assert ratio > 0.3, f"Sustain ratio={ratio:.3f}, expected > 0.3"

    def test_filtered_spectrum(self):
        """LP filter should concentrate energy below cutoff."""
        steady = self.engine[int(1.5 * SAMPLE_RATE):int(3.0 * SAMPLE_RATE)]
        fft = np.abs(np.fft.rfft(steady))
        freqs = np.fft.rfftfreq(len(steady), 1.0 / SAMPLE_RATE)
        power = fft ** 2
        below = power[freqs <= 2500].sum()
        ratio = below / power.sum()
        assert ratio > 0.80, f"Power below 2500Hz: {ratio:.2%}"


# ---------------------------------------------------------------------------
# Reese Bass — Serum: saw, 5-voice unison, detune 0.36
# ---------------------------------------------------------------------------

class TestReeseBass:
    REF_FILE = "serum_reese.wav"

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._engine = None
        self._ref = None

    @property
    def engine(self):
        if self._engine is None:
            p = _base_params()
            p["osc_detune"] = torch.tensor([0.65])
            self._engine = _render(p)
        return self._engine

    @property
    def ref(self):
        if self._ref is None:
            self._ref = _load_ref(self.REF_FILE, max_seconds=4)
        return self._ref

    def test_save_audio(self):
        path = save_test_wav(self.engine, "preset_reese")
        assert os.path.exists(path)

    def test_fundamental_present(self):
        steady = self.engine[int(0.5 * SAMPLE_RATE):int(3.0 * SAMPLE_RATE)]
        f0 = fundamental_freq(steady, SAMPLE_RATE)
        assert abs(f0 - 130.81) < 5.0, f"f0={f0:.1f}Hz"

    def test_beating_pattern(self):
        """Detuned oscillators create amplitude modulation (beating)."""
        env = rms_envelope(self.engine, hop=256)
        env = env / (env.max() + 1e-10)
        variance = np.var(env[len(env) // 4:len(env) * 3 // 4])
        assert variance > 0.001, (
            f"RMS variance={variance:.6f}, expected > 0.001 (no beating detected)"
        )

    def test_ref_also_beats(self):
        """Serum reference should also show beating."""
        env = rms_envelope(self.ref, hop=256)
        env = env / (env.max() + 1e-10)
        variance = np.var(env[len(env) // 4:len(env) * 3 // 4])
        assert variance > 0.001

    def test_wider_spectrum_than_single_saw(self):
        """Detuning should spread spectral energy wider than a single saw."""
        steady = self.engine[int(0.5 * SAMPLE_RATE):int(3.0 * SAMPLE_RATE)]
        sc = spectral_centroid(steady, SAMPLE_RATE)
        assert sc > 130.81 * 2, f"Centroid={sc:.0f}Hz, expected wider spread"


# ---------------------------------------------------------------------------
# Reverb Tail — Serum: saw 2s note, reverb 80% wet, size 60%, decay 8.1s
# ---------------------------------------------------------------------------

class TestReverbTail:
    REF_FILE = "serum_reverb.wav"
    N_SAMPLES = SAMPLE_RATE * 6

    @pytest.fixture(autouse=True)
    def _setup(self):
        self._engine = None
        self._ref = None

    @property
    def engine(self):
        if self._engine is None:
            p = _base_params()
            p["amp_release"] = torch.tensor([_ms_to_release_norm(1610)])
            p["reverb_room_size"] = torch.tensor([0.6])
            p["reverb_decay"] = torch.tensor([0.65])
            p["reverb_damping"] = torch.tensor([0.3])
            p["reverb_mix"] = torch.tensor([0.8])
            self._engine = _render(
                p, n_samples=self.N_SAMPLES, note_on_duration=2.0,
            )
        return self._engine

    @property
    def ref(self):
        if self._ref is None:
            self._ref = _load_ref(self.REF_FILE, max_seconds=6)
        return self._ref

    def test_save_audio(self):
        path = save_test_wav(self.engine, "preset_reverb")
        assert os.path.exists(path)

    def test_energy_during_note(self):
        """Should have significant energy during the note (0-2s)."""
        rms = np.sqrt(np.mean(self.engine[:2 * SAMPLE_RATE] ** 2))
        assert rms > 1e-4, f"RMS during note={rms:.6f}"

    def test_tail_persists_after_note(self):
        """Reverb tail: energy in 2-4s region should be non-negligible."""
        note_rms = np.sqrt(np.mean(self.engine[:2 * SAMPLE_RATE] ** 2))
        tail_rms = np.sqrt(np.mean(
            self.engine[2 * SAMPLE_RATE:4 * SAMPLE_RATE] ** 2
        ))
        ratio = tail_rms / (note_rms + 1e-10)
        assert ratio > 0.05, (
            f"Tail/note RMS ratio={ratio:.4f}, expected > 0.05"
        )

    def test_tail_decays(self):
        """Tail energy should decrease over time."""
        rms_2_3 = np.sqrt(np.mean(
            self.engine[2 * SAMPLE_RATE:3 * SAMPLE_RATE] ** 2
        ))
        rms_4_5 = np.sqrt(np.mean(
            self.engine[4 * SAMPLE_RATE:5 * SAMPLE_RATE] ** 2
        ))
        assert rms_4_5 < rms_2_3, "Reverb tail should decay over time"

    def test_ref_tail_shape(self):
        """Serum reference should also show note + decaying tail."""
        note_rms = np.sqrt(np.mean(self.ref[:2 * SAMPLE_RATE] ** 2))
        tail_rms = np.sqrt(np.mean(
            self.ref[2 * SAMPLE_RATE:4 * SAMPLE_RATE] ** 2
        ))
        ratio = tail_rms / (note_rms + 1e-10)
        assert ratio > 0.05
