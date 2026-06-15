# Timbre Test Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a layered test framework that verifies acoustic correctness of every synth module, from oscillators up through the full effects chain.

**Architecture:** Bottom-up test pyramid — utility functions for spectral analysis, then per-module acoustic tests (oscillator → envelope/LFO → filter → effects), then full-chain preset golden snapshots. Each layer is verified before testing the next.

**Tech Stack:** pytest, numpy, scipy.signal (spectral analysis), torch (rendering)

---

### Task 1: Project Setup

**Files:**
- Modify: `pyproject.toml`
- Modify: `tests/conftest.py`
- Create: `tests/fixtures/golden/.gitkeep`
- Create: `tests/fixtures/reference/.gitkeep`

- [ ] **Step 1: Add scipy to dev dependencies**

```toml
# In pyproject.toml, update [dependency-groups] dev:
[dependency-groups]
dev = [
    "pytest>=9.1.0",
    "scipy>=1.11",
]
```

- [ ] **Step 2: Create fixture directories**

Run:
```bash
mkdir -p tests/fixtures/golden tests/fixtures/reference
touch tests/fixtures/golden/.gitkeep tests/fixtures/reference/.gitkeep
```

- [ ] **Step 3: Update conftest.py with golden test infrastructure and markers**

```python
import torch
import pytest
import os
from loom.core import DEVICE

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
GOLDEN_DIR = os.path.join(FIXTURES_DIR, "golden")
REFERENCE_DIR = os.path.join(FIXTURES_DIR, "reference")


def pytest_addoption(parser):
    parser.addoption(
        "--update-golden", action="store_true", default=False,
        help="Regenerate golden audio snapshots",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "reference: tests requiring Serum reference audio files")


def pytest_report_header():
    return (
        f"loom device: {DEVICE} (CUDA {torch.version.cuda})"
        if DEVICE.type == "cuda"
        else f"loom device: {DEVICE}"
    )


@pytest.fixture
def update_golden(request):
    return request.config.getoption("--update-golden")


@pytest.fixture
def golden_dir():
    return GOLDEN_DIR


@pytest.fixture
def reference_dir():
    return REFERENCE_DIR
```

- [ ] **Step 4: Verify setup**

Run: `uv sync --group dev`
Run: `uv run pytest --co -q`
Expected: All existing tests collected without errors.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/conftest.py tests/fixtures/
git commit -m "chore: add scipy dev dep, golden/reference fixture dirs, pytest golden infrastructure"
```

---

### Task 2: Test Utility Functions — `timbre_helpers.py`

**Files:**
- Create: `tests/timbre_helpers.py`
- Create: `tests/test_timbre_helpers.py`

- [ ] **Step 1: Write sanity tests for all helper functions**

```python
# tests/test_timbre_helpers.py
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
        assert amps[0] > -3.0  # fundamental strong
        for i in range(1, 4):
            assert amps[i] < -50.0  # harmonics absent

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
        # attack
        env[:attack_samples] = np.linspace(0, 1, attack_samples)
        # decay to 0.6
        env[attack_samples:attack_samples + decay_samples] = np.linspace(1, 0.6, decay_samples)
        # sustain
        env[attack_samples + decay_samples:sustain_end] = 0.6
        # release
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
        assert np.abs(mag_db[1:len(mag_db)//2]).max() < 1.0  # flat within 1dB
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_timbre_helpers.py -v`
Expected: ImportError — `timbre_helpers` module not found.

- [ ] **Step 3: Implement all helper functions**

```python
# tests/timbre_helpers.py
import numpy as np
from scipy.signal import stft


def _to_numpy(audio):
    if hasattr(audio, "detach"):
        return audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float64)


def fundamental_freq(audio, sr):
    audio = _to_numpy(audio).flatten()
    fft = np.fft.rfft(audio)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    peak_idx = np.argmax(magnitudes[1:]) + 1
    return float(freqs[peak_idx])


def harmonic_amplitudes(audio, sr, f0, n):
    audio = _to_numpy(audio).flatten()
    fft = np.fft.rfft(audio)
    magnitudes = np.abs(fft)
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    freq_resolution = freqs[1] - freqs[0]
    amps_db = np.zeros(n)
    for k in range(n):
        target = f0 * (k + 1)
        idx = int(round(target / freq_resolution))
        if idx < len(magnitudes):
            window = magnitudes[max(0, idx - 2):idx + 3]
            peak = window.max() if len(window) > 0 else 1e-10
        else:
            peak = 1e-10
        amps_db[k] = 20 * np.log10(max(peak, 1e-10))
    return amps_db


def spectral_centroid(audio, sr):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    total = fft.sum()
    if total < 1e-10:
        return 0.0
    return float(np.sum(freqs * fft) / total)


def spectral_rolloff(audio, sr, pct=0.85):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    cumulative = np.cumsum(fft)
    threshold = pct * cumulative[-1]
    idx = np.searchsorted(cumulative, threshold)
    return float(freqs[min(idx, len(freqs) - 1)])


def rms_envelope(audio, hop=512):
    audio = _to_numpy(audio).flatten()
    n_frames = len(audio) // hop
    env = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop:(i + 1) * hop]
        env[i] = np.sqrt(np.mean(frame ** 2))
    return env


def thd(audio, sr, f0):
    audio = _to_numpy(audio).flatten()
    fft = np.abs(np.fft.rfft(audio))
    freqs = np.fft.rfftfreq(len(audio), 1.0 / sr)
    freq_res = freqs[1] - freqs[0]

    def _peak_at(freq):
        idx = int(round(freq / freq_res))
        if idx >= len(fft):
            return 0.0
        window = fft[max(0, idx - 2):idx + 3]
        return float(window.max()) if len(window) > 0 else 0.0

    fundamental_power = _peak_at(f0) ** 2
    harmonic_power = 0.0
    for k in range(2, 20):
        harmonic_power += _peak_at(f0 * k) ** 2
    if fundamental_power < 1e-20:
        return 0.0
    return float(np.sqrt(harmonic_power / fundamental_power))


def envelope_shape(audio, sr):
    audio = _to_numpy(audio).flatten()
    env = rms_envelope(np.abs(audio), hop=256)
    hop_sec = 256 / sr

    peak_idx = np.argmax(env)
    peak = float(env[peak_idx])
    attack_ms = float(peak_idx * hop_sec * 1000)

    if peak_idx + 1 < len(env):
        sustain_region = env[peak_idx + int(len(env) * 0.1):int(len(env) * 0.6)]
        sustain_level = float(np.median(sustain_region)) if len(sustain_region) > 0 else 0.0
    else:
        sustain_level = 0.0

    threshold = peak * 0.05
    release_start = len(env) - 1
    for i in range(len(env) - 1, peak_idx, -1):
        if env[i] > threshold:
            release_start = i
            break
    tail_end = len(env) - 1
    for i in range(release_start, len(env)):
        if env[i] < threshold:
            tail_end = i
            break
    release_ms = float((tail_end - release_start) * hop_sec * 1000)

    return attack_ms, peak, sustain_level, release_ms


def mel_spectrogram_distance(a, b, sr, n_mels=128, n_fft=2048, hop=512):
    a = _to_numpy(a).flatten()
    b = _to_numpy(b).flatten()
    min_len = min(len(a), len(b))
    a, b = a[:min_len], b[:min_len]

    from scipy.signal import spectrogram as _spectrogram

    def _mel_spec(x):
        f, t, Sxx = _spectrogram(x, fs=sr, nperseg=n_fft, noverlap=n_fft - hop)
        mel_freqs = np.linspace(0, _hz_to_mel(sr / 2), n_mels + 2)
        hz_freqs = _mel_to_hz(mel_freqs)
        filterbank = np.zeros((n_mels, len(f)))
        for i in range(n_mels):
            lo, center, hi = hz_freqs[i], hz_freqs[i + 1], hz_freqs[i + 2]
            for j, freq in enumerate(f):
                if lo <= freq <= center:
                    filterbank[i, j] = (freq - lo) / (center - lo + 1e-10)
                elif center < freq <= hi:
                    filterbank[i, j] = (hi - freq) / (hi - center + 1e-10)
        mel = filterbank @ Sxx
        return np.log(mel + 1e-10)

    return float(np.mean(np.abs(_mel_spec(a) - _mel_spec(b))))


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def freq_response(filter_fn, sr, n_samples=88200, seed=42):
    np.random.seed(seed)
    noise = np.random.randn(n_samples).astype(np.float32)

    import torch
    noise_t = torch.from_numpy(noise).unsqueeze(0)
    with torch.no_grad():
        filtered_t = filter_fn(noise_t)
    filtered = filtered_t.squeeze(0).numpy()

    fft_in = np.fft.rfft(noise)
    fft_out = np.fft.rfft(filtered)
    H = fft_out / (fft_in + 1e-10)
    freqs = np.fft.rfftfreq(n_samples, 1.0 / sr)
    mag_db = 20 * np.log10(np.abs(H) + 1e-10)
    return freqs, mag_db
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_timbre_helpers.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/timbre_helpers.py tests/test_timbre_helpers.py
git commit -m "feat: add timbre_helpers spectral analysis utilities with sanity tests"
```

---

### Task 3: Oscillator Layer Tests — `test_timbres_osc.py`

**Files:**
- Create: `tests/test_timbres_osc.py`

- [ ] **Step 1: Write AdditiveOscillator acoustic tests**

```python
# tests/test_timbres_osc.py
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


class TestAdditiveOscillatorTimbres:
    def setup_method(self):
        self.osc = AdditiveOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, pitch, waveform, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE)
        w = torch.tensor([waveform], device=DEVICE)
        d = torch.tensor([detune], device=DEVICE)
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
        audio = self._render(0.3, [0, 1, 0, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 7):
            expected_drop = 20 * np.log10(1.0 / (k + 1))
            actual_drop = amps[k] - amps[0]
            assert abs(actual_drop - expected_drop) < 1.0, (
                f"Harmonic {k+1}: expected {expected_drop:.1f}dB, got {actual_drop:.1f}dB"
            )

    def test_square_odd_harmonics_only(self):
        audio = self._render(0.3, [0, 0, 1, 0])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 8):
            if (k + 1) % 2 == 0:  # even harmonic
                assert amps[k] < amps[0] - 45, f"Even harmonic {k+1} too loud: {amps[k]:.1f}dB"
            else:  # odd harmonic
                expected_drop = 20 * np.log10(1.0 / (k + 1))
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 1.5

    def test_triangle_harmonic_decay(self):
        audio = self._render(0.3, [0, 0, 0, 1])
        f0 = fundamental_freq(audio, SAMPLE_RATE)
        amps = harmonic_amplitudes(audio, SAMPLE_RATE, f0, 8)
        for k in range(1, 8):
            if (k + 1) % 2 == 0:  # even harmonic
                assert amps[k] < amps[0] - 45
            else:  # odd harmonic, decays as 1/n^2
                expected_drop = 20 * np.log10(1.0 / ((k + 1) ** 2))
                actual_drop = amps[k] - amps[0]
                assert abs(actual_drop - expected_drop) < 2.0

    def test_detune_shifts_frequency(self):
        pitch = _pitch_from_midi(69)
        audio_center = self._render(pitch, [1, 0, 0, 0], detune=0.5)
        audio_up = self._render(pitch, [1, 0, 0, 0], detune=0.6)
        f_center = fundamental_freq(audio_center, SAMPLE_RATE)
        f_up = fundamental_freq(audio_up, SAMPLE_RATE)
        # detune 0.6 -> 20 cents up -> f * 2^(20/1200)
        expected = 440.0 * (2.0 ** (20.0 / 1200.0))
        assert abs(f_up - expected) < 0.5

    @pytest.mark.parametrize("waveform", [
        [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
    ])
    def test_amplitude_range(self, waveform):
        audio = self._render(0.5, waveform)
        assert np.abs(audio).max() <= 1.05
```

- [ ] **Step 2: Write WavetableOscillator acoustic tests**

Append to `tests/test_timbres_osc.py`:

```python
class TestWavetableOscillatorTimbres:
    def setup_method(self):
        self.wt = WavetableOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)
        self.additive = AdditiveOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render_wt(self, pitch, position, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE)
        d = torch.tensor([detune], device=DEVICE)
        pos = torch.tensor([position], device=DEVICE)
        with torch.no_grad():
            return self.wt(p, d, pos)[0].cpu().numpy()

    def _render_additive(self, pitch, waveform, detune=0.5):
        p = torch.tensor([pitch], device=DEVICE)
        w = torch.tensor([waveform], device=DEVICE)
        d = torch.tensor([detune], device=DEVICE)
        with torch.no_grad():
            return self.additive(p, w, d)[0].cpu().numpy()

    def test_position_0_matches_saw(self):
        pitch = 0.3
        wt_audio = self._render_wt(pitch, 0.0)
        saw_audio = self._render_additive(pitch, [0, 1, 0, 0])
        f0 = fundamental_freq(wt_audio, SAMPLE_RATE)
        wt_amps = harmonic_amplitudes(wt_audio, SAMPLE_RATE, f0, 8)
        saw_amps = harmonic_amplitudes(saw_audio, SAMPLE_RATE, f0, 8)
        for k in range(8):
            assert abs(wt_amps[k] - saw_amps[k]) < 2.0, (
                f"Harmonic {k+1}: WT={wt_amps[k]:.1f}dB, Saw={saw_amps[k]:.1f}dB"
            )

    def test_position_1_matches_square(self):
        pitch = 0.3
        wt_audio = self._render_wt(pitch, 1.0)
        sq_audio = self._render_additive(pitch, [0, 0, 1, 0])
        f0 = fundamental_freq(wt_audio, SAMPLE_RATE)
        wt_amps = harmonic_amplitudes(wt_audio, SAMPLE_RATE, f0, 8)
        sq_amps = harmonic_amplitudes(sq_audio, SAMPLE_RATE, f0, 8)
        for k in range(8):
            assert abs(wt_amps[k] - sq_amps[k]) < 2.0

    def test_mid_position_centroid_between(self):
        pitch = 0.3
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
```

- [ ] **Step 3: Write FMOscillator acoustic tests**

Append to `tests/test_timbres_osc.py`:

```python
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
        audio_clean = self._render(_pitch_from_midi(69), mod_index=0.0)
        audio_mod = self._render(_pitch_from_midi(69), mod_index=0.1)
        sc_clean = spectral_centroid(audio_clean, SAMPLE_RATE)
        sc_mod = spectral_centroid(audio_mod, SAMPLE_RATE)
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
        # carrier_ratio=0.0 -> ratio 1.0, carrier_ratio=~0.143 -> ratio 2.0
        audio_r1 = self._render(0.5, carrier_ratio=0.0, mod_index=0.0)
        ratio_2_norm = (2.0 - 1.0) / (8.0 - 1.0)  # ~0.143
        audio_r2 = self._render(0.5, carrier_ratio=ratio_2_norm, mod_index=0.0)
        f1 = fundamental_freq(audio_r1, SAMPLE_RATE)
        f2 = fundamental_freq(audio_r2, SAMPLE_RATE)
        assert abs(f2 / f1 - 2.0) < 0.05
```

- [ ] **Step 4: Run all oscillator tests**

Run: `uv run pytest tests/test_timbres_osc.py -v`
Expected: All tests PASS (or failures reveal engine bugs to fix).

- [ ] **Step 5: Commit**

```bash
git add tests/test_timbres_osc.py
git commit -m "test: add oscillator layer acoustic tests — additive, wavetable, FM"
```

---

### Task 4: Envelope & LFO Layer Tests — `test_timbres_envelope.py`

**Files:**
- Create: `tests/test_timbres_envelope.py`

- [ ] **Step 1: Write ADSR envelope timing and shape tests**

```python
# tests/test_timbres_envelope.py
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
                torch.tensor([attack], device=DEVICE),
                torch.tensor([decay], device=DEVICE),
                torch.tensor([sustain], device=DEVICE),
                torch.tensor([release], device=DEVICE),
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
        expected_ms = self._expected_ms(0.0, 2000.0)  # 1.0ms
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
        # After attack+decay, should be near 0.6
        attack_ms = self._expected_ms(0.1, 2000.0)
        decay_ms = self._expected_ms(0.5, 2000.0)
        settle_sample = int((attack_ms + decay_ms * 3) / 1000 * SAMPLE_RATE)
        settle_sample = min(settle_sample, int(SAMPLE_RATE * 2.5))
        if settle_sample < len(env):
            level = env[settle_sample]
            assert abs(level - 0.6) < 0.1, f"Sustain level: expected 0.6, got {level:.3f}"

    def test_sustain_holds_flat(self):
        env = self._render(0.1, 0.2, 0.7, 0.3)
        # Sustain region: middle of buffer, before note_on_duration (3.0s)
        start = int(SAMPLE_RATE * 0.5)
        end = int(SAMPLE_RATE * 2.5)
        segment = env[start:end]
        assert np.std(segment) < 0.02, f"Sustain std: {np.std(segment):.4f}"

    def test_release_decays_to_zero(self):
        env = self._render(0.1, 0.2, 0.7, 0.3)
        # note_on_duration = 3.0s, release starts there
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
```

- [ ] **Step 2: Write LFO waveform and parameter tests**

Append to `tests/test_timbres_envelope.py`:

```python
class TestLFOTimbres:
    def setup_method(self):
        self.lfo = LFO(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _render(self, rate, depth, waveform, phase=0.0):
        with torch.no_grad():
            return self.lfo(
                torch.tensor([rate], device=DEVICE),
                torch.tensor([depth], device=DEVICE),
                torch.tensor([waveform], device=DEVICE),
                torch.tensor([phase], device=DEVICE),
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
        # Derivative should be mostly positive (ramp up) with periodic negative spikes
        diff = np.diff(signal)
        positive_ratio = np.sum(diff > 0) / len(diff)
        assert positive_ratio > 0.45  # mostly ramping up

    def test_square_binary_values(self):
        signal = self._render(0.3, 0.8, [0, 0, 1, 0])
        unique_abs = np.unique(np.round(np.abs(signal), decimals=2))
        # Should mostly be near 0.0 and 0.8
        assert len(unique_abs) <= 5  # some transition samples allowed
        assert np.abs(signal).max() <= 0.81

    def test_triangle_symmetry(self):
        signal = self._render(0.3, 1.0, [0, 0, 0, 1])
        # Triangle should be symmetric: positive and negative halves similar
        assert abs(signal.max() + signal.min()) < 0.1

    def test_phase_offset(self):
        signal_0 = self._render(0.3, 1.0, [1, 0, 0, 0], phase=0.0)
        signal_half = self._render(0.3, 1.0, [1, 0, 0, 0], phase=0.5)
        # phase=0.5 -> pi radians. sin(pi) = 0, so start should be near 0
        assert abs(signal_half[0]) < 0.1
        # And the two signals should be offset
        assert not np.allclose(signal_0, signal_half, atol=0.1)

    @pytest.mark.parametrize("norm,expected_hz", [
        (0.0, 0.1), (1.0, 20.0),
    ])
    def test_rate_denorm_range(self, norm, expected_hz):
        actual = self._expected_rate_hz(norm)
        assert abs(actual - expected_hz) < 0.01 * expected_hz
```

- [ ] **Step 3: Write LFO target routing tests**

Append to `tests/test_timbres_envelope.py`:

```python
class TestLFOTargetRouting:
    """Test that lfo_target vector routes modulation to the correct parameter."""

    def setup_method(self):
        from loom.synth import SubtractiveSynth
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _base_params(self):
        return {
            "osc_pitch": torch.tensor([0.4], device=DEVICE),
            "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE),
            "osc_detune": torch.full((1,), 0.5, device=DEVICE),
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "wt_position": torch.full((1,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((1,), 0.0, device=DEVICE),
            "lfo_rate": torch.full((1,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((1,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            "lfo_target": torch.zeros(1, 4, device=DEVICE),
            "lfo_phase": torch.full((1,), 0.0, device=DEVICE),
            "amp_attack": torch.full((1,), 0.1, device=DEVICE),
            "amp_decay": torch.full((1,), 0.3, device=DEVICE),
            "amp_sustain": torch.full((1,), 0.9, device=DEVICE),
            "amp_release": torch.full((1,), 0.3, device=DEVICE),
            "filter_cutoff": torch.full((1,), 0.5, device=DEVICE),
            "filter_q": torch.full((1,), 0.3, device=DEVICE),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
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
        from tests.timbre_helpers import spectral_centroid as sc_fn, rms_envelope

        params_static = self._base_params()
        params_mod = self._base_params()
        params_mod["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_mod["lfo_target"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)

        with torch.no_grad():
            audio_static = self.synth(params_static)[0].cpu().numpy()
            audio_mod = self.synth(params_mod)[0].cpu().numpy()

        assert not np.allclose(audio_static, audio_mod, atol=0.01)

    def test_drive_target_modulates_thd(self):
        from tests.timbre_helpers import thd as thd_fn

        params_static = self._base_params()
        params_mod = self._base_params()
        params_mod["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_mod["lfo_target"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=DEVICE)

        with torch.no_grad():
            audio_static = self.synth(params_static)[0].cpu().numpy()
            audio_mod = self.synth(params_mod)[0].cpu().numpy()

        assert not np.allclose(audio_static, audio_mod, atol=0.01)
```

- [ ] **Step 4: Run all envelope/LFO tests**

Run: `uv run pytest tests/test_timbres_envelope.py -v`
Expected: All tests PASS (or failures reveal engine bugs).

- [ ] **Step 5: Commit**

```bash
git add tests/test_timbres_envelope.py
git commit -m "test: add ADSR, LFO, and LFO routing acoustic tests"
```

---

### Task 5: Filter Layer Tests — `test_timbres_filter.py`

**Files:**
- Create: `tests/test_timbres_filter.py`

- [ ] **Step 1: Write SVFilter frequency response and denormalization tests**

```python
# tests/test_timbres_filter.py
import torch
import numpy as np
import math
import pytest
from loom.svfilter import SVFilter
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE
from tests.timbre_helpers import freq_response, spectral_centroid


def _cutoff_norm(hz):
    """Convert Hz to normalized [0,1] for SVFilter."""
    log_min = math.log(20.0)
    log_max = math.log(20000.0)
    return (math.log(hz) - log_min) / (log_max - log_min)


def _q_norm(q_val):
    """Convert Q value to normalized [0,1] for SVFilter."""
    log_min = math.log(0.5)
    log_max = math.log(20.0)
    return (math.log(q_val) - log_min) / (log_max - log_min)


def _find_3db_point(freqs, mag_db, direction="lowpass"):
    """Find -3dB cutoff frequency from a frequency response curve."""
    if direction == "lowpass":
        passband = np.mean(mag_db[1:max(2, len(mag_db) // 20)])
    else:
        passband = np.mean(mag_db[-len(mag_db) // 20:])
    target = passband - 3.0
    if direction == "lowpass":
        for i in range(len(mag_db) - 1):
            if mag_db[i] >= target and mag_db[i + 1] < target:
                frac = (target - mag_db[i]) / (mag_db[i + 1] - mag_db[i] + 1e-10)
                return freqs[i] + frac * (freqs[i + 1] - freqs[i])
    else:
        for i in range(len(mag_db) - 1, 0, -1):
            if mag_db[i] >= target and mag_db[i - 1] < target:
                frac = (target - mag_db[i]) / (mag_db[i - 1] - mag_db[i] + 1e-10)
                return freqs[i] + frac * (freqs[i - 1] - freqs[i])
    return 0.0


class TestSVFilterDenorm:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    @pytest.mark.parametrize("norm,expected", [
        (0.0, 20.0), (1.0, 20000.0), (0.5, 632.5),
    ])
    def test_cutoff_denorm(self, norm, expected):
        actual = self.filt._denorm_cutoff(torch.tensor([norm])).item()
        assert abs(actual - expected) / expected < 0.01

    @pytest.mark.parametrize("norm,expected", [
        (0.0, 0.5), (1.0, 20.0),
    ])
    def test_q_denorm(self, norm, expected):
        actual = self.filt._denorm_q(torch.tensor([norm])).item()
        assert abs(actual - expected) / expected < 0.01


class TestSVFilterFreqResponse:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    def _make_filter_fn(self, cutoff_hz, q_val, filter_type_vec):
        cutoff_n = _cutoff_norm(cutoff_hz)
        q_n = _q_norm(q_val)
        filt = self.filt

        def fn(x):
            x = x.to(DEVICE)
            n = x.shape[1]
            cutoff = torch.full((1, n), cutoff_n, device=DEVICE)
            q = torch.tensor([q_n], device=DEVICE)
            ft = torch.tensor([filter_type_vec], device=DEVICE, dtype=torch.float32)
            return filt(x, cutoff, q, ft).cpu()
        return fn

    def test_lp_cutoff_1000hz(self):
        fn = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        freqs, mag = freq_response(fn, SAMPLE_RATE)
        cutoff = _find_3db_point(freqs, mag, "lowpass")
        assert abs(cutoff - 1000.0) < 100.0, f"LP -3dB at {cutoff:.0f}Hz, expected 1000Hz"

    def test_lp_rolloff_slope(self):
        fn = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        freqs, mag = freq_response(fn, SAMPLE_RATE)
        # Check slope between 2000Hz and 8000Hz (2 octaves above cutoff)
        idx_2k = np.searchsorted(freqs, 2000)
        idx_8k = np.searchsorted(freqs, 8000)
        if idx_2k < len(mag) and idx_8k < len(mag):
            octaves = np.log2(8000 / 2000)
            slope = (mag[idx_8k] - mag[idx_2k]) / octaves
            assert -16 < slope < -10, f"LP slope: {slope:.1f} dB/oct, expected ~-12"

    def test_hp_cutoff_1000hz(self):
        fn = self._make_filter_fn(1000.0, 0.707, [0, 1, 0])
        freqs, mag = freq_response(fn, SAMPLE_RATE)
        cutoff = _find_3db_point(freqs, mag, "highpass")
        assert abs(cutoff - 1000.0) < 100.0, f"HP -3dB at {cutoff:.0f}Hz, expected 1000Hz"

    def test_bp_center_freq(self):
        fn = self._make_filter_fn(1000.0, 2.0, [0, 0, 1])
        freqs, mag = freq_response(fn, SAMPLE_RATE)
        peak_idx = np.argmax(mag[1:]) + 1
        peak_freq = freqs[peak_idx]
        assert abs(peak_freq - 1000.0) < 100.0

    def test_q_resonance_peak(self):
        fn_low_q = self._make_filter_fn(1000.0, 0.707, [1, 0, 0])
        fn_high_q = self._make_filter_fn(1000.0, 10.0, [1, 0, 0])
        _, mag_low = freq_response(fn_low_q, SAMPLE_RATE)
        _, mag_high = freq_response(fn_high_q, SAMPLE_RATE)
        assert mag_high.max() > mag_low.max() + 3.0

    def test_bp_bandwidth_narrows_with_q(self):
        fn_low_q = self._make_filter_fn(1000.0, 1.0, [0, 0, 1])
        fn_high_q = self._make_filter_fn(1000.0, 8.0, [0, 0, 1])
        freqs, mag_low = freq_response(fn_low_q, SAMPLE_RATE)
        _, mag_high = freq_response(fn_high_q, SAMPLE_RATE)

        def _bandwidth(m):
            peak = m.max()
            above = freqs[m > peak - 3.0]
            return above[-1] - above[0] if len(above) > 1 else 0

        bw_low = _bandwidth(mag_low)
        bw_high = _bandwidth(mag_high)
        assert bw_high < bw_low

    def test_time_varying_cutoff_sweep(self):
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.linspace(0.1, 0.9, N_SAMPLES, device=DEVICE).unsqueeze(0)
        q = torch.tensor([0.3], device=DEVICE)
        ft = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        with torch.no_grad():
            filtered = self.filt(noise, cutoff, q, ft)[0].cpu().numpy()
        half = N_SAMPLES // 2
        sc_first = spectral_centroid(filtered[:half], SAMPLE_RATE)
        sc_second = spectral_centroid(filtered[half:], SAMPLE_RATE)
        assert sc_second > sc_first

    def test_extreme_params_stable(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.01, device=DEVICE)
        q = torch.tensor([0.99], device=DEVICE)
        ft = torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE)
        with torch.no_grad():
            out = self.filt(signal, cutoff, q, ft)
        assert not torch.isnan(out).any()
        assert out.abs().max().item() < 100.0


class TestFilterEnvelopeInteraction:
    def setup_method(self):
        from loom.synth import SubtractiveSynth
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def _base_params(self):
        return {
            "osc_pitch": torch.tensor([0.4], device=DEVICE),
            "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=DEVICE),
            "osc_detune": torch.full((1,), 0.5, device=DEVICE),
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "wt_position": torch.full((1,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((1,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((1,), 0.0, device=DEVICE),
            "lfo_rate": torch.full((1,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((1,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            "lfo_target": torch.zeros(1, 4, device=DEVICE),
            "lfo_phase": torch.full((1,), 0.0, device=DEVICE),
            "amp_attack": torch.full((1,), 0.1, device=DEVICE),
            "amp_decay": torch.full((1,), 0.3, device=DEVICE),
            "amp_sustain": torch.full((1,), 0.9, device=DEVICE),
            "amp_release": torch.full((1,), 0.3, device=DEVICE),
            "filter_cutoff": torch.full((1,), 0.4, device=DEVICE),
            "filter_q": torch.full((1,), 0.3, device=DEVICE),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "filt_env_attack": torch.full((1,), 0.2, device=DEVICE),
            "filt_env_decay": torch.full((1,), 0.4, device=DEVICE),
            "filt_env_sustain": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_release": torch.full((1,), 0.3, device=DEVICE),
            "filt_env_amount": torch.full((1,), 0.5, device=DEVICE),
            "dist_amount": torch.full((1,), 0.0, device=DEVICE),
            "dist_mix": torch.full((1,), 0.0, device=DEVICE),
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

    def test_positive_env_opens_filter(self):
        params = self._base_params()
        params["filt_env_amount"] = torch.tensor([0.8], device=DEVICE)
        with torch.no_grad():
            audio = self.synth(params)[0].cpu().numpy()
        attack_sc = spectral_centroid(audio[:N_SAMPLES // 4], SAMPLE_RATE)
        sustain_sc = spectral_centroid(audio[N_SAMPLES // 3:2 * N_SAMPLES // 3], SAMPLE_RATE)
        assert attack_sc > sustain_sc, (
            f"Positive env: attack SC {attack_sc:.0f} should > sustain SC {sustain_sc:.0f}"
        )

    def test_negative_env_closes_filter(self):
        params = self._base_params()
        params["filt_env_amount"] = torch.tensor([0.2], device=DEVICE)
        with torch.no_grad():
            audio = self.synth(params)[0].cpu().numpy()
        attack_sc = spectral_centroid(audio[:N_SAMPLES // 4], SAMPLE_RATE)
        sustain_sc = spectral_centroid(audio[N_SAMPLES // 3:2 * N_SAMPLES // 3], SAMPLE_RATE)
        assert attack_sc < sustain_sc, (
            f"Negative env: attack SC {attack_sc:.0f} should < sustain SC {sustain_sc:.0f}"
        )
```

- [ ] **Step 2: Run all filter tests**

Run: `uv run pytest tests/test_timbres_filter.py -v`
Expected: All tests PASS (or failures reveal engine bugs).

- [ ] **Step 3: Commit**

```bash
git add tests/test_timbres_filter.py
git commit -m "test: add SVFilter frequency response, denorm, and envelope interaction tests"
```

---

### Task 6: Effects Chain Tests — `test_timbres_effects.py`

**Files:**
- Create: `tests/test_timbres_effects.py`

- [ ] **Step 1: Write Distortion and Compressor tests**

```python
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
```

- [ ] **Step 2: Write Chorus, Delay, Reverb, and EQ tests**

Append to `tests/test_timbres_effects.py`:

```python
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
            tail_rms = np.sqrt(np.mean(audio[int(SAMPLE_RATE * 1.5):] ** 2))
            tails.append(tail_rms)
        assert tails[1] > tails[0]

    def test_damping_reduces_high_freq(self):
        signal = self._impulse()
        centroids = []
        for damp in [0.2, 0.8]:
            out = self.reverb(
                signal,
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
                torch.tensor([damp], device=DEVICE),
                torch.tensor([0.5], device=DEVICE),
            )
            tail = out[0].detach().cpu().numpy()[int(SAMPLE_RATE * 0.5):]
            centroids.append(spectral_centroid(tail, SAMPLE_RATE))
        assert centroids[1] < centroids[0]


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
```

- [ ] **Step 3: Run all effects tests**

Run: `uv run pytest tests/test_timbres_effects.py -v`
Expected: All tests PASS (or failures reveal engine bugs).

- [ ] **Step 4: Commit**

```bash
git add tests/test_timbres_effects.py
git commit -m "test: add effects chain acoustic tests — distortion, compressor, chorus, delay, reverb, EQ"
```

---

### Task 7: Preset Test Infrastructure — `test_timbres_presets.py`

**Files:**
- Create: `tests/test_timbres_presets.py`

- [ ] **Step 1: Write golden snapshot infrastructure and preset template**

```python
# tests/test_timbres_presets.py
import torch
import numpy as np
import os
import pytest
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE, DEVICE
from tests.conftest import GOLDEN_DIR, REFERENCE_DIR

N_SAMPLES = SAMPLE_RATE * 4


class PresetTestBase:
    """Base class for preset golden + acoustic tests.

    Subclass and set:
        PRESET_NAME = "01_sub_bass"
        PARAMS = {...}  # full synth param dict

    Then add acoustic assertion methods.
    """

    PRESET_NAME = None
    PARAMS = None

    @pytest.fixture(autouse=True)
    def _setup(self, update_golden):
        self.synth = SubtractiveSynth(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)
        self.update_golden = update_golden
        self._audio = None

    @property
    def audio(self):
        if self._audio is None:
            params = {k: v.to(DEVICE) for k, v in self.PARAMS.items()}
            with torch.no_grad():
                self._audio = self.synth(params)
        return self._audio

    @property
    def audio_np(self):
        return self.audio[0].cpu().numpy()

    def _golden_path(self):
        return os.path.join(GOLDEN_DIR, f"{self.PRESET_NAME}.pt")

    def test_golden_snapshot(self):
        golden_path = self._golden_path()
        if self.update_golden or not os.path.exists(golden_path):
            torch.save(self.audio.cpu(), golden_path)
            pytest.skip(f"Golden updated: {golden_path}")
        golden = torch.load(golden_path, weights_only=True).to(DEVICE)
        assert torch.allclose(self.audio, golden, atol=1e-5), (
            f"Golden mismatch for {self.PRESET_NAME}. "
            f"Max diff: {(self.audio - golden).abs().max().item():.6f}. "
            f"Run with --update-golden to regenerate."
        )

    def _reference_path(self):
        return os.path.join(REFERENCE_DIR, f"serum_{self.PRESET_NAME}.wav")

    @pytest.mark.reference
    def test_reference_comparison(self):
        ref_path = self._reference_path()
        if not os.path.exists(ref_path):
            pytest.skip(f"Reference not found: {ref_path}")
        from scipy.io import wavfile
        from tests.timbre_helpers import mel_spectrogram_distance
        sr, ref_audio = wavfile.read(ref_path)
        ref_audio = ref_audio.astype(np.float32) / 32768.0
        if sr != SAMPLE_RATE:
            pytest.skip(f"Sample rate mismatch: {sr} != {SAMPLE_RATE}")
        distance = mel_spectrogram_distance(self.audio_np, ref_audio, SAMPLE_RATE)
        assert distance < 0.2, (
            f"Mel-spec distance to Serum reference: {distance:.4f} (threshold: 0.2)"
        )


# --- Preset tests are added here as they are onboarded ---
# Example (uncomment and fill in when sub_bass preset is validated):
#
# class TestSubBass(PresetTestBase):
#     PRESET_NAME = "01_sub_bass"
#     PARAMS = { ... }  # full param dict
#
#     def test_fundamental_below_80hz(self):
#         from tests.timbre_helpers import fundamental_freq
#         f0 = fundamental_freq(self.audio_np, SAMPLE_RATE)
#         assert f0 < 80.0
#
#     def test_spectral_centroid_below_200hz(self):
#         from tests.timbre_helpers import spectral_centroid
#         sc = spectral_centroid(self.audio_np, SAMPLE_RATE)
#         assert sc < 200.0
```

- [ ] **Step 2: Run to verify infrastructure works**

Run: `uv run pytest tests/test_timbres_presets.py -v`
Expected: 0 tests collected (no concrete preset classes yet). No errors.

- [ ] **Step 3: Commit**

```bash
git add tests/test_timbres_presets.py
git commit -m "test: add preset golden/reference test infrastructure — ready for preset onboarding"
```

- [ ] **Step 4: Run full test suite to verify no regressions**

Run: `uv run pytest -v`
Expected: All existing tests + all new timbre tests PASS.

- [ ] **Step 5: Final commit (if any fixes were needed)**

```bash
git add -A
git commit -m "test: timbre test framework complete — helpers, osc, envelope, filter, effects, preset infra"
```
