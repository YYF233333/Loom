# Differentiable Synth Engine — Phase 0 最小切片实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 搭建纯 PyTorch 可微分减法合成引擎，跑通"参数 → 音频 → 梯度回传 → 参数恢复"全链路。

**Architecture:** 每个 DSP 模块是一个 `SynthModule(nn.Module)`，信号流为 Oscillator → ADSR → Filter → VCA → Distortion → Out。振荡器用 additive harmonic synthesis（天然带限），滤波器用 `torchaudio.functional.lfilter`（可微分 IIR），ADSR 用乘法分解（无 hard gate）。所有参数归一化到 [0,1]，模块内反归一化。

**Tech Stack:** Python 3.11+, PyTorch 2.x, torchaudio, pytest, uv

**Spec:** `docs/superpowers/specs/2026-06-15-differentiable-synth-engine-design.md`

---

## File Structure

```
loom/
├── pyproject.toml
├── src/
│   └── loom/
│       ├── __init__.py
│       ├── core.py             # SAMPLE_RATE, DURATION, N_SAMPLES 常量 + SynthModule 基类
│       ├── oscillators.py      # AdditiveOscillator: 谐波求和生成带限波形
│       ├── envelope.py         # ADSR: 乘法分解包络
│       ├── filters.py          # BiquadFilter: torchaudio.lfilter + cookbook 系数
│       ├── amplifier.py        # VCA: 信号 × 包络
│       ├── effects/
│       │   ├── __init__.py
│       │   └── distortion.py   # Distortion: tanh waveshaper + dry/wet
│       ├── synth.py            # SubtractiveSynth: 组装所有模块
│       └── render.py           # render(): 参数字典 → 音频 tensor
├── tests/
│   ├── test_oscillators.py
│   ├── test_envelope.py
│   ├── test_filters.py
│   ├── test_amplifier.py
│   ├── test_distortion.py
│   ├── test_synth.py
│   └── test_gradients.py
└── scripts/
    └── param_recovery.py
```

---

## Task 1: 项目脚手架

**Files:**
- Create: `pyproject.toml`
- Create: `src/loom/__init__.py`
- Create: `src/loom/core.py`
- Create: `src/loom/effects/__init__.py`

- [ ] **Step 1: 创建 pyproject.toml**

```toml
[project]
name = "loom"
version = "0.1.0"
description = "Differentiable synthesis engine for audio reverse engineering"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.0",
    "torchaudio>=2.0",
    "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/loom"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: 创建 core.py**

```python
import torch
import torch.nn as nn

SAMPLE_RATE = 44100
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)


class SynthModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
```

- [ ] **Step 3: 创建 __init__.py 文件**

`src/loom/__init__.py`:
```python
from loom.core import SAMPLE_RATE, DURATION, N_SAMPLES
```

`src/loom/effects/__init__.py`:
```python
```

- [ ] **Step 4: 安装项目并验证**

```bash
uv sync --dev
uv run python -c "from loom import SAMPLE_RATE; print(SAMPLE_RATE)"
```

Expected: `44100`

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml src/
git commit -m "feat: project scaffolding with core constants and SynthModule base"
```

---

## Task 2: Oscillator — additive harmonic synthesis

**Files:**
- Create: `src/loom/oscillators.py`
- Create: `tests/test_oscillators.py`

振荡器用谐波求和实现带限波形。4 种波形（saw/square/sine/tri）各自有确定的谐波振幅系数，通过加权混合实现连续可微的波形选择。

- [ ] **Step 1: 写失败测试**

`tests/test_oscillators.py`:
```python
import torch
import pytest
from loom.oscillators import AdditiveOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestAdditiveOscillator:
    def setup_method(self):
        self.osc = AdditiveOscillator(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        batch = 4
        pitch = torch.full((batch,), 0.5)  # normalized pitch
        waveform = torch.zeros(batch, 4)
        waveform[:, 0] = 1.0  # pure sine
        audio = self.osc(pitch, waveform)
        assert audio.shape == (batch, N_SAMPLES)

    def test_sine_frequency(self):
        """Pure sine at A4 (440Hz) should have dominant FFT peak at 440Hz."""
        batch = 1
        midi_note = 69  # A4
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)])  # normalize to [0,1]
        waveform = torch.zeros(batch, 4)
        waveform[:, 0] = 1.0  # sine
        audio = self.osc(pitch, waveform)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]  # skip DC
        assert abs(peak_freq.item() - 440.0) < 2.0  # within 2Hz tolerance

    def test_amplitude_range(self):
        """Output should be roughly in [-1, 1]."""
        pitch = torch.tensor([0.5])
        waveform = torch.zeros(1, 4)
        waveform[:, 0] = 1.0
        audio = self.osc(pitch, waveform)
        assert audio.abs().max().item() <= 1.01

    def test_saw_has_harmonics(self):
        """Saw wave should have energy beyond the fundamental."""
        pitch = torch.tensor([0.3])
        waveform = torch.zeros(1, 4)
        waveform[:, 1] = 1.0  # saw
        audio = self.osc(pitch, waveform)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        fundamental_idx = torch.argmax(magnitudes[1:]) + 1
        harmonic_energy = magnitudes[fundamental_idx * 2:].sum()
        assert harmonic_energy.item() > 0.1  # has significant harmonic content

    def test_detune(self):
        """Detuning should shift the peak frequency."""
        midi_note = 69
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)])
        waveform = torch.zeros(1, 4)
        waveform[:, 0] = 1.0
        detune = torch.tensor([0.7])  # 0.5 = no detune, 0.7 = +40 cents

        audio_detuned = self.osc(pitch, waveform, detune)
        fft = torch.fft.rfft(audio_detuned[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert peak_freq.item() > 440.0  # detuned up
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_oscillators.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.oscillators'`

- [ ] **Step 3: 实现 AdditiveOscillator**

`src/loom/oscillators.py`:
```python
import torch
import torch.nn as nn
import math


class AdditiveOscillator(nn.Module):
    """Bandlimited oscillator via additive harmonic synthesis.

    Waveforms are weighted sums of sinusoidal harmonics. The 4 waveform types
    (sine, saw, square, triangle) are blended via a continuous weight vector,
    making waveform selection differentiable.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
        max_harmonics: Maximum number of harmonics to sum.
    """

    MIDI_MIN = 24
    MIDI_MAX = 96

    def __init__(self, sample_rate: int, n_samples: int, max_harmonics: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.max_harmonics = max_harmonics
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0  # [0,1] -> [-100, +100] cents

    def _harmonic_amplitudes(self, n_harmonics: int, device: torch.device):
        """Compute per-harmonic amplitudes for each waveform type.

        Returns: (4, n_harmonics) tensor — rows are [sine, saw, square, tri].
        """
        n = torch.arange(1, n_harmonics + 1, dtype=torch.float32, device=device)

        sine = torch.zeros(n_harmonics, device=device)
        sine[0] = 1.0

        saw = 1.0 / n
        saw = saw * (2.0 / math.pi)  # normalize

        square = torch.where(n % 2 == 1, 1.0 / n, torch.zeros_like(n))
        square = square * (4.0 / math.pi)

        tri = torch.where(
            n % 2 == 1,
            ((-1.0) ** ((n - 1) / 2.0)) / (n * n),
            torch.zeros_like(n),
        )
        tri = tri * (8.0 / (math.pi**2))

        return torch.stack([sine, saw, square, tri], dim=0)

    def forward(
        self,
        pitch: torch.Tensor,
        waveform: torch.Tensor,
        detune: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render audio from oscillator parameters.

        Args:
            pitch: (batch,) normalized pitch [0,1] -> MIDI [24,96].
            waveform: (batch, 4) waveform blend weights (sine, saw, square, tri).
            detune: (batch,) optional normalized detune [0,1] -> [-100, +100] cents.

        Returns:
            (batch, n_samples) audio tensor in roughly [-1, 1].
        """
        batch = pitch.shape[0]
        f0 = self._denorm_pitch(pitch)  # (batch,)

        if detune is not None:
            cents = self._denorm_detune(detune)
            f0 = f0 * torch.pow(2.0, cents / 1200.0)

        nyquist = self.sample_rate / 2.0
        max_h = torch.clamp(
            torch.floor(nyquist / f0).long(), min=1, max=self.max_harmonics
        )  # (batch,)
        n_h = max_h.max().item()

        harm_amps = self._harmonic_amplitudes(n_h, pitch.device)  # (4, n_h)
        blended = torch.einsum("bw,wh->bh", waveform, harm_amps)  # (batch, n_h)

        harmonic_n = torch.arange(1, n_h + 1, device=pitch.device).float()  # (n_h,)
        mask = harmonic_n.unsqueeze(0) <= max_h.unsqueeze(1)  # (batch, n_h)
        blended = blended * mask.float()

        freqs = f0.unsqueeze(1) * harmonic_n.unsqueeze(0)  # (batch, n_h)
        phases = (
            2.0 * math.pi * freqs.unsqueeze(2) * self.t.unsqueeze(0).unsqueeze(0)
        )  # (batch, n_h, n_samples)
        harmonics = torch.sin(phases)  # (batch, n_h, n_samples)

        audio = torch.einsum("bh,bht->bt", blended, harmonics)  # (batch, n_samples)
        return audio
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_oscillators.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/oscillators.py tests/test_oscillators.py
git commit -m "feat: additive harmonic oscillator with bandlimited waveforms"
```

---

## Task 3: ADSR Envelope

**Files:**
- Create: `src/loom/envelope.py`
- Create: `tests/test_envelope.py`

使用 torchsynth 的乘法分解方案：`envelope = attack_ramp × decay_ramp × release_ramp`。每个 ramp 用 `torch.pow(ramp, alpha)` 控制曲率，alpha=1 为线性。

- [ ] **Step 1: 写失败测试**

`tests/test_envelope.py`:
```python
import torch
import pytest
from loom.envelope import ADSR
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestADSR:
    def setup_method(self):
        self.adsr = ADSR(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        batch = 4
        attack = torch.full((batch,), 0.3)
        decay = torch.full((batch,), 0.3)
        sustain = torch.full((batch,), 0.7)
        release = torch.full((batch,), 0.3)
        env = self.adsr(attack, decay, sustain, release)
        assert env.shape == (batch, N_SAMPLES)

    def test_range(self):
        """Envelope should be in [0, 1]."""
        attack = torch.tensor([0.3])
        decay = torch.tensor([0.3])
        sustain = torch.tensor([0.7])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        assert env.min().item() >= -0.01
        assert env.max().item() <= 1.01

    def test_peak_at_attack_end(self):
        """Envelope should reach ~1.0 at the end of the attack phase."""
        attack = torch.tensor([0.3])  # -> ~0.19s
        decay = torch.tensor([0.5])
        sustain = torch.tensor([0.5])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        # Denorm attack: exp(0.3 * (ln(2000) - ln(1)) + ln(1)) / 1000 ≈ 0.009s
        # The peak should be close to 1.0 somewhere in the envelope
        assert env.max().item() > 0.95

    def test_sustain_level(self):
        """With long sustain, envelope should settle near sustain level."""
        attack = torch.tensor([0.1])  # very short attack
        decay = torch.tensor([0.2])   # short decay
        sustain = torch.tensor([0.6])  # sustain level 0.6
        release = torch.tensor([0.1])
        env = self.adsr(attack, decay, sustain, release)
        # Middle of the envelope should be near sustain level
        mid = env[0, N_SAMPLES // 2].item()
        assert abs(mid - 0.6) < 0.15

    def test_zero_attack(self):
        """Zero attack should not produce NaN."""
        attack = torch.tensor([0.0])
        decay = torch.tensor([0.3])
        sustain = torch.tensor([0.5])
        release = torch.tensor([0.3])
        env = self.adsr(attack, decay, sustain, release)
        assert not torch.isnan(env).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_envelope.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.envelope'`

- [ ] **Step 3: 实现 ADSR**

`src/loom/envelope.py`:
```python
import torch
import torch.nn as nn
import math


class ADSR(nn.Module):
    """Differentiable ADSR envelope using multiplicative ramp decomposition.

    Each ADSR stage is a monotonic ramp computed over the full time axis.
    The final envelope is their product, avoiding hard if/else transitions.

    All time parameters are normalized [0,1] and mapped to physical
    durations via log scale. Sustain is linear [0,1].

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
        note_on_duration: Duration in seconds the note is held before release.
            Defaults to 3.0 (release starts at t=3.0 for a 4s buffer).
    """

    MIN_MS = 1.0
    MAX_ATTACK_MS = 2000.0
    MAX_DECAY_MS = 2000.0
    MAX_RELEASE_MS = 4000.0

    def __init__(
        self, sample_rate: int, n_samples: int, note_on_duration: float = 3.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.note_on_duration = note_on_duration
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _denorm_time(
        self, normalized: torch.Tensor, max_ms: float
    ) -> torch.Tensor:
        """[0,1] -> seconds via log scale."""
        log_min = math.log(self.MIN_MS)
        log_max = math.log(max_ms)
        ms = torch.exp(normalized * (log_max - log_min) + log_min)
        return ms / 1000.0

    def forward(
        self,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
    ) -> torch.Tensor:
        """Generate ADSR envelope.

        Args:
            attack: (batch,) normalized attack time [0,1].
            decay: (batch,) normalized decay time [0,1].
            sustain: (batch,) sustain level [0,1].
            release: (batch,) normalized release time [0,1].

        Returns:
            (batch, n_samples) envelope in [0, 1].
        """
        a_sec = self._denorm_time(attack, self.MAX_ATTACK_MS)   # (batch,)
        d_sec = self._denorm_time(decay, self.MAX_DECAY_MS)     # (batch,)
        r_sec = self._denorm_time(release, self.MAX_RELEASE_MS) # (batch,)
        s_level = sustain  # (batch,)

        t = self.t.unsqueeze(0)  # (1, n_samples)

        # Attack ramp: 0 -> 1 over a_sec
        a_sec_safe = a_sec.unsqueeze(1).clamp(min=1e-6)
        attack_ramp = (t / a_sec_safe).clamp(0.0, 1.0)

        # Decay ramp: 1 -> sustain over d_sec, starting at a_sec
        d_sec_safe = d_sec.unsqueeze(1).clamp(min=1e-6)
        a_sec_expanded = a_sec.unsqueeze(1)
        decay_progress = ((t - a_sec_expanded) / d_sec_safe).clamp(0.0, 1.0)
        s_expanded = s_level.unsqueeze(1)
        decay_ramp = 1.0 - (1.0 - s_expanded) * decay_progress

        # Release ramp: sustain -> 0 over r_sec, starting at note_on_duration
        r_sec_safe = r_sec.unsqueeze(1).clamp(min=1e-6)
        release_progress = (
            (t - self.note_on_duration) / r_sec_safe
        ).clamp(0.0, 1.0)
        release_ramp = 1.0 - release_progress

        envelope = attack_ramp * decay_ramp * release_ramp
        return envelope
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_envelope.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/envelope.py tests/test_envelope.py
git commit -m "feat: differentiable ADSR envelope with multiplicative ramp decomposition"
```

---

## Task 4: Biquad Filter

**Files:**
- Create: `src/loom/filters.py`
- Create: `tests/test_filters.py`

使用 `torchaudio.functional.lfilter` 做可微分 IIR 滤波。系数用 Audio EQ Cookbook 公式从 (cutoff, Q, type) 计算——全部是可微分的 torch 操作。

- [ ] **Step 1: 写失败测试**

`tests/test_filters.py`:
```python
import torch
import pytest
from loom.filters import BiquadFilter
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestBiquadFilter:
    def setup_method(self):
        self.filt = BiquadFilter(sample_rate=SAMPLE_RATE)

    def test_output_shape(self):
        batch = 4
        signal = torch.randn(batch, N_SAMPLES)
        cutoff = torch.full((batch,), 0.5)
        q = torch.full((batch,), 0.5)
        filter_type = torch.zeros(batch, 3)
        filter_type[:, 0] = 1.0  # LP
        out = self.filt(signal, cutoff, q, filter_type)
        assert out.shape == (batch, N_SAMPLES)

    def test_lowpass_attenuates_highs(self):
        """LP filter at 1kHz should attenuate energy above 1kHz."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES)
        cutoff = torch.tensor([0.3])  # ~300Hz region
        q = torch.tensor([0.5])
        filter_type = torch.zeros(1, 3)
        filter_type[:, 0] = 1.0  # LP

        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))

        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        high_mask = freqs > 5000
        high_energy_orig = fft_orig[high_mask].mean()
        high_energy_filt = fft_filt[high_mask].mean()
        assert high_energy_filt < high_energy_orig * 0.5

    def test_highpass_attenuates_lows(self):
        """HP filter should attenuate energy below cutoff."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES)
        cutoff = torch.tensor([0.7])  # high cutoff
        q = torch.tensor([0.5])
        filter_type = torch.zeros(1, 3)
        filter_type[:, 1] = 1.0  # HP

        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))

        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        low_mask = freqs < 500
        low_energy_orig = fft_orig[low_mask].mean()
        low_energy_filt = fft_filt[low_mask].mean()
        assert low_energy_filt < low_energy_orig * 0.5

    def test_no_nan(self):
        """Should not produce NaN for extreme parameters."""
        signal = torch.randn(1, N_SAMPLES)
        cutoff = torch.tensor([0.01])
        q = torch.tensor([0.99])
        filter_type = torch.zeros(1, 3)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_filters.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.filters'`

- [ ] **Step 3: 实现 BiquadFilter**

`src/loom/filters.py`:
```python
import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class BiquadFilter(nn.Module):
    """Differentiable biquad filter using torchaudio.lfilter.

    Supports LP, HP, BP filter types via continuous blending.
    Coefficients computed from Audio EQ Cookbook formulas.

    Args:
        sample_rate: Audio sample rate in Hz.
    """

    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    MIN_Q = 0.5
    MAX_Q = 20.0

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def _denorm_cutoff(self, cutoff: torch.Tensor) -> torch.Tensor:
        """[0,1] -> Hz via log scale."""
        log_min = math.log(self.MIN_HZ)
        log_max = math.log(self.MAX_HZ)
        return torch.exp(cutoff * (log_max - log_min) + log_min)

    def _denorm_q(self, q: torch.Tensor) -> torch.Tensor:
        """[0,1] -> Q via log scale."""
        log_min = math.log(self.MIN_Q)
        log_max = math.log(self.MAX_Q)
        return torch.exp(q * (log_max - log_min) + log_min)

    def _compute_coeffs(
        self, cutoff_hz: torch.Tensor, q: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute biquad coefficients for LP, HP, BP (Audio EQ Cookbook).

        Returns dict with keys 'lp', 'hp', 'bp', each containing
        (b0, b1, b2, a0, a1, a2) as a (batch, 6) tensor.
        """
        w0 = 2.0 * math.pi * cutoff_hz / self.sample_rate
        alpha = torch.sin(w0) / (2.0 * q)
        cos_w0 = torch.cos(w0)

        # LP
        lp_b0 = (1.0 - cos_w0) / 2.0
        lp_b1 = 1.0 - cos_w0
        lp_b2 = (1.0 - cos_w0) / 2.0

        # HP
        hp_b0 = (1.0 + cos_w0) / 2.0
        hp_b1 = -(1.0 + cos_w0)
        hp_b2 = (1.0 + cos_w0) / 2.0

        # BP (constant skirt gain)
        bp_b0 = alpha
        bp_b1 = torch.zeros_like(alpha)
        bp_b2 = -alpha

        # Common denominator
        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        return {
            "lp": torch.stack([lp_b0, lp_b1, lp_b2, a0, a1, a2], dim=-1),
            "hp": torch.stack([hp_b0, hp_b1, hp_b2, a0, a1, a2], dim=-1),
            "bp": torch.stack([bp_b0, bp_b1, bp_b2, a0, a1, a2], dim=-1),
        }

    def forward(
        self,
        signal: torch.Tensor,
        cutoff: torch.Tensor,
        q: torch.Tensor,
        filter_type: torch.Tensor,
    ) -> torch.Tensor:
        """Apply biquad filter.

        Args:
            signal: (batch, n_samples) input audio.
            cutoff: (batch,) normalized cutoff [0,1].
            q: (batch,) normalized Q [0,1].
            filter_type: (batch, 3) blend weights for [LP, HP, BP].

        Returns:
            (batch, n_samples) filtered audio.
        """
        cutoff_hz = self._denorm_cutoff(cutoff)
        q_val = self._denorm_q(q)
        all_coeffs = self._compute_coeffs(cutoff_hz, q_val)

        results = []
        for i in range(signal.shape[0]):
            sample_out = torch.zeros_like(signal[i])
            for j, key in enumerate(["lp", "hp", "bp"]):
                coeffs = all_coeffs[key][i]
                b = coeffs[:3] / coeffs[3]  # normalize by a0
                a = torch.cat(
                    [torch.ones(1, device=signal.device), coeffs[4:6] / coeffs[3]]
                )
                filtered = AF.lfilter(signal[i].unsqueeze(0), a, b, clamp=False)
                sample_out = sample_out + filter_type[i, j] * filtered.squeeze(0)
            results.append(sample_out)

        return torch.stack(results, dim=0)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_filters.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/filters.py tests/test_filters.py
git commit -m "feat: differentiable biquad filter with LP/HP/BP via torchaudio.lfilter"
```

---

## Task 5: VCA (Amplifier)

**Files:**
- Create: `src/loom/amplifier.py`
- Create: `tests/test_amplifier.py`

VCA 是最简单的模块：信号 × 包络 × 增益。

- [ ] **Step 1: 写失败测试**

`tests/test_amplifier.py`:
```python
import torch
import pytest
from loom.amplifier import VCA


class TestVCA:
    def setup_method(self):
        self.vca = VCA()

    def test_output_shape(self):
        signal = torch.randn(4, 1000)
        envelope = torch.ones(4, 1000)
        gain = torch.full((4,), 0.5)
        out = self.vca(signal, envelope, gain)
        assert out.shape == (4, 1000)

    def test_zero_gain_is_silence(self):
        signal = torch.randn(1, 1000)
        envelope = torch.ones(1, 1000)
        gain = torch.tensor([0.0])  # -60dB -> ~0.001, but 0.0 normalized
        out = self.vca(signal, envelope, gain)
        assert out.abs().max().item() < 0.01

    def test_envelope_shapes_output(self):
        """Applying a half-amplitude envelope should halve the signal."""
        signal = torch.ones(1, 1000)
        envelope = torch.full((1, 1000), 0.5)
        gain = torch.tensor([1.0])  # 0dB
        out = self.vca(signal, envelope, gain)
        assert torch.allclose(out, torch.full_like(out, 0.5), atol=0.01)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_amplifier.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.amplifier'`

- [ ] **Step 3: 实现 VCA**

`src/loom/amplifier.py`:
```python
import torch
import torch.nn as nn


class VCA(nn.Module):
    """Voltage-controlled amplifier: signal * envelope * gain.

    Gain is normalized [0,1] mapped to [-60dB, 0dB].
    """

    MIN_DB = -60.0
    MAX_DB = 0.0

    def _denorm_gain(self, gain: torch.Tensor) -> torch.Tensor:
        db = gain * (self.MAX_DB - self.MIN_DB) + self.MIN_DB
        return torch.pow(10.0, db / 20.0)

    def forward(
        self,
        signal: torch.Tensor,
        envelope: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply envelope and gain to signal.

        Args:
            signal: (batch, n_samples) input audio.
            envelope: (batch, n_samples) amplitude envelope [0, 1].
            gain: (batch,) normalized gain [0,1] -> [-60dB, 0dB].

        Returns:
            (batch, n_samples) output audio.
        """
        linear_gain = self._denorm_gain(gain).unsqueeze(1)
        return signal * envelope * linear_gain
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_amplifier.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/amplifier.py tests/test_amplifier.py
git commit -m "feat: VCA with dB-scaled gain"
```

---

## Task 6: Distortion

**Files:**
- Create: `src/loom/effects/distortion.py`
- Create: `tests/test_distortion.py`

tanh waveshaper + dry/wet 混合。amount 控制 pre-gain（drive），mix 控制干湿比。

- [ ] **Step 1: 写失败测试**

`tests/test_distortion.py`:
```python
import torch
import pytest
from loom.effects.distortion import Distortion


class TestDistortion:
    def setup_method(self):
        self.dist = Distortion()

    def test_output_shape(self):
        signal = torch.randn(4, 1000)
        amount = torch.full((4,), 0.5)
        mix = torch.full((4,), 0.5)
        out = self.dist(signal, amount, mix)
        assert out.shape == (4, 1000)

    def test_bypass_when_zero_mix(self):
        """Zero mix should pass signal through unchanged."""
        signal = torch.randn(1, 1000)
        amount = torch.tensor([0.8])
        mix = torch.tensor([0.0])  # fully dry
        out = self.dist(signal, amount, mix)
        assert torch.allclose(out, signal, atol=1e-6)

    def test_adds_harmonics(self):
        """Distortion should add harmonic content."""
        t = torch.linspace(0, 1, 44100).unsqueeze(0)
        signal = torch.sin(2 * 3.14159 * 440 * t)  # pure sine

        amount = torch.tensor([0.9])  # heavy distortion
        mix = torch.tensor([1.0])     # fully wet
        out = self.dist(signal, amount, mix)

        fft_orig = torch.abs(torch.fft.rfft(signal[0]))
        fft_dist = torch.abs(torch.fft.rfft(out[0]))

        fundamental_idx = 440  # approximately
        harmonic_energy_orig = fft_orig[fundamental_idx * 2:].sum()
        harmonic_energy_dist = fft_dist[fundamental_idx * 2:].sum()
        assert harmonic_energy_dist > harmonic_energy_orig * 2

    def test_output_bounded(self):
        """Output should stay in reasonable range due to tanh."""
        signal = torch.randn(1, 1000) * 5.0  # loud input
        amount = torch.tensor([1.0])
        mix = torch.tensor([1.0])
        out = self.dist(signal, amount, mix)
        assert out.abs().max().item() <= 1.01
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_distortion.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.effects.distortion'`

- [ ] **Step 3: 实现 Distortion**

`src/loom/effects/distortion.py`:
```python
import torch
import torch.nn as nn


class Distortion(nn.Module):
    """Tanh waveshaper distortion with dry/wet mix.

    amount controls pre-gain (drive): [0,1] -> [1x, 50x].
    mix controls dry/wet blend: 0 = fully dry, 1 = fully wet.
    """

    MIN_DRIVE = 1.0
    MAX_DRIVE = 50.0

    def _denorm_drive(self, amount: torch.Tensor) -> torch.Tensor:
        return amount * (self.MAX_DRIVE - self.MIN_DRIVE) + self.MIN_DRIVE

    def forward(
        self,
        signal: torch.Tensor,
        amount: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply distortion.

        Args:
            signal: (batch, n_samples) input audio.
            amount: (batch,) normalized drive [0,1].
            mix: (batch,) dry/wet [0,1].

        Returns:
            (batch, n_samples) distorted audio.
        """
        drive = self._denorm_drive(amount).unsqueeze(1)
        mix = mix.unsqueeze(1)
        wet = torch.tanh(signal * drive)
        return (1.0 - mix) * signal + mix * wet
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_distortion.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/distortion.py tests/test_distortion.py
git commit -m "feat: tanh waveshaper distortion with dry/wet mix"
```

---

## Task 7: SubtractiveSynth 组装

**Files:**
- Create: `src/loom/synth.py`
- Create: `tests/test_synth.py`

将所有模块组装成完整的减法合成器。

- [ ] **Step 1: 写失败测试**

`tests/test_synth.py`:
```python
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
        # Set batch item 0 to same as single
        audio_single = self.synth(params_single)
        audio_batch = self.synth(params_batch)
        assert torch.allclose(audio_single[0], audio_batch[0], atol=1e-5)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_synth.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'loom.synth'`

- [ ] **Step 3: 实现 SubtractiveSynth**

`src/loom/synth.py`:
```python
import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.envelope import ADSR
from loom.filters import BiquadFilter
from loom.amplifier import VCA
from loom.effects.distortion import Distortion


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer.

    Signal flow: Oscillator -> Filter (with envelope) -> VCA (with envelope) -> Distortion

    The filter envelope modulates cutoff: effective_cutoff = cutoff + amount * filt_env.
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.amp_envelope = ADSR(sample_rate, n_samples)
        self.filter_envelope = ADSR(sample_rate, n_samples)
        self.filter = BiquadFilter(sample_rate)
        self.vca = VCA()
        self.distortion = Distortion()

    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary.

        Args:
            params: Dict with keys matching the parameter table in the spec.

        Returns:
            (batch, n_samples) audio tensor.
        """
        # Oscillator
        audio = self.oscillator(
            params["osc_pitch"],
            params["osc_waveform"],
            params["osc_detune"],
        )

        # Filter envelope -> modulate cutoff
        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        # filt_env_amount: [0,1] normalized, treat 0.5 as zero modulation
        amount = (params["filt_env_amount"] - 0.5) * 2.0  # -> [-1, 1]
        filt_env_mean = filt_env.mean(dim=1)  # (batch,)
        modulated_cutoff = (
            params["filter_cutoff"] + amount * filt_env_mean * 0.3
        ).clamp(0.0, 1.0)

        # Filter
        audio = self.filter(
            audio, modulated_cutoff, params["filter_q"], params["filter_type"]
        )

        # Amplitude envelope + VCA
        amp_env = self.amp_envelope(
            params["amp_attack"],
            params["amp_decay"],
            params["amp_sustain"],
            params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        # Distortion
        audio = self.distortion(audio, params["dist_amount"], params["dist_mix"])

        return audio
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_synth.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/synth.py tests/test_synth.py
git commit -m "feat: SubtractiveSynth assembling oscillator, filter, VCA, distortion"
```

---

## Task 8: Render 入口

**Files:**
- Create: `src/loom/render.py`

render.py 是薄封装层，方便外部调用。synth.py 已经完成核心工作，render 只提供参数随机采样的工具函数。

- [ ] **Step 1: 实现 render.py**

`src/loom/render.py`:
```python
import torch
from loom.core import SAMPLE_RATE, N_SAMPLES
from loom.synth import SubtractiveSynth


def random_params(batch: int, device: torch.device = torch.device("cpu")) -> dict[str, torch.Tensor]:
    """Sample a random parameter dictionary for SubtractiveSynth."""
    def _rand(shape):
        return torch.rand(shape, device=device)

    def _one_hot_rand(batch: int, n: int):
        idx = torch.randint(0, n, (batch,), device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    return {
        "osc_pitch": _rand((batch,)),
        "osc_waveform": _one_hot_rand(batch, 4),
        "osc_detune": _rand((batch,)),
        "amp_attack": _rand((batch,)),
        "amp_decay": _rand((batch,)),
        "amp_sustain": _rand((batch,)),
        "amp_release": _rand((batch,)),
        "filter_cutoff": _rand((batch,)),
        "filter_q": _rand((batch,)),
        "filter_type": _one_hot_rand(batch, 3),
        "filt_env_attack": _rand((batch,)),
        "filt_env_decay": _rand((batch,)),
        "filt_env_sustain": _rand((batch,)),
        "filt_env_release": _rand((batch,)),
        "filt_env_amount": _rand((batch,)),
        "dist_amount": _rand((batch,)),
        "dist_mix": _rand((batch,)),
        "master_gain": _rand((batch,)),
    }


def render(params: dict[str, torch.Tensor], sample_rate: int = SAMPLE_RATE, n_samples: int = N_SAMPLES) -> torch.Tensor:
    """Render audio from a parameter dictionary.

    Args:
        params: Parameter dictionary (see SubtractiveSynth).
        sample_rate: Sample rate in Hz.
        n_samples: Number of output samples.

    Returns:
        (batch, n_samples) audio tensor.
    """
    synth = SubtractiveSynth(sample_rate, n_samples)
    return synth(params)
```

- [ ] **Step 2: 快速验证**

```bash
uv run python -c "from loom.render import random_params, render; audio = render(random_params(2)); print(audio.shape, audio.abs().max().item())"
```

Expected: `torch.Size([2, 176400])` 加一个非零数值

- [ ] **Step 3: Commit**

```bash
git add src/loom/render.py
git commit -m "feat: render entry point with random parameter sampling"
```

---

## Task 9: 梯度验证与参数恢复测试

**Files:**
- Create: `tests/test_gradients.py`
- Create: `scripts/param_recovery.py`

这是 Phase 0 的 go/no-go 判据。

- [ ] **Step 1: 写梯度测试**

`tests/test_gradients.py`:
```python
import torch
import pytest
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.core import SAMPLE_RATE

SHORT_SAMPLES = 4410  # 0.1s for fast gradcheck


class TestGradients:
    def test_synth_has_gradients(self):
        """All continuous parameters should receive gradients."""
        synth = SubtractiveSynth(SAMPLE_RATE, SHORT_SAMPLES)
        params = random_params(1)

        continuous_keys = [
            "osc_pitch", "osc_detune",
            "amp_attack", "amp_decay", "amp_sustain", "amp_release",
            "filter_cutoff", "filter_q",
            "filt_env_attack", "filt_env_decay", "filt_env_sustain",
            "filt_env_release", "filt_env_amount",
            "dist_amount", "dist_mix", "master_gain",
        ]
        blend_keys = ["osc_waveform", "filter_type"]

        for key in continuous_keys + blend_keys:
            params[key] = params[key].detach().clone().requires_grad_(True)

        audio = synth(params)
        loss = audio.pow(2).mean()
        loss.backward()

        for key in continuous_keys + blend_keys:
            grad = params[key].grad
            assert grad is not None, f"No gradient for {key}"
            assert not torch.isnan(grad).any(), f"NaN gradient for {key}"

    def test_parameter_recovery_converges(self):
        """Gradient descent should recover known parameters from audio."""
        torch.manual_seed(0)
        n_samples = 22050  # 0.5s for tractable test
        synth = SubtractiveSynth(SAMPLE_RATE, n_samples)

        # Target: fixed known parameters
        target_params = {
            "osc_pitch": torch.tensor([0.5]),
            "osc_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "osc_detune": torch.tensor([0.5]),
            "amp_attack": torch.tensor([0.2]),
            "amp_decay": torch.tensor([0.3]),
            "amp_sustain": torch.tensor([0.7]),
            "amp_release": torch.tensor([0.3]),
            "filter_cutoff": torch.tensor([0.6]),
            "filter_q": torch.tensor([0.4]),
            "filter_type": torch.tensor([[1.0, 0.0, 0.0]]),
            "filt_env_attack": torch.tensor([0.2]),
            "filt_env_decay": torch.tensor([0.3]),
            "filt_env_sustain": torch.tensor([0.5]),
            "filt_env_release": torch.tensor([0.3]),
            "filt_env_amount": torch.tensor([0.5]),
            "dist_amount": torch.tensor([0.3]),
            "dist_mix": torch.tensor([0.4]),
            "master_gain": torch.tensor([0.8]),
        }
        with torch.no_grad():
            target_audio = synth(target_params)

        # Prediction: slightly perturbed parameters
        pred_params = {}
        optimize_keys = [
            "osc_pitch", "osc_detune",
            "amp_attack", "amp_decay", "amp_sustain", "amp_release",
            "filter_cutoff", "filter_q",
            "filt_env_attack", "filt_env_decay", "filt_env_sustain",
            "filt_env_release", "filt_env_amount",
            "dist_amount", "dist_mix", "master_gain",
        ]
        for key, val in target_params.items():
            if key in optimize_keys:
                perturbed = (val + torch.randn_like(val) * 0.15).clamp(0.01, 0.99)
                pred_params[key] = perturbed.detach().clone().requires_grad_(True)
            else:
                pred_params[key] = val.clone()

        optimizer = torch.optim.Adam(
            [pred_params[k] for k in optimize_keys], lr=0.01
        )

        # Compute mel spectrogram loss
        n_fft = 1024
        hop = 256
        mel_basis = torch.zeros(1)  # simple STFT loss as proxy

        initial_loss = None
        for step in range(200):
            optimizer.zero_grad()
            clamped = {}
            for key, val in pred_params.items():
                if key in optimize_keys:
                    clamped[key] = val.clamp(0.01, 0.99)
                else:
                    clamped[key] = val
            pred_audio = synth(clamped)

            # Multi-resolution STFT loss
            loss = torch.tensor(0.0)
            for fft_size in [512, 1024, 2048]:
                target_stft = torch.stft(
                    target_audio[0], fft_size,
                    hop_length=fft_size // 4,
                    return_complex=True,
                    window=torch.hann_window(fft_size),
                )
                pred_stft = torch.stft(
                    pred_audio[0], fft_size,
                    hop_length=fft_size // 4,
                    return_complex=True,
                    window=torch.hann_window(fft_size),
                )
                loss = loss + (target_stft.abs() - pred_stft.abs()).pow(2).mean()

            if initial_loss is None:
                initial_loss = loss.item()
            loss.backward()
            optimizer.step()

        final_loss = loss.item()
        assert final_loss < initial_loss * 0.5, (
            f"Loss did not converge: {initial_loss:.4f} -> {final_loss:.4f}"
        )
```

- [ ] **Step 2: 运行测试确认失败然后通过**

```bash
uv run pytest tests/test_gradients.py -v --timeout=120
```

Expected: 两个测试都 PASS。`test_parameter_recovery_converges` 可能需要 30-60 秒。

如果 `test_parameter_recovery_converges` 失败，这是真正的 debug 点——需要检查哪个模块的梯度有问题。

- [ ] **Step 3: 写参数恢复可视化脚本**

`scripts/param_recovery.py`:
```python
"""End-to-end parameter recovery visualization.

Generates a target audio from known parameters, then optimizes
a randomly initialized parameter set to match it via gradient descent.
Outputs a convergence plot and parameter comparison.

Usage:
    uv run python scripts/param_recovery.py
"""

import torch
import matplotlib.pyplot as plt
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.core import SAMPLE_RATE, N_SAMPLES


def main():
    torch.manual_seed(42)
    n_samples = 44100  # 1 second for reasonable speed
    synth = SubtractiveSynth(SAMPLE_RATE, n_samples)

    target_params = random_params(1)
    with torch.no_grad():
        target_audio = synth(target_params)

    optimize_keys = [
        "osc_pitch", "osc_detune",
        "amp_attack", "amp_decay", "amp_sustain", "amp_release",
        "filter_cutoff", "filter_q",
        "filt_env_attack", "filt_env_decay", "filt_env_sustain",
        "filt_env_release", "filt_env_amount",
        "dist_amount", "dist_mix", "master_gain",
    ]

    pred_params = {}
    for key, val in target_params.items():
        if key in optimize_keys:
            init = torch.rand_like(val).clamp(0.05, 0.95)
            pred_params[key] = init.detach().clone().requires_grad_(True)
        else:
            pred_params[key] = val.clone()

    optimizer = torch.optim.Adam(
        [pred_params[k] for k in optimize_keys], lr=0.01
    )

    losses = []
    n_steps = 500
    for step in range(n_steps):
        optimizer.zero_grad()
        clamped = {}
        for key, val in pred_params.items():
            if key in optimize_keys:
                clamped[key] = val.clamp(0.01, 0.99)
            else:
                clamped[key] = val
        pred_audio = synth(clamped)

        loss = torch.tensor(0.0)
        for fft_size in [512, 1024, 2048]:
            target_stft = torch.stft(
                target_audio[0], fft_size,
                hop_length=fft_size // 4,
                return_complex=True,
                window=torch.hann_window(fft_size),
            )
            pred_stft = torch.stft(
                pred_audio[0], fft_size,
                hop_length=fft_size // 4,
                return_complex=True,
                window=torch.hann_window(fft_size),
            )
            loss = loss + (target_stft.abs() - pred_stft.abs()).pow(2).mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f}")

    # Plot convergence
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Multi-res STFT Loss")
    axes[0].set_title("Convergence")
    axes[0].set_yscale("log")

    # Parameter comparison
    names, targets, preds = [], [], []
    for key in optimize_keys:
        names.append(key.replace("filt_env_", "fe_").replace("amp_", "a_"))
        targets.append(target_params[key].item())
        preds.append(pred_params[key].detach().clamp(0.01, 0.99).item())

    x = range(len(names))
    axes[1].bar([i - 0.15 for i in x], targets, 0.3, label="Target", alpha=0.8)
    axes[1].bar([i + 0.15 for i in x], preds, 0.3, label="Predicted", alpha=0.8)
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    axes[1].set_ylabel("Value [0,1]")
    axes[1].set_title("Parameter Recovery")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("param_recovery.png", dpi=150)
    print(f"\nSaved to param_recovery.png")
    print(f"Final loss: {losses[-1]:.6f} (initial: {losses[0]:.6f})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Commit**

```bash
git add tests/test_gradients.py scripts/param_recovery.py
git commit -m "feat: gradient tests and parameter recovery verification script"
```

---

## Task 10: 集成验证与收尾

- [ ] **Step 1: 运行全量测试**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: 运行参数恢复脚本**

```bash
uv run python scripts/param_recovery.py
```

Expected: loss 收敛（最终 < 初始的 50%），输出 `param_recovery.png`。

- [ ] **Step 3: 检查参数恢复结果**

查看 `param_recovery.png`：
- 左图：loss 曲线应持续下降
- 右图：predicted 柱状图应大致匹配 target

如果收敛不理想，按以下优先级排查：
1. 振荡器频率参数是否有梯度（pitch 的对数映射可能导致梯度消失）
2. 滤波器 cutoff 梯度是否流通（lfilter 的可微分性）
3. loss 函数是否合适（尝试调整 FFT size 或加 mel 加权）

- [ ] **Step 4: 最终 commit**

```bash
git add -A
git commit -m "feat: Phase 0 minimal slice complete — differentiable subtractive synth"
```
