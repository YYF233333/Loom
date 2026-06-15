# Effects Chain Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 SubtractiveSynth 补全完整效果器链：Compressor, Chorus, Delay, Reverb (FDN), EQ。

**Architecture:** 每个效果器是独立的 `nn.Module`，接口统一 `forward(signal, ...params) → audio`。所有参数归一化 [0,1]，模块内反归一化。效果器链在 SubtractiveSynth.forward 中按固定顺序串联：Distortion → Compressor → Chorus → Delay → Reverb → EQ。Reverb 使用频域 FDN（在 FFT bin 上求解传递函数，无时域循环）。

**Tech Stack:** PyTorch 2.x, torchaudio, pytest

**Spec:** `docs/superpowers/specs/2026-06-15-effects-chain-expansion-design.md`

---

## File Structure

```
src/loom/effects/
├── compressor.py      # 新增
├── chorus.py          # 新增
├── delay.py           # 新增
├── reverb.py          # 新增
└── eq.py              # 新增

src/loom/
├── synth.py           # 修改 — 集成新效果器
└── render.py          # 修改 — random_params 新增参数

tests/
├── test_compressor.py # 新增
├── test_chorus.py     # 新增
├── test_delay.py      # 新增
├── test_reverb.py     # 新增
├── test_eq.py         # 新增
└── test_gradients.py  # 修改 — 扩展梯度检查
```

---

## Task 1: Compressor

**Files:**
- Create: `src/loom/effects/compressor.py`
- Create: `tests/test_compressor.py`

Feed-forward compressor：RMS 检测 → gain reduction → makeup → dry/wet。

- [ ] **Step 1: 写失败测试**

`tests/test_compressor.py`:
```python
import torch
import pytest
from loom.effects.compressor import Compressor
from loom.core import N_SAMPLES


class TestCompressor:
    def setup_method(self):
        self.comp = Compressor()

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.full((4,), 0.5),
            ratio=torch.full((4,), 0.5),
            attack=torch.full((4,), 0.5),
            release=torch.full((4,), 0.5),
            makeup=torch.full((4,), 0.0),
            mix=torch.full((4,), 1.0),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.5]),
            ratio=torch.tensor([0.5]),
            attack=torch.tensor([0.5]),
            release=torch.tensor([0.5]),
            makeup=torch.tensor([0.0]),
            mix=torch.tensor([0.0]),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_compresses_loud_signal(self):
        """Loud signal should have lower RMS after compression."""
        signal = torch.randn(1, N_SAMPLES) * 2.0
        out = self.comp(
            signal,
            threshold=torch.tensor([0.3]),  # low threshold -> more compression
            ratio=torch.tensor([0.8]),       # high ratio
            attack=torch.tensor([0.3]),
            release=torch.tensor([0.5]),
            makeup=torch.tensor([0.0]),
            mix=torch.tensor([1.0]),
        )
        assert out.pow(2).mean().sqrt() < signal.pow(2).mean().sqrt()

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.comp(
            signal,
            threshold=torch.tensor([0.01]),
            ratio=torch.tensor([0.99]),
            attack=torch.tensor([0.01]),
            release=torch.tensor([0.99]),
            makeup=torch.tensor([0.99]),
            mix=torch.tensor([1.0]),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_compressor.py -v
```

Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 Compressor**

`src/loom/effects/compressor.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Compressor(nn.Module):
    """Differentiable feed-forward compressor.

    RMS envelope detection -> gain reduction -> smoothing -> makeup gain -> dry/wet.

    All parameters normalized [0,1], denormalized internally.
    """

    THRESH_MIN_DB = -40.0
    THRESH_MAX_DB = 0.0
    RATIO_MIN = 1.0
    RATIO_MAX = 20.0
    ATTACK_MIN_MS = 0.1
    ATTACK_MAX_MS = 100.0
    RELEASE_MIN_MS = 10.0
    RELEASE_MAX_MS = 1000.0
    MAKEUP_MIN_DB = 0.0
    MAKEUP_MAX_DB = 30.0
    RMS_WINDOW = 1024

    def _denorm_threshold(self, t: torch.Tensor) -> torch.Tensor:
        return t * (self.THRESH_MAX_DB - self.THRESH_MIN_DB) + self.THRESH_MIN_DB

    def _denorm_ratio(self, r: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.RATIO_MIN)
        log_max = math.log(self.RATIO_MAX)
        return torch.exp(r * (log_max - log_min) + log_min)

    def _denorm_makeup(self, m: torch.Tensor) -> torch.Tensor:
        db = m * (self.MAKEUP_MAX_DB - self.MAKEUP_MIN_DB) + self.MAKEUP_MIN_DB
        return torch.pow(10.0, db / 20.0)

    def _denorm_time_ms(
        self, normalized: torch.Tensor, min_ms: float, max_ms: float
    ) -> torch.Tensor:
        log_min = math.log(min_ms)
        log_max = math.log(max_ms)
        return torch.exp(normalized * (log_max - log_min) + log_min)

    def _rms_envelope(self, signal: torch.Tensor) -> torch.Tensor:
        x2 = signal.pow(2).unsqueeze(1)  # (batch, 1, n_samples)
        rms = F.avg_pool1d(
            x2, self.RMS_WINDOW, stride=1, padding=self.RMS_WINDOW // 2
        )
        # avg_pool1d may produce slightly different length; trim to match
        rms = rms[:, :, : signal.shape[1]]
        return rms.squeeze(1).sqrt().clamp(min=1e-8)

    def forward(
        self,
        signal: torch.Tensor,
        threshold: torch.Tensor,
        ratio: torch.Tensor,
        attack: torch.Tensor,
        release: torch.Tensor,
        makeup: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply compression.

        Args:
            signal: (batch, n_samples) input audio.
            threshold: (batch,) normalized [0,1] -> [-40dB, 0dB].
            ratio: (batch,) normalized [0,1] -> [1:1, 20:1].
            attack: (batch,) normalized [0,1] -> [0.1ms, 100ms].
            release: (batch,) normalized [0,1] -> [10ms, 1000ms].
            makeup: (batch,) normalized [0,1] -> [0dB, 30dB].
            mix: (batch,) dry/wet [0,1].
        """
        thresh_db = self._denorm_threshold(threshold).unsqueeze(1)
        ratio_val = self._denorm_ratio(ratio).unsqueeze(1)
        makeup_linear = self._denorm_makeup(makeup).unsqueeze(1)
        mix = mix.unsqueeze(1)

        rms = self._rms_envelope(signal)
        rms_db = 20.0 * torch.log10(rms.clamp(min=1e-8))

        gain_db = torch.min(
            torch.zeros_like(rms_db),
            (1.0 - 1.0 / ratio_val) * (thresh_db - rms_db),
        )
        gain = torch.pow(10.0, gain_db / 20.0)

        # Smooth gain with avg_pool1d (approximates attack/release)
        smooth_ms = self._denorm_time_ms(
            (attack + release) / 2.0, self.ATTACK_MIN_MS, self.RELEASE_MAX_MS
        )
        smooth_samples = (smooth_ms / 1000.0 * 44100).long().clamp(min=1)
        # Use a fixed window for batched processing
        window = smooth_samples.max().item()
        if window > 1:
            gain_smooth = F.avg_pool1d(
                gain.unsqueeze(1), window, stride=1, padding=window // 2
            )
            gain_smooth = gain_smooth[:, :, : signal.shape[1]].squeeze(1)
        else:
            gain_smooth = gain

        wet = signal * gain_smooth * makeup_linear
        return (1.0 - mix) * signal + mix * wet
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_compressor.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/compressor.py tests/test_compressor.py
git commit -m "feat: differentiable feed-forward compressor"
```

---

## Task 2: Chorus

**Files:**
- Create: `src/loom/effects/chorus.py`
- Create: `tests/test_chorus.py`

LFO 调制延迟线，用 `grid_sample` 做分数延迟插值。

- [ ] **Step 1: 写失败测试**

`tests/test_chorus.py`:
```python
import torch
import pytest
from loom.effects.chorus import Chorus
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestChorus:
    def setup_method(self):
        self.chorus = Chorus(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.chorus(
            signal,
            rate=torch.full((4,), 0.5),
            depth=torch.full((4,), 0.5),
            mix=torch.full((4,), 0.5),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.chorus(
            signal,
            rate=torch.tensor([0.5]),
            depth=torch.tensor([0.5]),
            mix=torch.tensor([0.0]),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_spectral_spreading(self):
        """Chorus should widen the spectrum around the fundamental."""
        t = torch.arange(N_SAMPLES, dtype=torch.float32) / SAMPLE_RATE
        signal = torch.sin(2 * 3.14159 * 440 * t).unsqueeze(0)

        out = self.chorus(
            signal,
            rate=torch.tensor([0.5]),
            depth=torch.tensor([0.8]),
            mix=torch.tensor([1.0]),
        )

        fft_orig = torch.abs(torch.fft.rfft(signal[0]))
        fft_chorus = torch.abs(torch.fft.rfft(out[0]))
        # Chorus should spread energy to neighboring bins
        peak = torch.argmax(fft_orig[1:]) + 1
        sideband_orig = fft_orig[peak - 20 : peak + 20].sum() - fft_orig[peak]
        sideband_chorus = fft_chorus[peak - 20 : peak + 20].sum() - fft_chorus[peak]
        assert sideband_chorus > sideband_orig

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.chorus(
            signal,
            rate=torch.tensor([0.99]),
            depth=torch.tensor([0.99]),
            mix=torch.tensor([1.0]),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_chorus.py -v
```

- [ ] **Step 3: 实现 Chorus**

`src/loom/effects/chorus.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Chorus(nn.Module):
    """Differentiable chorus effect via LFO-modulated delay line.

    Uses grid_sample for fractional delay interpolation.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
    """

    RATE_MIN_HZ = 0.1
    RATE_MAX_HZ = 5.0
    BASE_DELAY_MS = 7.0
    MAX_DEPTH_MS = 5.0  # ±5ms modulation around base

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _denorm_rate(self, rate: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.RATE_MIN_HZ)
        log_max = math.log(self.RATE_MAX_HZ)
        return torch.exp(rate * (log_max - log_min) + log_min)

    def forward(
        self,
        signal: torch.Tensor,
        rate: torch.Tensor,
        depth: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply chorus.

        Args:
            signal: (batch, n_samples) input audio.
            rate: (batch,) normalized LFO rate [0,1] -> [0.1, 5] Hz.
            depth: (batch,) normalized modulation depth [0,1].
            mix: (batch,) dry/wet [0,1].
        """
        batch = signal.shape[0]
        rate_hz = self._denorm_rate(rate)  # (batch,)
        mix = mix.unsqueeze(1)
        depth = depth.unsqueeze(1)

        # LFO: sine wave modulating delay time
        lfo = torch.sin(
            2.0 * math.pi * rate_hz.unsqueeze(1) * self.t.unsqueeze(0)
        )  # (batch, n_samples)

        # Delay in samples: base_delay ± depth * max_depth
        base_delay_samples = self.BASE_DELAY_MS / 1000.0 * self.sample_rate
        mod_samples = depth * self.MAX_DEPTH_MS / 1000.0 * self.sample_rate
        delay_samples = base_delay_samples + lfo * mod_samples  # (batch, n_samples)

        # Sample positions (where to read from)
        indices = torch.arange(
            self.n_samples, dtype=torch.float32, device=signal.device
        ).unsqueeze(0)
        read_pos = indices - delay_samples  # (batch, n_samples)

        # Normalize to [-1, 1] for grid_sample
        grid = (read_pos / (self.n_samples - 1)) * 2.0 - 1.0
        grid = grid.unsqueeze(1).unsqueeze(3)  # (batch, 1, n_samples, 1)

        signal_4d = signal.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n_samples)
        wet = F.grid_sample(
            signal_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        wet = wet.squeeze(1).squeeze(1)  # (batch, n_samples)

        return (1.0 - mix) * signal + mix * wet
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_chorus.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/chorus.py tests/test_chorus.py
git commit -m "feat: differentiable chorus with LFO-modulated delay line"
```

---

## Task 3: Delay

**Files:**
- Create: `src/loom/effects/delay.py`
- Create: `tests/test_delay.py`

延迟线 + feedback，用 `grid_sample` 做分数延迟，展开 8 次 feedback 迭代。

- [ ] **Step 1: 写失败测试**

`tests/test_delay.py`:
```python
import torch
import pytest
from loom.effects.delay import Delay
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestDelay:
    def setup_method(self):
        self.delay = Delay(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.delay(
            signal,
            time=torch.full((4,), 0.5),
            feedback=torch.full((4,), 0.3),
            mix=torch.full((4,), 0.5),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.delay(
            signal,
            time=torch.tensor([0.5]),
            feedback=torch.tensor([0.3]),
            mix=torch.tensor([0.0]),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_echo_at_delay_time(self):
        """Autocorrelation should have a peak near the delay time."""
        # Create an impulse
        signal = torch.zeros(1, N_SAMPLES)
        signal[0, 1000] = 1.0

        out = self.delay(
            signal,
            time=torch.tensor([0.5]),  # ~100ms delay
            feedback=torch.tensor([0.5]),
            mix=torch.tensor([1.0]),
        )
        # Should have energy after the impulse at the delay offset
        delay_ms = 100.0  # approx for time=0.5 in log scale
        delay_samples_approx = int(delay_ms / 1000.0 * SAMPLE_RATE)
        # Check there's energy in the delayed region
        delayed_region = out[0, 1000 + delay_samples_approx - 500 : 1000 + delay_samples_approx + 500]
        assert delayed_region.abs().max().item() > 0.01

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.delay(
            signal,
            time=torch.tensor([0.99]),
            feedback=torch.tensor([0.89]),
            mix=torch.tensor([1.0]),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_delay.py -v
```

- [ ] **Step 3: 实现 Delay**

`src/loom/effects/delay.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Delay(nn.Module):
    """Differentiable delay line with feedback.

    Uses grid_sample for fractional delay and unrolls feedback 8 times.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
        n_taps: Number of feedback iterations to unroll.
    """

    MIN_MS = 10.0
    MAX_MS = 500.0
    MAX_FEEDBACK = 0.9

    def __init__(self, sample_rate: int, n_samples: int, n_taps: int = 8):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_taps = n_taps

    def _denorm_time(self, time: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_MS)
        log_max = math.log(self.MAX_MS)
        ms = torch.exp(time * (log_max - log_min) + log_min)
        return ms / 1000.0 * self.sample_rate  # -> samples

    def _fractional_delay(
        self, signal: torch.Tensor, delay_samples: torch.Tensor
    ) -> torch.Tensor:
        """Shift signal by delay_samples using grid_sample."""
        batch, n = signal.shape
        indices = torch.arange(n, dtype=torch.float32, device=signal.device)
        read_pos = indices.unsqueeze(0) - delay_samples.unsqueeze(1)  # (batch, n)
        grid = (read_pos / (n - 1)) * 2.0 - 1.0
        grid = grid.unsqueeze(1).unsqueeze(3)  # (batch, 1, n, 1)
        sig_4d = signal.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, n)
        out = F.grid_sample(
            sig_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        return out.squeeze(1).squeeze(1)

    def forward(
        self,
        signal: torch.Tensor,
        time: torch.Tensor,
        feedback: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delay effect.

        Args:
            signal: (batch, n_samples) input audio.
            time: (batch,) normalized delay time [0,1] -> [10ms, 500ms].
            feedback: (batch,) feedback amount [0,1] -> [0, 0.9].
            mix: (batch,) dry/wet [0,1].
        """
        delay_samples = self._denorm_time(time)  # (batch,)
        fb = feedback * self.MAX_FEEDBACK  # (batch,)
        mix = mix.unsqueeze(1)

        wet = torch.zeros_like(signal)
        current = signal
        for i in range(self.n_taps):
            delayed = self._fractional_delay(current, delay_samples * (i + 1))
            wet = wet + (fb.unsqueeze(1) ** i) * delayed

        return (1.0 - mix) * signal + mix * wet
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_delay.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/delay.py tests/test_delay.py
git commit -m "feat: differentiable delay with unrolled feedback taps"
```

---

## Task 4: Reverb (FDN, frequency-domain)

**Files:**
- Create: `src/loom/effects/reverb.py`
- Create: `tests/test_reverb.py`

Feedback Delay Network in frequency domain：compute H(z) at each FFT bin, multiply, IFFT。无时域循环。

- [ ] **Step 1: 写失败测试**

`tests/test_reverb.py`:
```python
import torch
import pytest
from loom.effects.reverb import Reverb
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestReverb:
    def setup_method(self):
        self.reverb = Reverb(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.reverb(
            signal,
            room_size=torch.full((4,), 0.5),
            decay=torch.full((4,), 0.5),
            damping=torch.full((4,), 0.3),
            mix=torch.full((4,), 0.5),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_bypass_when_zero_mix(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.reverb(
            signal,
            room_size=torch.tensor([0.5]),
            decay=torch.tensor([0.5]),
            damping=torch.tensor([0.3]),
            mix=torch.tensor([0.0]),
        )
        assert torch.allclose(out, signal, atol=1e-6)

    def test_reverb_tail(self):
        """Reverb output should have energy in the tail that input doesn't."""
        signal = torch.zeros(1, N_SAMPLES)
        signal[0, :4410] = torch.randn(4410)  # 0.1s burst then silence

        out = self.reverb(
            signal,
            room_size=torch.tensor([0.5]),
            decay=torch.tensor([0.7]),
            damping=torch.tensor([0.3]),
            mix=torch.tensor([1.0]),
        )
        tail_energy = out[0, N_SAMPLES // 2 :].pow(2).mean()
        assert tail_energy.item() > 1e-6

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.reverb(
            signal,
            room_size=torch.tensor([0.99]),
            decay=torch.tensor([0.99]),
            damping=torch.tensor([0.99]),
            mix=torch.tensor([1.0]),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_reverb.py -v
```

- [ ] **Step 3: 实现 Reverb**

`src/loom/effects/reverb.py`:
```python
import torch
import torch.nn as nn
import math


class Reverb(nn.Module):
    """Differentiable FDN reverb computed in the frequency domain.

    Evaluates the FDN transfer function H(z) at each FFT bin and
    multiplies with the input spectrum. No time-domain loops.

    4 delay lines with mutually coprime lengths, Householder feedback matrix.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
    """

    BASE_DELAYS = [1433, 1601, 1867, 2053]  # mutually coprime
    ROOM_SCALE_MIN = 0.5
    ROOM_SCALE_MAX = 2.0
    MAX_DECAY = 0.95

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_delays = len(self.BASE_DELAYS)

        # Householder matrix: H = I - 2vvT/(vTv), v = ones(4)
        v = torch.ones(self.n_delays)
        H = torch.eye(self.n_delays) - 2.0 * torch.outer(v, v) / torch.dot(v, v)
        self.register_buffer("feedback_matrix", H)

        base = torch.tensor(self.BASE_DELAYS, dtype=torch.float32)
        self.register_buffer("base_delays", base)

    def forward(
        self,
        signal: torch.Tensor,
        room_size: torch.Tensor,
        decay: torch.Tensor,
        damping: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FDN reverb.

        Args:
            signal: (batch, n_samples) input audio.
            room_size: (batch,) normalized [0,1] -> scales delay lengths.
            decay: (batch,) normalized [0,1] -> feedback gain.
            damping: (batch,) normalized [0,1] -> high-freq absorption.
            mix: (batch,) dry/wet [0,1].
        """
        batch = signal.shape[0]
        device = signal.device
        mix_expand = mix.unsqueeze(1)

        # Denormalize
        scale = room_size * (self.ROOM_SCALE_MAX - self.ROOM_SCALE_MIN) + self.ROOM_SCALE_MIN
        delays = (self.base_delays.unsqueeze(0) * scale.unsqueeze(1)).long().clamp(min=1)
        # (batch, n_delays)

        g = decay * self.MAX_DECAY  # (batch,)
        damp = damping * 0.9 + 0.05  # (batch,) in [0.05, 0.95]

        # FFT of input
        n_fft = self.n_samples
        X = torch.fft.rfft(signal, n=n_fft)  # (batch, n_freq)
        n_freq = X.shape[1]
        freqs = torch.arange(n_freq, device=device, dtype=torch.float32)
        omega = 2.0 * math.pi * freqs / n_fft  # (n_freq,)

        # z^{-1} at each frequency bin
        z_inv = torch.exp(-1j * omega)  # (n_freq,)

        # Per-delay: z^{-m_i} and gain gamma_i
        # delays: (batch, n_delays), omega: (n_freq,)
        delays_f = delays.float()
        # z^{-m}: (batch, n_delays, n_freq)
        z_neg_m = torch.exp(
            -1j * omega.unsqueeze(0).unsqueeze(0) * delays_f.unsqueeze(2)
        )

        # Per-delay gain: g^{m_i}
        gamma = g.unsqueeze(1).pow(delays_f)  # (batch, n_delays)

        # One-pole damping in freq domain: H_lp(z) = (1-d) / (1 - d*z^{-1})
        damp_expand = damp.unsqueeze(1)  # (batch, 1)
        lp = (1.0 - damp_expand) / (
            1.0 - damp_expand * z_inv.unsqueeze(0)
        )  # (batch, n_freq)

        # Combined per-delay filter: gamma_i * H_lp(z)
        filt = gamma.unsqueeze(2) * lp.unsqueeze(1)  # (batch, n_delays, n_freq)

        # Build system for each freq bin:
        # H(z) = C @ (D - A @ Gamma)^{-1} @ B
        # D = diag(z^{-m_i}), A = feedback_matrix, Gamma = diag(filt)
        # B = ones(n_delays, 1), C = ones(1, n_delays) / n_delays

        # Construct (D - A @ Gamma) for each freq bin
        # D: (batch, n_delays, n_freq) diagonal entries = z^{-m_i}
        # A @ Gamma: A (n_delays, n_delays) @ diag(filt) -> broadcast

        A = self.feedback_matrix  # (n_delays, n_delays)
        # A @ diag(filt): (batch, n_delays, n_delays, n_freq)
        AG = A.unsqueeze(0).unsqueeze(3) * filt.unsqueeze(1)
        # (batch, n_delays, n_delays, n_freq)

        # D - AG: add diagonal z^{-m_i}
        eye = torch.eye(self.n_delays, device=device)
        D = eye.unsqueeze(0).unsqueeze(3) * z_neg_m.unsqueeze(2)
        # (batch, n_delays, n_delays, n_freq)

        system = D - AG  # (batch, n_delays, n_delays, n_freq)

        # Solve for each freq: system @ x = B, then H = C @ x
        B = torch.ones(self.n_delays, 1, device=device, dtype=torch.cfloat)
        C = torch.ones(1, self.n_delays, device=device, dtype=torch.cfloat) / self.n_delays

        # Permute to (batch, n_freq, n_delays, n_delays) for batched solve
        system_perm = system.permute(0, 3, 1, 2).contiguous()
        B_expand = B.unsqueeze(0).unsqueeze(0).expand(batch, n_freq, -1, -1)

        x = torch.linalg.solve(system_perm, B_expand)  # (batch, n_freq, n_delays, 1)
        H = (C.unsqueeze(0).unsqueeze(0) @ x).squeeze(-1).squeeze(-1)
        # (batch, n_freq)

        wet = torch.fft.irfft(X * H, n=n_fft)  # (batch, n_samples)
        wet = wet[:, : self.n_samples]

        return (1.0 - mix_expand) * signal + mix_expand * wet
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_reverb.py -v
```

Expected: 全部 PASS。注意 reverb 测试可能需要几秒（频域矩阵求逆）。

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/reverb.py tests/test_reverb.py
git commit -m "feat: differentiable FDN reverb in frequency domain"
```

---

## Task 5: EQ (3-band parametric)

**Files:**
- Create: `src/loom/effects/eq.py`
- Create: `tests/test_eq.py`

3 段 EQ：low shelf @ 200Hz, mid peak @ 1kHz, high shelf @ 5kHz。复用 biquad 系数计算和 `torchaudio.functional.lfilter`。

- [ ] **Step 1: 写失败测试**

`tests/test_eq.py`:
```python
import torch
import pytest
from loom.effects.eq import EQ
from loom.core import SAMPLE_RATE, N_SAMPLES


class TestEQ:
    def setup_method(self):
        self.eq = EQ(sample_rate=SAMPLE_RATE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.full((4,), 0.5),
            mid_gain=torch.full((4,), 0.5),
            high_gain=torch.full((4,), 0.5),
        )
        assert out.shape == (4, N_SAMPLES)

    def test_flat_eq_is_passthrough(self):
        """All gains at 0.5 (= 0dB) should be near passthrough."""
        torch.manual_seed(42)
        signal = torch.randn(1, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.tensor([0.5]),
            mid_gain=torch.tensor([0.5]),
            high_gain=torch.tensor([0.5]),
        )
        # Should be very close to input
        assert torch.allclose(out, signal, atol=0.05)

    def test_low_boost_increases_low_energy(self):
        """Boosting low gain should increase energy below 200Hz."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES)
        out = self.eq(
            noise,
            low_gain=torch.tensor([1.0]),   # +12dB
            mid_gain=torch.tensor([0.5]),    # 0dB
            high_gain=torch.tensor([0.5]),   # 0dB
        )
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE)
        low_mask = freqs < 200
        orig_low = torch.abs(torch.fft.rfft(noise[0]))[low_mask].pow(2).sum()
        eq_low = torch.abs(torch.fft.rfft(out[0]))[low_mask].pow(2).sum()
        assert eq_low > orig_low * 1.5

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES)
        out = self.eq(
            signal,
            low_gain=torch.tensor([0.0]),
            mid_gain=torch.tensor([1.0]),
            high_gain=torch.tensor([0.0]),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_eq.py -v
```

- [ ] **Step 3: 实现 EQ**

`src/loom/effects/eq.py`:
```python
import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class EQ(nn.Module):
    """3-band parametric EQ: low shelf, mid peak, high shelf.

    Uses biquad filters via torchaudio.lfilter. Gain range ±12dB.

    Args:
        sample_rate: Audio sample rate in Hz.
    """

    LOW_FREQ = 200.0
    MID_FREQ = 1000.0
    HIGH_FREQ = 5000.0
    MID_Q = 1.0
    GAIN_RANGE_DB = 12.0

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def _denorm_gain(self, gain: torch.Tensor) -> torch.Tensor:
        """[0,1] -> [-12dB, +12dB] -> linear amplitude."""
        db = (gain - 0.5) * 2.0 * self.GAIN_RANGE_DB
        return torch.pow(10.0, db / 40.0)  # sqrt for shelf/peak "A" parameter

    def _low_shelf_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.LOW_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * math.sqrt(2.0))  # S=1 slope

        Ap1 = A + 1.0
        Am1 = A - 1.0
        sqrt_A_alpha = 2.0 * torch.sqrt(A) * alpha

        b0 = A * (Ap1 - Am1 * cos_w0 + sqrt_A_alpha)
        b1 = 2.0 * A * (Am1 - Ap1 * cos_w0)
        b2 = A * (Ap1 - Am1 * cos_w0 - sqrt_A_alpha)
        a0 = Ap1 + Am1 * cos_w0 + sqrt_A_alpha
        a1 = -2.0 * (Am1 + Ap1 * cos_w0)
        a2 = Ap1 + Am1 * cos_w0 - sqrt_A_alpha

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _peak_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.MID_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * self.MID_Q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0 * torch.ones_like(A)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0 * torch.ones_like(A)
        a2 = 1.0 - alpha / A

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _high_shelf_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.HIGH_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * math.sqrt(2.0))

        Ap1 = A + 1.0
        Am1 = A - 1.0
        sqrt_A_alpha = 2.0 * torch.sqrt(A) * alpha

        b0 = A * (Ap1 + Am1 * cos_w0 + sqrt_A_alpha)
        b1 = -2.0 * A * (Am1 + Ap1 * cos_w0)
        b2 = A * (Ap1 + Am1 * cos_w0 - sqrt_A_alpha)
        a0 = Ap1 - Am1 * cos_w0 + sqrt_A_alpha
        a1 = 2.0 * (Am1 - Ap1 * cos_w0)
        a2 = Ap1 - Am1 * cos_w0 - sqrt_A_alpha

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _apply_biquad(
        self, signal: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        results = []
        for i in range(signal.shape[0]):
            filtered = AF.lfilter(
                signal[i].unsqueeze(0), a[i], b[i], clamp=False
            )
            results.append(filtered.squeeze(0))
        return torch.stack(results, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        low_gain: torch.Tensor,
        mid_gain: torch.Tensor,
        high_gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 3-band EQ.

        Args:
            signal: (batch, n_samples) input audio.
            low_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 200Hz.
            mid_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 1kHz.
            high_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 5kHz.
        """
        A_low = self._denorm_gain(low_gain).clamp(min=0.01)
        A_mid = self._denorm_gain(mid_gain).clamp(min=0.01)
        A_high = self._denorm_gain(high_gain).clamp(min=0.01)

        a_l, b_l = self._low_shelf_coeffs(A_low)
        a_m, b_m = self._peak_coeffs(A_mid)
        a_h, b_h = self._high_shelf_coeffs(A_high)

        out = self._apply_biquad(signal, a_l, b_l)
        out = self._apply_biquad(out, a_m, b_m)
        out = self._apply_biquad(out, a_h, b_h)
        return out
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_eq.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/loom/effects/eq.py tests/test_eq.py
git commit -m "feat: 3-band parametric EQ with low shelf, mid peak, high shelf"
```

---

## Task 6: 集成到 SubtractiveSynth + 更新 render.py

**Files:**
- Modify: `src/loom/synth.py`
- Modify: `src/loom/render.py`
- Modify: `tests/test_synth.py`

- [ ] **Step 1: 更新 synth.py**

在 `src/loom/synth.py` 中新增 import 和效果器实例化，修改 `forward` 加入完整效果器链。

替换 `src/loom/synth.py` 全文：
```python
import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.envelope import ADSR
from loom.filters import BiquadFilter
from loom.amplifier import VCA
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer with full effects chain.

    Signal flow:
        Oscillator -> Filter (with envelope) -> VCA (with envelope)
        -> Distortion -> Compressor -> Chorus -> Delay -> Reverb -> EQ
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.amp_envelope = ADSR(sample_rate, n_samples)
        self.filter_envelope = ADSR(sample_rate, n_samples)
        self.filter = BiquadFilter(sample_rate)
        self.vca = VCA()
        self.distortion = Distortion()
        self.compressor = Compressor()
        self.chorus = Chorus(sample_rate, n_samples)
        self.delay = Delay(sample_rate, n_samples)
        self.reverb = Reverb(sample_rate, n_samples)
        self.eq = EQ(sample_rate)

    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary."""
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
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        filt_env_mean = filt_env.mean(dim=1)
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

        # Effects chain
        audio = self.distortion(audio, params["dist_amount"], params["dist_mix"])
        audio = self.compressor(
            audio,
            params["comp_threshold"],
            params["comp_ratio"],
            params["comp_attack"],
            params["comp_release"],
            params["comp_makeup"],
            params["comp_mix"],
        )
        audio = self.chorus(
            audio, params["chorus_rate"], params["chorus_depth"], params["chorus_mix"]
        )
        audio = self.delay(
            audio, params["delay_time"], params["delay_feedback"], params["delay_mix"]
        )
        audio = self.reverb(
            audio,
            params["reverb_room_size"],
            params["reverb_decay"],
            params["reverb_damping"],
            params["reverb_mix"],
        )
        audio = self.eq(
            audio, params["eq_low_gain"], params["eq_mid_gain"], params["eq_high_gain"]
        )

        return audio
```

- [ ] **Step 2: 更新 render.py 的 random_params**

在 `src/loom/render.py` 的 `random_params` 函数 return dict 中追加：

```python
        # Compressor
        "comp_threshold": _rand((batch,)),
        "comp_ratio": _rand((batch,)),
        "comp_attack": _rand((batch,)),
        "comp_release": _rand((batch,)),
        "comp_makeup": _rand((batch,)),
        "comp_mix": _rand((batch,)),
        # Chorus
        "chorus_rate": _rand((batch,)),
        "chorus_depth": _rand((batch,)),
        "chorus_mix": _rand((batch,)),
        # Delay
        "delay_time": _rand((batch,)),
        "delay_feedback": _rand((batch,)),
        "delay_mix": _rand((batch,)),
        # Reverb
        "reverb_room_size": _rand((batch,)),
        "reverb_decay": _rand((batch,)),
        "reverb_damping": _rand((batch,)),
        "reverb_mix": _rand((batch,)),
        # EQ
        "eq_low_gain": _rand((batch,)),
        "eq_mid_gain": _rand((batch,)),
        "eq_high_gain": _rand((batch,)),
```

- [ ] **Step 3: 更新 test_synth.py 的 _make_params**

在 `tests/test_synth.py` 的 `_make_params` 方法中追加与上面相同的参数（全部设为 0.5 或 0.0 for mix params）：

```python
            # Compressor
            "comp_threshold": torch.full((batch,), 0.5),
            "comp_ratio": torch.full((batch,), 0.3),
            "comp_attack": torch.full((batch,), 0.5),
            "comp_release": torch.full((batch,), 0.5),
            "comp_makeup": torch.full((batch,), 0.0),
            "comp_mix": torch.full((batch,), 0.0),
            # Chorus
            "chorus_rate": torch.full((batch,), 0.5),
            "chorus_depth": torch.full((batch,), 0.5),
            "chorus_mix": torch.full((batch,), 0.0),
            # Delay
            "delay_time": torch.full((batch,), 0.5),
            "delay_feedback": torch.full((batch,), 0.3),
            "delay_mix": torch.full((batch,), 0.0),
            # Reverb
            "reverb_room_size": torch.full((batch,), 0.5),
            "reverb_decay": torch.full((batch,), 0.5),
            "reverb_damping": torch.full((batch,), 0.3),
            "reverb_mix": torch.full((batch,), 0.0),
            # EQ
            "eq_low_gain": torch.full((batch,), 0.5),
            "eq_mid_gain": torch.full((batch,), 0.5),
            "eq_high_gain": torch.full((batch,), 0.5),
```

- [ ] **Step 4: 运行全量测试**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS（已有测试不 break，因为 _make_params 中新效果器 mix=0 即 bypass）

- [ ] **Step 5: Commit**

```bash
git add src/loom/synth.py src/loom/render.py tests/test_synth.py
git commit -m "feat: integrate full effects chain into SubtractiveSynth"
```

---

## Task 7: 更新梯度测试

**Files:**
- Modify: `tests/test_gradients.py`

- [ ] **Step 1: 扩展 test_synth_has_gradients**

在 `tests/test_gradients.py` 的 `continuous_keys` 列表中追加新效果器参数：

```python
        continuous_keys = [
            "osc_pitch", "osc_detune",
            "amp_attack", "amp_decay", "amp_sustain", "amp_release",
            "filter_cutoff", "filter_q",
            "filt_env_attack", "filt_env_decay", "filt_env_sustain",
            "filt_env_release", "filt_env_amount",
            "dist_amount", "dist_mix", "master_gain",
            # New effects
            "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
            "comp_makeup", "comp_mix",
            "chorus_rate", "chorus_depth", "chorus_mix",
            "delay_time", "delay_feedback", "delay_mix",
            "reverb_room_size", "reverb_decay", "reverb_damping", "reverb_mix",
            "eq_low_gain", "eq_mid_gain", "eq_high_gain",
        ]
```

- [ ] **Step 2: 更新 test_parameter_recovery_converges 的 target_params**

Add the new params to `target_params` dict (all effects bypassed with mix=0 to keep convergence tractable, except EQ at flat 0.5):

```python
            # Compressor
            "comp_threshold": torch.tensor([0.5]),
            "comp_ratio": torch.tensor([0.3]),
            "comp_attack": torch.tensor([0.5]),
            "comp_release": torch.tensor([0.5]),
            "comp_makeup": torch.tensor([0.0]),
            "comp_mix": torch.tensor([0.0]),
            # Chorus
            "chorus_rate": torch.tensor([0.5]),
            "chorus_depth": torch.tensor([0.5]),
            "chorus_mix": torch.tensor([0.0]),
            # Delay
            "delay_time": torch.tensor([0.5]),
            "delay_feedback": torch.tensor([0.3]),
            "delay_mix": torch.tensor([0.0]),
            # Reverb
            "reverb_room_size": torch.tensor([0.5]),
            "reverb_decay": torch.tensor([0.5]),
            "reverb_damping": torch.tensor([0.3]),
            "reverb_mix": torch.tensor([0.0]),
            # EQ
            "eq_low_gain": torch.tensor([0.5]),
            "eq_mid_gain": torch.tensor([0.5]),
            "eq_high_gain": torch.tensor([0.5]),
```

Also add the new keys to `optimize_keys`.

- [ ] **Step 3: 运行梯度测试**

```bash
uv run pytest tests/test_gradients.py -v --timeout=120
```

Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_gradients.py
git commit -m "feat: extend gradient tests to cover all effects parameters"
```

---

## Task 8: 集成验证

- [ ] **Step 1: 全量测试**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: 快速渲染验证**

```bash
uv run python -c "from loom.render import random_params, render; audio = render(random_params(2)); print(audio.shape, audio.abs().max().item())"
```

Expected: `torch.Size([2, 176400])` + nonzero value

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "feat: Phase 0 effects chain expansion complete"
```
