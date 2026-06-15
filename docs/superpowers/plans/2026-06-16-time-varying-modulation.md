# Time-Varying Modulation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 LFO 调制——让 wobble/vibrato 真正生效。新增 SVFilter，改造振荡器和 distortion 支持 per-sample 时变参数。

**Architecture:** SVFilter 用 chunk-based SVF 实现时变滤波；振荡器改为接受可选 `freq_mod (batch, n_samples)` 做 per-sample 相位累积；Distortion 的 drive 支持 per-sample tensor。synth.py 中 LFO 信号直接传给各模块，不再 `.mean()`。

**Tech Stack:** PyTorch 2.x, torchaudio, pytest

**Spec:** `docs/superpowers/specs/2026-06-16-time-varying-modulation-design.md`

---

## File Structure

```
src/loom/
├── svfilter.py             # 新增 — chunk-based State Variable Filter
├── oscillators.py          # 修改 — 支持 freq_mod per-sample
├── wavetable.py            # 修改 — 支持 freq_mod per-sample
├── fm.py                   # 修改 — 支持 freq_mod per-sample
├── effects/distortion.py   # 修改 — amount 支持 (batch, n_samples)
├── synth.py                # 修改 — LFO 时变调制集成
└── render.py               # 不变

tests/
├── test_svfilter.py        # 新增
├── test_oscillators.py     # 修改
├── test_wavetable.py       # 修改
├── test_fm.py              # 修改
├── test_distortion.py      # 修改
├── test_synth.py           # 修改
└── test_gradients.py       # 修改

scripts/
└── demo.py                 # 修改 — 更新 wobble demo
```

---

## Task 1: SVFilter (chunk-based State Variable Filter)

**Files:**
- Create: `src/loom/svfilter.py`
- Create: `tests/test_svfilter.py`

- [ ] **Step 1: 写失败测试**

`tests/test_svfilter.py`:
```python
import torch
import pytest
from loom.svfilter import SVFilter
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestSVFilter:
    def setup_method(self):
        self.filt = SVFilter(sample_rate=SAMPLE_RATE).to(DEVICE)

    def test_output_shape(self):
        signal = torch.randn(4, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((4, N_SAMPLES), 0.5, device=DEVICE)
        q = torch.full((4,), 0.5, device=DEVICE)
        filter_type = torch.zeros(4, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert out.shape == (4, N_SAMPLES)

    def test_static_lowpass_attenuates_highs(self):
        """Static low cutoff should attenuate high frequencies."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.3, device=DEVICE)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0

        filtered = self.filt(noise, cutoff, q, filter_type)
        fft_orig = torch.abs(torch.fft.rfft(noise[0]))
        fft_filt = torch.abs(torch.fft.rfft(filtered[0]))

        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        high_mask = freqs > 5000
        assert fft_filt[high_mask].mean() < fft_orig[high_mask].mean() * 0.5

    def test_time_varying_cutoff_produces_sweep(self):
        """Sweeping cutoff should produce different spectra in first vs second half."""
        torch.manual_seed(42)
        noise = torch.randn(1, N_SAMPLES, device=DEVICE)
        # Cutoff sweeps from low to high
        cutoff = torch.linspace(0.1, 0.9, N_SAMPLES, device=DEVICE).unsqueeze(0)
        q = torch.tensor([0.5], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0

        filtered = self.filt(noise, cutoff, q, filter_type)
        half = N_SAMPLES // 2
        first_half_high = torch.abs(torch.fft.rfft(filtered[0, :half])).mean()
        second_half_high = torch.abs(torch.fft.rfft(filtered[0, half:])).mean()
        # Second half has higher cutoff -> more high freq energy
        assert second_half_high > first_half_high

    def test_no_nan(self):
        signal = torch.randn(1, N_SAMPLES, device=DEVICE)
        cutoff = torch.full((1, N_SAMPLES), 0.01, device=DEVICE)
        q = torch.tensor([0.99], device=DEVICE)
        filter_type = torch.zeros(1, 3, device=DEVICE)
        filter_type[:, 0] = 1.0
        out = self.filt(signal, cutoff, q, filter_type)
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_svfilter.py -v
```

- [ ] **Step 3: 实现 SVFilter**

`src/loom/svfilter.py`:
```python
import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class SVFilter(nn.Module):
    """Chunk-based State Variable Filter with per-sample cutoff modulation.

    Processes audio in chunks (default 64 samples). Each chunk uses the cutoff
    value at its midpoint. Filter state carries across chunks for continuity.

    Supports LP, HP, BP via continuous blend weights.

    Args:
        sample_rate: Audio sample rate in Hz.
        chunk_size: Samples per processing chunk. Controls modulation resolution.
    """

    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    MIN_Q = 0.5
    MAX_Q = 20.0

    def __init__(self, sample_rate: int, chunk_size: int = 64):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def _denorm_cutoff(self, cutoff: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_HZ)
        log_max = math.log(self.MAX_HZ)
        return torch.exp(cutoff * (log_max - log_min) + log_min)

    def _denorm_q(self, q: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_Q)
        log_max = math.log(self.MAX_Q)
        return torch.exp(q * (log_max - log_min) + log_min)

    def _svf_coeffs(self, cutoff_hz: torch.Tensor, q_val: torch.Tensor):
        """Compute SVF coefficients g and R from cutoff and Q."""
        g = torch.tan(math.pi * cutoff_hz / self.sample_rate)
        R = 1.0 / (2.0 * q_val)
        return g, R

    def forward(
        self,
        signal: torch.Tensor,
        cutoff: torch.Tensor,
        q: torch.Tensor,
        filter_type: torch.Tensor,
    ) -> torch.Tensor:
        """Apply SVF with time-varying cutoff.

        Args:
            signal: (batch, n_samples) input audio.
            cutoff: (batch, n_samples) normalized cutoff [0,1] (per-sample).
            q: (batch,) normalized Q [0,1].
            filter_type: (batch, 3) blend weights for [LP, HP, BP].

        Returns:
            (batch, n_samples) filtered audio.
        """
        batch, n_samples = signal.shape
        device = signal.device
        q_val = self._denorm_q(q)  # (batch,)

        n_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size
        output_chunks = []

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, n_samples)
            chunk = signal[:, start:end]  # (batch, chunk_len)
            chunk_len = end - start

            # Cutoff at chunk midpoint
            mid = min((start + end) // 2, n_samples - 1)
            chunk_cutoff_hz = self._denorm_cutoff(cutoff[:, mid])  # (batch,)
            g, R = self._svf_coeffs(chunk_cutoff_hz, q_val)

            # SVF as biquad-equivalent for this chunk:
            # LP: b = [g^2, 2*g^2, g^2], a = [1+2*R*g+g^2, 2*(g^2-1), 1-2*R*g+g^2]
            g2 = g * g
            a0 = 1.0 + 2.0 * R * g + g2
            # LP coefficients
            lp_b = torch.stack([g2/a0, 2.0*g2/a0, g2/a0], dim=-1)
            lp_a = torch.stack([
                torch.ones_like(a0),
                (2.0 * (g2 - 1.0)) / a0,
                (1.0 - 2.0 * R * g + g2) / a0,
            ], dim=-1)
            # HP coefficients
            hp_b = torch.stack([1.0/a0, -2.0/a0, 1.0/a0], dim=-1)
            hp_a = lp_a  # same denominator
            # BP coefficients
            bp_b = torch.stack([2.0*R*g/a0, torch.zeros_like(a0), -2.0*R*g/a0], dim=-1)
            bp_a = lp_a

            # Apply each filter type and blend
            chunk_out = torch.zeros_like(chunk)
            for j, (b, a) in enumerate([(lp_b, lp_a), (hp_b, hp_a), (bp_b, bp_a)]):
                w = filter_type[:, j]  # (batch,)
                if (w.abs() < 1e-6).all():
                    continue
                # Batch lfilter
                filtered = torch.zeros_like(chunk)
                for i in range(batch):
                    f = AF.lfilter(chunk[i:i+1], a[i], b[i], clamp=False)
                    filtered[i] = f.squeeze(0)
                chunk_out = chunk_out + w.unsqueeze(1) * filtered

            output_chunks.append(chunk_out)

        return torch.cat(output_chunks, dim=1)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_svfilter.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/loom/svfilter.py tests/test_svfilter.py
git commit -m "feat: chunk-based SVF with per-sample cutoff modulation"
```

---

## Task 2: Oscillators per-sample frequency modulation

**Files:**
- Modify: `src/loom/oscillators.py`
- Modify: `src/loom/wavetable.py`
- Modify: `src/loom/fm.py`
- Modify: `tests/test_oscillators.py`
- Modify: `tests/test_wavetable.py`
- Modify: `tests/test_fm.py`

All three oscillators get a new optional `freq_mod: Tensor | None` parameter. When provided, it's `(batch, n_samples)` and modulates frequency per-sample: `f(t) = f0 * freq_mod[t]`. When `None`, behavior is identical to before.

- [ ] **Step 1: Add vibrato test to each oscillator test file**

In `tests/test_oscillators.py`, add to `TestAdditiveOscillator`:
```python
    def test_freq_mod_vibrato(self):
        """Per-sample freq_mod should spread spectrum (vibrato)."""
        pitch = torch.tensor([0.5], device=DEVICE)
        waveform = torch.zeros(1, 4, device=DEVICE)
        waveform[:, 0] = 1.0
        # No modulation
        audio_static = self.osc(pitch, waveform)
        # With vibrato: 5Hz sine modulation ±5%
        t = torch.arange(N_SAMPLES, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        freq_mod = 1.0 + 0.05 * torch.sin(2 * 3.14159 * 5.0 * t)
        freq_mod = freq_mod.unsqueeze(0)  # (1, n_samples)
        audio_vibrato = self.osc(pitch, waveform, freq_mod=freq_mod)

        fft_s = torch.abs(torch.fft.rfft(audio_static[0]))
        fft_v = torch.abs(torch.fft.rfft(audio_vibrato[0]))
        peak = torch.argmax(fft_s[1:]) + 1
        # Vibrato should spread energy to sidebands
        side_s = fft_s[max(1,peak-30):peak+30].sum() - fft_s[peak]
        side_v = fft_v[max(1,peak-30):peak+30].sum() - fft_v[peak]
        assert side_v > side_s * 1.5
```

In `tests/test_wavetable.py`, add to `TestWavetableOscillator`:
```python
    def test_freq_mod_vibrato(self):
        """Per-sample freq_mod should produce vibrato effect."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        position = torch.tensor([0.0], device=DEVICE)
        audio_static = self.osc(pitch, detune, position)
        t = torch.arange(N_SAMPLES, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        freq_mod = 1.0 + 0.05 * torch.sin(2 * 3.14159 * 5.0 * t)
        audio_vibrato = self.osc(pitch, detune, position, freq_mod=freq_mod.unsqueeze(0))
        assert not torch.allclose(audio_static, audio_vibrato, atol=0.01)
```

In `tests/test_fm.py`, add to `TestFMOscillator`:
```python
    def test_freq_mod_vibrato(self):
        """Per-sample freq_mod should modulate the base frequency."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        cr = torch.tensor([0.0], device=DEVICE)
        mr = torch.tensor([0.0], device=DEVICE)
        mi = torch.tensor([0.0], device=DEVICE)
        audio_static = self.osc(pitch, detune, cr, mr, mi)
        t = torch.arange(N_SAMPLES, dtype=torch.float32, device=DEVICE) / SAMPLE_RATE
        freq_mod = 1.0 + 0.05 * torch.sin(2 * 3.14159 * 5.0 * t)
        audio_vibrato = self.osc(pitch, detune, cr, mr, mi, freq_mod=freq_mod.unsqueeze(0))
        assert not torch.allclose(audio_static, audio_vibrato, atol=0.01)
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_oscillators.py::TestAdditiveOscillator::test_freq_mod_vibrato tests/test_wavetable.py::TestWavetableOscillator::test_freq_mod_vibrato tests/test_fm.py::TestFMOscillator::test_freq_mod_vibrato -v
```

Expected: FAIL — TypeError (unexpected keyword argument 'freq_mod')

- [ ] **Step 3: Modify AdditiveOscillator**

In `src/loom/oscillators.py`, change `forward` signature to:
```python
    def forward(
        self,
        pitch: torch.Tensor,
        waveform: torch.Tensor,
        detune: torch.Tensor | None = None,
        freq_mod: torch.Tensor | None = None,
    ) -> torch.Tensor:
```

Replace the phase computation block (lines ~81-108) with:
```python
        batch = pitch.shape[0]
        f0 = self._denorm_pitch(pitch)

        if detune is not None:
            cents = self._denorm_detune(detune)
            f0 = f0 * torch.pow(2.0, cents / 1200.0)

        nyquist = self.sample_rate / 2.0
        max_h = torch.clamp(
            torch.floor(nyquist / f0).long(), min=1, max=self.max_harmonics
        )
        n_h = max_h.max().item()

        harm_amps = self._harmonic_amplitudes(n_h, pitch.device)
        blended = torch.einsum("bw,wh->bh", waveform, harm_amps)

        harmonic_n = torch.arange(1, n_h + 1, device=pitch.device).float()
        mask = harmonic_n.unsqueeze(0) <= max_h.unsqueeze(1)
        blended = blended * mask.float()

        if freq_mod is not None:
            # Per-sample frequency: f0 * freq_mod * harmonic_n
            # freq_mod: (batch, n_samples), f0: (batch,)
            f_t = f0.unsqueeze(1) * freq_mod  # (batch, n_samples)
            # Phase = cumsum(2π * f(t) * n / sr) for each harmonic
            phase_inc = 2.0 * math.pi * f_t.unsqueeze(1) * harmonic_n.unsqueeze(0).unsqueeze(2) / self.sample_rate
            # (batch, n_h, n_samples)
            phases = torch.cumsum(phase_inc, dim=2)
        else:
            freqs = f0.unsqueeze(1) * harmonic_n.unsqueeze(0)
            phases = (
                2.0 * math.pi * freqs.unsqueeze(2) * self.t.unsqueeze(0).unsqueeze(0)
            )

        harmonics = torch.sin(phases)
        audio = torch.einsum("bh,bht->bt", blended, harmonics)
        return audio
```

- [ ] **Step 4: Modify WavetableOscillator**

In `src/loom/wavetable.py`, change `forward` signature to:
```python
    def forward(self, pitch, detune, position, freq_mod=None):
```

Replace the phase accumulation section with:
```python
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        if freq_mod is not None:
            # Per-sample phase increment
            f_t = f0.unsqueeze(1) * freq_mod  # (batch, n_samples)
            phase_inc = f_t / self.sample_rate
            phase = torch.cumsum(phase_inc, dim=1) % 1.0
        else:
            phase_inc = f0 / self.sample_rate
            phase = torch.cumsum(phase_inc.unsqueeze(1).expand(-1, self.n_samples), dim=1)
            phase = phase % 1.0
```

- [ ] **Step 5: Modify FMOscillator**

In `src/loom/fm.py`, change `forward` signature to:
```python
    def forward(self, pitch, detune, carrier_ratio, mod_ratio, mod_index, freq_mod=None):
```

Replace the phase computation with:
```python
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        c_ratio = self._denorm_ratio(carrier_ratio)
        m_ratio = self._denorm_ratio(mod_ratio)
        m_idx = self._denorm_mod_index(mod_index)

        t = self.t.unsqueeze(0)
        c_ratio = c_ratio.unsqueeze(1)
        m_ratio = m_ratio.unsqueeze(1)
        m_idx = m_idx.unsqueeze(1)

        if freq_mod is not None:
            f0_t = f0.unsqueeze(1) * freq_mod  # (batch, n_samples)
        else:
            f0_t = f0.unsqueeze(1).expand(-1, self.n_samples)

        mod_phase = torch.cumsum(2.0 * math.pi * f0_t * m_ratio / self.sample_rate, dim=1)
        mod_signal = m_idx * torch.sin(mod_phase)
        carrier_phase = torch.cumsum(2.0 * math.pi * f0_t * c_ratio / self.sample_rate, dim=1) + mod_signal

        return torch.sin(carrier_phase)
```

- [ ] **Step 6: 运行全量测试**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS（新测试 + 旧测试）

- [ ] **Step 7: Commit**

```bash
git add src/loom/oscillators.py src/loom/wavetable.py src/loom/fm.py tests/test_oscillators.py tests/test_wavetable.py tests/test_fm.py
git commit -m "feat: oscillators support per-sample freq_mod for vibrato/LFO"
```

---

## Task 3: Distortion per-sample drive

**Files:**
- Modify: `src/loom/effects/distortion.py`
- Modify: `tests/test_distortion.py`

- [ ] **Step 1: Add per-sample drive test**

In `tests/test_distortion.py`, add:
```python
    def test_per_sample_drive(self):
        """amount can be (batch, n_samples) for time-varying drive."""
        signal = torch.sin(torch.linspace(0, 100, 1000, device=DEVICE)).unsqueeze(0)
        # Ramp drive from 0 to 1 over time
        amount = torch.linspace(0.0, 1.0, 1000, device=DEVICE).unsqueeze(0)
        mix = torch.tensor([1.0], device=DEVICE)
        out = self.dist(signal, amount, mix)
        assert out.shape == (1, 1000)
        # End should be more distorted (more harmonics) than start
        fft_start = torch.abs(torch.fft.rfft(out[0, :500]))
        fft_end = torch.abs(torch.fft.rfft(out[0, 500:]))
        assert fft_end.sum() > fft_start.sum()
```

- [ ] **Step 2: Modify Distortion to handle both shapes**

In `src/loom/effects/distortion.py`, change `forward`:
```python
    def forward(self, signal, amount, mix):
        """Apply distortion.

        Args:
            signal: (batch, n_samples) input audio.
            amount: (batch,) or (batch, n_samples) normalized drive [0,1].
            mix: (batch,) dry/wet [0,1].
        """
        drive = self._denorm_drive(amount)
        if drive.dim() == 1:
            drive = drive.unsqueeze(1)
        mix_v = mix
        if mix_v.dim() == 1:
            mix_v = mix_v.unsqueeze(1)
        wet = torch.tanh(signal * drive)
        return (1.0 - mix_v) * signal + mix_v * wet
```

- [ ] **Step 3: Run tests and commit**

```bash
uv run pytest tests/test_distortion.py -v
git add src/loom/effects/distortion.py tests/test_distortion.py
git commit -m "feat: distortion supports per-sample drive for LFO modulation"
```

---

## Task 4: Synth LFO integration — remove .mean(), use time-varying signals

**Files:**
- Modify: `src/loom/synth.py`
- Modify: `tests/test_synth.py`

- [ ] **Step 1: Add wobble test to test_synth.py**

In `tests/test_synth.py`, add:
```python
    def test_lfo_wobble_differs_from_static(self):
        """LFO with depth>0 targeting cutoff should produce different audio."""
        params_static = self._make_params()
        params_wobble = self._make_params()
        params_wobble["lfo_depth"] = torch.full((1,), 0.9, device=DEVICE)
        params_wobble["lfo_target"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
        params_wobble["lfo_rate"] = torch.tensor([0.3], device=DEVICE)
        audio_s = self.synth(params_static)
        audio_w = self.synth(params_wobble)
        assert not torch.allclose(audio_s, audio_w, atol=0.01)
```

- [ ] **Step 2: Rewrite synth.py forward with time-varying LFO**

Read current `src/loom/synth.py`. Replace BiquadFilter with SVFilter import and instance:

```python
from loom.svfilter import SVFilter
```

In `__init__`, replace `self.filter = BiquadFilter(sample_rate)` with:
```python
        self.filter = SVFilter(sample_rate)
```

Replace the entire `forward` method with:
```python
    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary."""
        n_samples = self.oscillator.n_samples

        # LFO signal: (batch, n_samples) in [-depth, +depth]
        lfo_signal = self.lfo(
            params["lfo_rate"],
            params["lfo_depth"],
            params["lfo_waveform"],
            params["lfo_phase"],
        )
        lfo_target = params["lfo_target"]  # (batch, 4)

        # Per-sample frequency modulation for oscillators
        # freq_mod: multiplicative, centered at 1.0
        pitch_lfo = lfo_target[:, 1:2] * lfo_signal * 0.05  # (batch, n_samples)
        freq_mod = 1.0 + pitch_lfo  # (batch, n_samples)
        # Only pass freq_mod if LFO is actually active (avoids unnecessary computation)
        has_pitch_mod = (params["lfo_depth"].abs() > 1e-4).any() and (lfo_target[:, 1].abs() > 1e-4).any()
        fm_arg = freq_mod if has_pitch_mod else None

        # Oscillators with per-sample freq_mod
        additive_out = self.oscillator(
            params["osc_pitch"],
            params["osc_waveform"],
            params["osc_detune"],
            freq_mod=fm_arg,
        )
        wavetable_out = self.wavetable_osc(
            params["osc_pitch"],
            params["osc_detune"],
            params["wt_position"],
            freq_mod=fm_arg,
        )
        fm_out = self.fm_osc(
            params["osc_pitch"],
            params["osc_detune"],
            params["fm_carrier_ratio"],
            params["fm_mod_ratio"],
            params["fm_mod_index"],
            freq_mod=fm_arg,
        )
        osc_type = params["osc_type"]
        audio = (
            osc_type[:, 0:1] * additive_out
            + osc_type[:, 1:2] * wavetable_out
            + osc_type[:, 2:3] * fm_out
        )

        # Filter with time-varying cutoff (envelope + LFO)
        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        # Build per-sample cutoff signal
        base_cutoff = params["filter_cutoff"].unsqueeze(1)  # (batch, 1)
        env_mod = amount.unsqueeze(1) * filt_env * 0.3       # (batch, n_samples)
        lfo_cutoff = lfo_target[:, 0:1] * lfo_signal * 0.3   # (batch, n_samples)
        cutoff_signal = (base_cutoff + env_mod + lfo_cutoff).clamp(0.0, 1.0)

        audio = self.filter(
            audio, cutoff_signal, params["filter_q"], params["filter_type"]
        )

        # Amplitude envelope + VCA
        amp_env = self.amp_envelope(
            params["amp_attack"],
            params["amp_decay"],
            params["amp_sustain"],
            params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        # Distortion with per-sample drive
        dist_lfo = lfo_target[:, 2:3] * lfo_signal * 0.3  # (batch, n_samples)
        dist_drive = (params["dist_amount"].unsqueeze(1) + dist_lfo).clamp(0.0, 1.0)
        audio = self.distortion(audio, dist_drive, params["dist_mix"])

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

Note: `BiquadFilter` import can be removed from synth.py since it's replaced by SVFilter. BiquadFilter is still used by EQ independently.

- [ ] **Step 3: Run full tests**

```bash
uv run pytest tests/ -v --timeout=180
```

- [ ] **Step 4: Commit**

```bash
git add src/loom/synth.py tests/test_synth.py
git commit -m "feat: time-varying LFO modulation — SVFilter cutoff sweep, vibrato, drive modulation"
```

---

## Task 5: Update gradient tests

**Files:**
- Modify: `tests/test_gradients.py`

The gradient test's `test_synth_has_gradients` should still pass since all the new time-varying code paths are differentiable. But the synth now uses SVFilter instead of BiquadFilter, which may change gradient behavior.

- [ ] **Step 1: Run gradient tests**

```bash
uv run pytest tests/test_gradients.py -v --timeout=180
```

If they pass, no changes needed — commit nothing. If they fail, debug and fix.

- [ ] **Step 2: Commit if changes were needed**

```bash
git add tests/test_gradients.py
git commit -m "fix: update gradient tests for SVFilter-based synth"
```

---

## Task 6: Demo update — fix wobble + update all LFO-using sounds

**Files:**
- Modify: `scripts/demo.py`

Read the current demo.py. The wobble bass (sound 10) should now work correctly since LFO actually modulates cutoff over time. But verify and also update other sounds that could benefit from LFO.

- [ ] **Step 1: Run demo and verify wobble works**

```bash
uv run python scripts/demo.py
```

Listen to `output/10_wobble_bass.wav` — should have audible filter sweep / wobble effect.

- [ ] **Step 2: If wobble sounds good, commit as-is. If not, adjust parameters.**

Likely working parameters for a good wobble:
```python
    p["lfo_rate"] = torch.tensor([0.35])     # ~2Hz wobble
    p["lfo_depth"] = torch.tensor([0.9])
    p["lfo_waveform"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # sine LFO
    p["lfo_target"] = torch.tensor([[1.0, 0.0, 0.3, 0.0]])    # cutoff + slight dist
```

Also consider adding LFO to some existing sounds:
- Sound 4 (Neuro Bass): add LFO→cutoff for actual neuro wobble
- Sound 2 (Supersaw Lead): add subtle LFO→pitch for vibrato

- [ ] **Step 3: Commit**

```bash
git add scripts/demo.py
git commit -m "feat: update demo with working LFO wobble and vibrato effects"
```

---

## Task 7: Integration verification

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -v --timeout=180
```

- [ ] **Step 2: Run demo and spot-check audio files**

```bash
uv run python scripts/demo.py
```

Verify:
- `10_wobble_bass.wav` has audible wobble
- Other sounds didn't break
- No silent/broken files

- [ ] **Step 3: Final commit**

```bash
git add -A
git commit -m "feat: time-varying modulation complete — LFO wobble/vibrato working"
```
