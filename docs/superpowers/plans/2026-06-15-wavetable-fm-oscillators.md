# Wavetable + FM Oscillators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 WavetableOscillator 和 FMOscillator，通过 osc_type 混合权重集成到 SubtractiveSynth。

**Architecture:** 两个新振荡器作为独立 nn.Module，与 AdditiveOscillator 平级。SubtractiveSynth 通过 `osc_type (batch, 3)` 连续混合三者输出。Wavetable 用 `grid_sample` 做相位→波形插值，FM 用 `cumsum` 相位累积 + 固定频率比。

**Tech Stack:** PyTorch 2.x, pytest

**Spec:** `docs/superpowers/specs/2026-06-15-wavetable-fm-oscillators-design.md`

---

## File Structure

```
src/loom/
├── wavetable.py        # 新增
├── fm.py               # 新增
├── synth.py            # 修改
└── render.py           # 修改

tests/
├── test_wavetable.py   # 新增
├── test_fm.py          # 新增
├── test_synth.py       # 修改
└── test_gradients.py   # 修改
```

---

## Task 1: WavetableOscillator

**Files:**
- Create: `src/loom/wavetable.py`
- Create: `tests/test_wavetable.py`

- [ ] **Step 1: 写失败测试**

`tests/test_wavetable.py`:
```python
import torch
import pytest
from loom.wavetable import WavetableOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestWavetableOscillator:
    def setup_method(self):
        self.osc = WavetableOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        pitch = torch.full((batch,), 0.5, device=DEVICE)
        detune = torch.full((batch,), 0.5, device=DEVICE)
        position = torch.full((batch,), 0.5, device=DEVICE)
        audio = self.osc(pitch, detune, position)
        assert audio.shape == (batch, N_SAMPLES)

    def test_frequency(self):
        """Should produce correct fundamental frequency."""
        midi_note = 69  # A4
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        position = torch.tensor([0.0], device=DEVICE)  # saw-like
        audio = self.osc(pitch, detune, position)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert abs(peak_freq.item() - 440.0) < 2.0

    def test_position_changes_timbre(self):
        """Different wt_position should produce different waveforms."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        audio_a = self.osc(pitch, detune, torch.tensor([0.0], device=DEVICE))
        audio_b = self.osc(pitch, detune, torch.tensor([1.0], device=DEVICE))
        assert not torch.allclose(audio_a, audio_b)

    def test_no_nan(self):
        pitch = torch.tensor([0.01], device=DEVICE)
        detune = torch.tensor([0.99], device=DEVICE)
        position = torch.tensor([0.99], device=DEVICE)
        audio = self.osc(pitch, detune, position)
        assert not torch.isnan(audio).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_wavetable.py -v
```

Expected: FAIL — ModuleNotFoundError

- [ ] **Step 3: 实现 WavetableOscillator**

`src/loom/wavetable.py`:
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WavetableOscillator(nn.Module):
    """Wavetable oscillator with frame morphing.

    Reads from a 2D wavetable (n_frames, frame_size) using grid_sample
    for phase interpolation and linear blending for frame morphing.

    Built-in wavetable: 16 frames morphing from saw to square.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
        n_frames: Number of wavetable frames.
        frame_size: Samples per wavetable frame (single cycle).
    """

    MIDI_MIN = 24
    MIDI_MAX = 96

    def __init__(
        self,
        sample_rate: int,
        n_samples: int,
        n_frames: int = 16,
        frame_size: int = 2048,
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.frame_size = frame_size

        wavetable = self._build_default_wavetable(n_frames, frame_size)
        self.register_buffer("wavetable", wavetable)

        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _build_default_wavetable(self, n_frames: int, frame_size: int) -> torch.Tensor:
        """Build saw-to-square morph wavetable."""
        n_harmonics = frame_size // 2
        n = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        phase = torch.linspace(0, 2 * math.pi, frame_size, dtype=torch.float32).unsqueeze(0)
        harmonics = torch.sin(n.unsqueeze(1) * phase)  # (n_harmonics, frame_size)

        frames = []
        for i in range(n_frames):
            alpha = i / max(n_frames - 1, 1)
            # Saw: all harmonics with 1/n amplitude
            saw_amps = 1.0 / n * (2.0 / math.pi)
            # Square: odd harmonics only with 1/n amplitude
            square_amps = torch.where(
                n % 2 == 1, 1.0 / n * (4.0 / math.pi), torch.zeros_like(n)
            )
            amps = (1.0 - alpha) * saw_amps + alpha * square_amps
            frame = (amps.unsqueeze(1) * harmonics).sum(dim=0)
            # Normalize
            peak = frame.abs().max().clamp(min=1e-6)
            frames.append(frame / peak)

        return torch.stack(frames, dim=0)  # (n_frames, frame_size)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0

    def forward(
        self,
        pitch: torch.Tensor,
        detune: torch.Tensor,
        position: torch.Tensor,
    ) -> torch.Tensor:
        """Render audio from wavetable.

        Args:
            pitch: (batch,) normalized pitch [0,1] -> MIDI [24,96].
            detune: (batch,) normalized detune [0,1] -> [-100, +100] cents.
            position: (batch,) wavetable position [0,1] for frame morphing.

        Returns:
            (batch, n_samples) audio tensor.
        """
        batch = pitch.shape[0]
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        # Phase accumulation: normalized [0, 1)
        phase_inc = f0 / self.sample_rate  # (batch,)
        phase = torch.cumsum(
            phase_inc.unsqueeze(1).expand(-1, self.n_samples),
            dim=1,
        )
        phase = phase % 1.0  # (batch, n_samples)

        # Frame interpolation: get two adjacent frames and blend
        pos_scaled = position * (self.n_frames - 1)  # (batch,)
        frame_lo = pos_scaled.long().clamp(0, self.n_frames - 2)
        frame_hi = (frame_lo + 1).clamp(max=self.n_frames - 1)
        frac = (pos_scaled - frame_lo.float()).unsqueeze(1)  # (batch, 1)

        wt_lo = self.wavetable[frame_lo]  # (batch, frame_size)
        wt_hi = self.wavetable[frame_hi]  # (batch, frame_size)
        wt_blended = (1.0 - frac) * wt_lo + frac * wt_hi  # (batch, frame_size)

        # Read from blended wavetable using grid_sample
        # Reshape for grid_sample: wavetable as (batch, 1, 1, frame_size) "image"
        wt_4d = wt_blended.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, frame_size)

        # Grid: phase -> x coordinate in [-1, 1]
        grid_x = phase * 2.0 - 1.0  # (batch, n_samples) -> [-1, 1]
        grid_y = torch.zeros_like(grid_x)
        grid = torch.stack([grid_x, grid_y], dim=-1)  # (batch, n_samples, 2)
        grid = grid.unsqueeze(1)  # (batch, 1, n_samples, 2)

        audio = F.grid_sample(
            wt_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return audio.squeeze(1).squeeze(1)  # (batch, n_samples)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_wavetable.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/wavetable.py tests/test_wavetable.py
git commit -m "feat: wavetable oscillator with saw-to-square morph"
```

---

## Task 2: FMOscillator

**Files:**
- Create: `src/loom/fm.py`
- Create: `tests/test_fm.py`

- [ ] **Step 1: 写失败测试**

`tests/test_fm.py`:
```python
import torch
import pytest
from loom.fm import FMOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestFMOscillator:
    def setup_method(self):
        self.osc = FMOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        out = self.osc(
            pitch=torch.full((batch,), 0.5, device=DEVICE),
            detune=torch.full((batch,), 0.5, device=DEVICE),
            carrier_ratio=torch.full((batch,), 0.0, device=DEVICE),
            mod_ratio=torch.full((batch,), 0.0, device=DEVICE),
            mod_index=torch.full((batch,), 0.3, device=DEVICE),
        )
        assert out.shape == (batch, N_SAMPLES)

    def test_zero_mod_is_sine(self):
        """With mod_index=0, FM reduces to a pure carrier sine."""
        midi_note = 69
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        out = self.osc(
            pitch=pitch,
            detune=torch.tensor([0.5], device=DEVICE),
            carrier_ratio=torch.tensor([0.0], device=DEVICE),  # ratio=1
            mod_ratio=torch.tensor([0.0], device=DEVICE),
            mod_index=torch.tensor([0.0], device=DEVICE),  # no modulation
        )
        fft = torch.fft.rfft(out[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert abs(peak_freq.item() - 440.0) < 2.0

        # Should be nearly a pure sine — very little harmonic content
        peak_idx = torch.argmax(magnitudes[1:]) + 1
        fundamental_energy = magnitudes[peak_idx].item()
        total_energy = magnitudes[1:].sum().item()
        assert fundamental_energy / total_energy > 0.9

    def test_mod_index_adds_harmonics(self):
        """Higher mod_index should introduce more harmonics."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        carrier = torch.tensor([0.0], device=DEVICE)
        mod = torch.tensor([0.0], device=DEVICE)

        out_low = self.osc(pitch, detune, carrier, mod,
                           torch.tensor([0.05], device=DEVICE))
        out_high = self.osc(pitch, detune, carrier, mod,
                            torch.tensor([0.8], device=DEVICE))

        fft_low = torch.abs(torch.fft.rfft(out_low[0]))
        fft_high = torch.abs(torch.fft.rfft(out_high[0]))

        peak_low = torch.argmax(fft_low[1:]) + 1
        peak_high = torch.argmax(fft_high[1:]) + 1

        # High mod_index should have more energy spread across harmonics
        ratio_low = fft_low[peak_low] / fft_low[1:].sum()
        ratio_high = fft_high[peak_high] / fft_high[1:].sum()
        assert ratio_high < ratio_low  # more spread = lower peak ratio

    def test_no_nan(self):
        out = self.osc(
            pitch=torch.tensor([0.99], device=DEVICE),
            detune=torch.tensor([0.99], device=DEVICE),
            carrier_ratio=torch.tensor([0.99], device=DEVICE),
            mod_ratio=torch.tensor([0.99], device=DEVICE),
            mod_index=torch.tensor([0.99], device=DEVICE),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_fm.py -v
```

- [ ] **Step 3: 实现 FMOscillator**

`src/loom/fm.py`:
```python
import torch
import torch.nn as nn
import math


class FMOscillator(nn.Module):
    """FM oscillator with fixed carrier/modulator frequency ratios.

    Single carrier + single modulator. Uses frequency ratios (not absolute
    frequencies) to avoid the FM frequency non-convergence problem (DDX7).

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
    """

    MIDI_MIN = 24
    MIDI_MAX = 96
    RATIO_MIN = 1.0
    RATIO_MAX = 8.0
    MOD_INDEX_MAX = 20.0

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0

    def _denorm_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        return ratio * (self.RATIO_MAX - self.RATIO_MIN) + self.RATIO_MIN

    def _denorm_mod_index(self, mod_index: torch.Tensor) -> torch.Tensor:
        return mod_index * self.MOD_INDEX_MAX

    def forward(
        self,
        pitch: torch.Tensor,
        detune: torch.Tensor,
        carrier_ratio: torch.Tensor,
        mod_ratio: torch.Tensor,
        mod_index: torch.Tensor,
    ) -> torch.Tensor:
        """Render FM audio.

        Args:
            pitch: (batch,) normalized [0,1] -> MIDI [24,96].
            detune: (batch,) normalized [0,1] -> [-100, +100] cents.
            carrier_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_index: (batch,) normalized [0,1] -> [0, 20].

        Returns:
            (batch, n_samples) audio tensor.
        """
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        c_ratio = self._denorm_ratio(carrier_ratio)
        m_ratio = self._denorm_ratio(mod_ratio)
        m_idx = self._denorm_mod_index(mod_index)

        t = self.t.unsqueeze(0)  # (1, n_samples)
        f0 = f0.unsqueeze(1)  # (batch, 1)
        c_ratio = c_ratio.unsqueeze(1)
        m_ratio = m_ratio.unsqueeze(1)
        m_idx = m_idx.unsqueeze(1)

        # Modulator phase and signal
        mod_freq = f0 * m_ratio
        mod_phase = 2.0 * math.pi * mod_freq * t
        mod_signal = m_idx * torch.sin(mod_phase)

        # Carrier phase with FM modulation
        carrier_freq = f0 * c_ratio
        carrier_phase = 2.0 * math.pi * carrier_freq * t + mod_signal

        audio = torch.sin(carrier_phase)
        return audio
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_fm.py -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/fm.py tests/test_fm.py
git commit -m "feat: FM oscillator with fixed frequency ratios"
```

---

## Task 3: Integrate into SubtractiveSynth + update render.py

**Files:**
- Modify: `src/loom/synth.py`
- Modify: `src/loom/render.py`
- Modify: `tests/test_synth.py`

- [ ] **Step 1: Update synth.py**

Read the current `src/loom/synth.py` first, then apply these changes:

Add imports at the top:
```python
from loom.wavetable import WavetableOscillator
from loom.fm import FMOscillator
```

In `__init__`, after `self.oscillator = AdditiveOscillator(...)`, add:
```python
        self.wavetable_osc = WavetableOscillator(sample_rate, n_samples)
        self.fm_osc = FMOscillator(sample_rate, n_samples)
```

In `forward`, replace the oscillator section (the single `audio = self.oscillator(...)` block) with:
```python
        # Three oscillators, mixed by osc_type weights
        additive_out = self.oscillator(
            params["osc_pitch"],
            params["osc_waveform"],
            params["osc_detune"],
        )
        wavetable_out = self.wavetable_osc(
            params["osc_pitch"],
            params["osc_detune"],
            params["wt_position"],
        )
        fm_out = self.fm_osc(
            params["osc_pitch"],
            params["osc_detune"],
            params["fm_carrier_ratio"],
            params["fm_mod_ratio"],
            params["fm_mod_index"],
        )
        osc_type = params["osc_type"]  # (batch, 3)
        audio = (
            osc_type[:, 0:1] * additive_out
            + osc_type[:, 1:2] * wavetable_out
            + osc_type[:, 2:3] * fm_out
        )
```

- [ ] **Step 2: Update render.py**

Add to the `random_params` return dict (after `"osc_detune"`):
```python
        "osc_type": _one_hot_rand(batch, 3),
        "wt_position": _rand((batch,)),
        "fm_carrier_ratio": _rand((batch,)),
        "fm_mod_ratio": _rand((batch,)),
        "fm_mod_index": _rand((batch,)),
```

- [ ] **Step 3: Update test_synth.py _make_params**

Read the current `tests/test_synth.py` first. Add to `_make_params` return dict (after `"osc_detune"`):
```python
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]] * batch, device=DEVICE),
            "wt_position": torch.full((batch,), 0.5, device=DEVICE),
            "fm_carrier_ratio": torch.full((batch,), 0.0, device=DEVICE),
            "fm_mod_ratio": torch.full((batch,), 0.0, device=DEVICE),
            "fm_mod_index": torch.full((batch,), 0.0, device=DEVICE),
```

Note: `osc_type = [1,0,0]` means additive only, so existing tests pass unchanged.

- [ ] **Step 4: Run full test suite**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 5: Commit**

```bash
git add src/loom/synth.py src/loom/render.py tests/test_synth.py
git commit -m "feat: integrate wavetable + FM oscillators into SubtractiveSynth"
```

---

## Task 4: Update gradient tests

**Files:**
- Modify: `tests/test_gradients.py`

- [ ] **Step 1: Update test_synth_has_gradients**

Read the current `tests/test_gradients.py` first. Add to `continuous_keys`:
```python
            "wt_position",
            "fm_carrier_ratio", "fm_mod_ratio", "fm_mod_index",
```

Add to `blend_keys`:
```python
        blend_keys = ["osc_waveform", "filter_type", "osc_type"]
```

- [ ] **Step 2: Update test_parameter_recovery_converges**

Add to `target_params`:
```python
            "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=DEVICE),
            "wt_position": torch.tensor([0.5], device=DEVICE),
            "fm_carrier_ratio": torch.tensor([0.0], device=DEVICE),
            "fm_mod_ratio": torch.tensor([0.0], device=DEVICE),
            "fm_mod_index": torch.tensor([0.0], device=DEVICE),
```

Add the new keys to `optimize_keys`:
```python
            "wt_position",
            "fm_carrier_ratio", "fm_mod_ratio", "fm_mod_index",
```

Add to `bypass_keys`:
```python
            "wt_position",
            "fm_carrier_ratio", "fm_mod_ratio", "fm_mod_index",
```

Add `"osc_type"` handling — it's a blend key like `"osc_waveform"`, not optimized (kept at `[1,0,0]`).

- [ ] **Step 3: Run gradient tests**

```bash
uv run pytest tests/test_gradients.py -v --timeout=180
```

Expected: 全部 PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_gradients.py
git commit -m "feat: extend gradient tests for wavetable + FM parameters"
```

---

## Task 5: Integration verification + demo update

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: Update demo script**

Add two new sounds to `scripts/demo.py` (read it first, add after the existing 6 sounds):

```python
    # --- 7. Wavetable Morph: position sweep ---
    print("7. Wavetable Morph")
    p = make_base_params()
    p["osc_type"] = torch.tensor([[0.0, 1.0, 0.0]])  # wavetable
    p["osc_pitch"] = torch.tensor([0.5])
    p["wt_position"] = torch.tensor([0.3])
    p["filter_cutoff"] = torch.tensor([0.6])
    p["filt_env_amount"] = torch.tensor([0.6])
    p["chorus_rate"] = torch.tensor([0.4])
    p["chorus_depth"] = torch.tensor([0.5])
    p["chorus_mix"] = torch.tensor([0.4])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "07_wavetable_morph")

    # --- 8. FM Electric Piano ---
    print("8. FM Electric Piano")
    p = make_base_params()
    p["osc_type"] = torch.tensor([[0.0, 0.0, 1.0]])  # FM
    p["osc_pitch"] = torch.tensor([0.55])  # ~A3
    p["fm_carrier_ratio"] = torch.tensor([0.0])  # ratio=1
    p["fm_mod_ratio"] = torch.tensor([0.0])     # ratio=1
    p["fm_mod_index"] = torch.tensor([0.15])    # mild FM
    p["amp_attack"] = torch.tensor([0.1])
    p["amp_decay"] = torch.tensor([0.5])
    p["amp_sustain"] = torch.tensor([0.3])
    p["filter_cutoff"] = torch.tensor([0.7])
    p["reverb_room_size"] = torch.tensor([0.4])
    p["reverb_decay"] = torch.tensor([0.4])
    p["reverb_mix"] = torch.tensor([0.3])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "08_fm_epiano")
```

Also add to `make_base_params`:
```python
        "osc_type": torch.tensor([[1.0, 0.0, 0.0]] * batch),
        "wt_position": torch.full((batch,), 0.5),
        "fm_carrier_ratio": torch.full((batch,), 0.0),
        "fm_mod_ratio": torch.full((batch,), 0.0),
        "fm_mod_index": torch.full((batch,), 0.0),
```

- [ ] **Step 3: Commit**

```bash
git add scripts/demo.py
git commit -m "feat: add wavetable morph and FM epiano demos"
```
