# LFO + Sequencer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 新增 LFO 调制模块（时变 wobble 效果）和 32-step sequencer（多音符节奏渲染）。

**Architecture:** LFO 作为 nn.Module 集成到 SubtractiveSynth，在 forward 中对 filter_cutoff/pitch/dist_amount 做时变调制。Sequencer 是独立函数，在 synth 之上循环渲染每个音符片段，拼接到 output buffer。ADSR 需要改为接受 runtime 的 note_on_duration 参数。

**Tech Stack:** PyTorch 2.x, pytest

**Spec:** `docs/superpowers/specs/2026-06-16-lfo-sequencer-design.md`

---

## File Structure

```
src/loom/
├── lfo.py              # 新增
├── envelope.py         # 修改 — note_on_duration 改为 forward 参数
├── synth.py            # 修改 — LFO 集成 + note_on_duration 透传
├── sequencer.py        # 新增
└── render.py           # 修改 — LFO 参数

tests/
├── test_lfo.py         # 新增
├── test_envelope.py    # 修改 — 测试动态 note_on_duration
├── test_sequencer.py   # 新增
├── test_synth.py       # 修改 — LFO 参数
└── test_gradients.py   # 修改 — LFO 参数
```

---

## Task 1: ADSR 支持动态 note_on_duration

**Files:**
- Modify: `src/loom/envelope.py`
- Modify: `tests/test_envelope.py`

Sequencer 需要每个音符有不同的 note_on_duration。当前 ADSR 在 `__init__` 时固定 note_on_duration=3.0。改为 forward 时接受可选参数。

- [ ] **Step 1: 增加测试验证动态 note_on_duration**

在 `tests/test_envelope.py` 的 `TestADSR` 类中追加：

```python
    def test_dynamic_note_on_duration(self):
        """Different note_on_duration should change where release starts."""
        attack = torch.tensor([0.1], device=DEVICE)
        decay = torch.tensor([0.2], device=DEVICE)
        sustain = torch.tensor([0.6], device=DEVICE)
        release = torch.tensor([0.3], device=DEVICE)

        env_short = self.adsr(attack, decay, sustain, release, note_on_duration=0.5)
        env_long = self.adsr(attack, decay, sustain, release, note_on_duration=2.0)

        # Short note should have less energy in the second half
        short_tail = env_short[0, N_SAMPLES // 2:].mean()
        long_tail = env_long[0, N_SAMPLES // 2:].mean()
        assert long_tail > short_tail
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_envelope.py::TestADSR::test_dynamic_note_on_duration -v
```

Expected: FAIL — TypeError (unexpected keyword argument)

- [ ] **Step 3: 修改 ADSR.forward 接受可选 note_on_duration**

Read `src/loom/envelope.py` first. Change the `forward` signature to:

```python
    def forward(
        self,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        note_on_duration: float | None = None,
    ) -> torch.Tensor:
```

And at the top of the method body, replace `self.note_on_duration` usage:

```python
        if note_on_duration is None:
            note_on_duration = self.note_on_duration
```

The rest of the method stays the same — it already uses `self.note_on_duration` only in the release ramp calculation.

- [ ] **Step 4: 运行全量测试确认通过**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS（现有测试不 break，因为参数是可选的）

- [ ] **Step 5: Commit**

```bash
git add src/loom/envelope.py tests/test_envelope.py
git commit -m "feat: ADSR supports dynamic note_on_duration parameter"
```

---

## Task 2: LFO 模块

**Files:**
- Create: `src/loom/lfo.py`
- Create: `tests/test_lfo.py`

- [ ] **Step 1: 写失败测试**

`tests/test_lfo.py`:
```python
import torch
import pytest
from loom.lfo import LFO
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestLFO:
    def setup_method(self):
        self.lfo = LFO(sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        out = self.lfo(
            rate=torch.full((batch,), 0.5, device=DEVICE),
            depth=torch.full((batch,), 0.5, device=DEVICE),
            waveform=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch, device=DEVICE),
            phase=torch.full((batch,), 0.0, device=DEVICE),
        )
        assert out.shape == (batch, N_SAMPLES)

    def test_zero_depth_is_zero(self):
        """depth=0 should produce all-zero output."""
        out = self.lfo(
            rate=torch.tensor([0.5], device=DEVICE),
            depth=torch.tensor([0.0], device=DEVICE),
            waveform=torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            phase=torch.tensor([0.0], device=DEVICE),
        )
        assert out.abs().max().item() < 1e-6

    def test_rate_affects_frequency(self):
        """Higher rate should produce higher frequency LFO."""
        waveform = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE)
        depth = torch.tensor([1.0], device=DEVICE)
        phase = torch.tensor([0.0], device=DEVICE)

        out_slow = self.lfo(torch.tensor([0.0], device=DEVICE), depth, waveform, phase)
        out_fast = self.lfo(torch.tensor([1.0], device=DEVICE), depth, waveform, phase)

        # Count zero crossings as proxy for frequency
        slow_crossings = ((out_slow[0, :-1] * out_slow[0, 1:]) < 0).sum()
        fast_crossings = ((out_fast[0, :-1] * out_fast[0, 1:]) < 0).sum()
        assert fast_crossings > slow_crossings

    def test_no_nan(self):
        out = self.lfo(
            rate=torch.tensor([0.99], device=DEVICE),
            depth=torch.tensor([0.99], device=DEVICE),
            waveform=torch.tensor([[0.25, 0.25, 0.25, 0.25]], device=DEVICE),
            phase=torch.tensor([0.99], device=DEVICE),
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_lfo.py -v
```

- [ ] **Step 3: 实现 LFO**

`src/loom/lfo.py`:
```python
import torch
import torch.nn as nn
import math


class LFO(nn.Module):
    """Low-frequency oscillator for parameter modulation.

    Generates a time-varying signal in [-1, 1] range, scaled by depth.
    Supports sine/saw/square/tri waveforms via continuous blending.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
    """

    RATE_MIN_HZ = 0.1
    RATE_MAX_HZ = 20.0

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
        rate: torch.Tensor,
        depth: torch.Tensor,
        waveform: torch.Tensor,
        phase: torch.Tensor,
    ) -> torch.Tensor:
        """Generate LFO modulation signal.

        Args:
            rate: (batch,) normalized [0,1] -> [0.1, 20] Hz.
            depth: (batch,) modulation depth [0,1]. 0 = no modulation.
            waveform: (batch, 4) blend weights [sine, saw, square, tri].
            phase: (batch,) initial phase offset [0,1] -> [0, 2π].

        Returns:
            (batch, n_samples) modulation signal in [-depth, +depth].
        """
        rate_hz = self._denorm_rate(rate)
        t = self.t.unsqueeze(0)  # (1, n_samples)
        phase_rad = phase.unsqueeze(1) * 2.0 * math.pi

        theta = 2.0 * math.pi * rate_hz.unsqueeze(1) * t + phase_rad
        # Normalize theta to [0, 2π) for non-sine waveforms
        theta_norm = theta % (2.0 * math.pi)
        frac = theta_norm / (2.0 * math.pi)  # [0, 1)

        # Four waveforms
        sine = torch.sin(theta)
        saw = 2.0 * frac - 1.0
        square = torch.sign(torch.sin(theta))
        tri = 4.0 * torch.abs(frac - 0.5) - 1.0

        # Blend
        waves = torch.stack([sine, saw, square, tri], dim=1)  # (batch, 4, n_samples)
        w = waveform.unsqueeze(2)  # (batch, 4, 1)
        blended = (w * waves).sum(dim=1)  # (batch, n_samples)

        return blended * depth.unsqueeze(1)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_lfo.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/loom/lfo.py tests/test_lfo.py
git commit -m "feat: LFO module with sine/saw/square/tri waveforms"
```

---

## Task 3: 集成 LFO 到 SubtractiveSynth

**Files:**
- Modify: `src/loom/synth.py`
- Modify: `src/loom/render.py`
- Modify: `tests/test_synth.py`
- Modify: `tests/test_gradients.py`

- [ ] **Step 1: 更新 synth.py**

Read current `src/loom/synth.py`. Add import at top:
```python
from loom.lfo import LFO
```

In `__init__`, add after `self.eq = EQ(sample_rate)`:
```python
        self.lfo = LFO(sample_rate, n_samples)
```

In `forward`, insert LFO computation right at the start (before oscillators), and use modulated values downstream:

```python
    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary."""
        # LFO modulation
        lfo_signal = self.lfo(
            params["lfo_rate"],
            params["lfo_depth"],
            params["lfo_waveform"],
            params["lfo_phase"],
        )
        lfo_target = params["lfo_target"]  # (batch, 4): cutoff, pitch, dist, pan

        # Modulated parameters
        pitch_mod = (
            params["osc_pitch"]
            + lfo_target[:, 1] * lfo_signal.mean(dim=1) * 0.05
        ).clamp(0.0, 1.0)

        # Oscillators use modulated pitch
        additive_out = self.oscillator(
            pitch_mod,
            params["osc_waveform"],
            params["osc_detune"],
        )
        wavetable_out = self.wavetable_osc(
            pitch_mod,
            params["osc_detune"],
            params["wt_position"],
        )
        fm_out = self.fm_osc(
            pitch_mod,
            params["osc_detune"],
            params["fm_carrier_ratio"],
            params["fm_mod_ratio"],
            params["fm_mod_index"],
        )
        osc_type = params["osc_type"]
        audio = (
            osc_type[:, 0:1] * additive_out
            + osc_type[:, 1:2] * wavetable_out
            + osc_type[:, 2:3] * fm_out
        )

        # Filter with LFO + envelope modulation on cutoff
        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        filt_env_mean = filt_env.mean(dim=1)
        # LFO cutoff modulation: time-varying, take mean for the static filter
        lfo_cutoff_mod = lfo_target[:, 0] * lfo_signal.mean(dim=1) * 0.3
        modulated_cutoff = (
            params["filter_cutoff"] + amount * filt_env_mean * 0.3 + lfo_cutoff_mod
        ).clamp(0.0, 1.0)

        audio = self.filter(
            audio, modulated_cutoff, params["filter_q"], params["filter_type"]
        )

        amp_env = self.amp_envelope(
            params["amp_attack"],
            params["amp_decay"],
            params["amp_sustain"],
            params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        # Distortion with LFO modulation on amount
        dist_mod = (
            params["dist_amount"]
            + lfo_target[:, 2] * lfo_signal.mean(dim=1) * 0.3
        ).clamp(0.0, 1.0)
        audio = self.distortion(audio, dist_mod, params["dist_mix"])

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

- [ ] **Step 2: 更新 render.py random_params**

Add after `"fm_mod_index"`:
```python
        "lfo_rate": _rand((batch,)),
        "lfo_depth": _rand((batch,)),
        "lfo_waveform": _one_hot_rand(batch, 4),
        "lfo_target": torch.zeros(batch, 4, device=device),
        "lfo_phase": _rand((batch,)),
```

Note: `lfo_target` defaults to zeros (no modulation routing) rather than random, because random routing + random depth would make most renders noisy.

- [ ] **Step 3: 更新 test_synth.py _make_params**

Add after `"fm_mod_index"`:
```python
            "lfo_rate": torch.full((batch,), 0.5, device=DEVICE),
            "lfo_depth": torch.full((batch,), 0.0, device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch, device=DEVICE),
            "lfo_target": torch.zeros(batch, 4, device=DEVICE),
            "lfo_phase": torch.full((batch,), 0.0, device=DEVICE),
```

`lfo_depth=0` bypasses LFO, existing tests pass unchanged.

- [ ] **Step 4: 更新 test_gradients.py**

Add to `continuous_keys`:
```python
            "lfo_rate", "lfo_depth", "lfo_phase",
```

Add to `blend_keys`:
```python
        blend_keys = ["osc_waveform", "filter_type", "osc_type", "lfo_waveform", "lfo_target"]
```

Add to `target_params`:
```python
            "lfo_rate": torch.tensor([0.5], device=DEVICE),
            "lfo_depth": torch.tensor([0.0], device=DEVICE),
            "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=DEVICE),
            "lfo_target": torch.zeros(1, 4, device=DEVICE),
            "lfo_phase": torch.tensor([0.0], device=DEVICE),
```

Add to `optimize_keys`:
```python
            "lfo_rate", "lfo_depth", "lfo_phase",
```

Add to `bypass_keys`:
```python
            "lfo_rate", "lfo_depth", "lfo_phase",
```

`lfo_waveform` and `lfo_target` are blend keys (not in optimize_keys), same as `osc_waveform`.

- [ ] **Step 5: Run full tests and commit**

```bash
uv run pytest tests/ -v
git add src/loom/synth.py src/loom/render.py tests/test_synth.py tests/test_gradients.py
git commit -m "feat: integrate LFO modulation into SubtractiveSynth"
```

---

## Task 4: Sequencer

**Files:**
- Create: `src/loom/sequencer.py`
- Create: `tests/test_sequencer.py`

- [ ] **Step 1: 写失败测试**

`tests/test_sequencer.py`:
```python
import torch
import pytest
from loom.sequencer import render_sequence
from loom.core import SAMPLE_RATE, DEVICE


def _make_synth_params(device=DEVICE):
    """Shared synth params for all notes (no sequencer-specific keys)."""
    return {
        "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device),
        "osc_detune": torch.full((1,), 0.5, device=device),
        "osc_type": torch.tensor([[1.0, 0.0, 0.0]], device=device),
        "wt_position": torch.full((1,), 0.5, device=device),
        "fm_carrier_ratio": torch.full((1,), 0.0, device=device),
        "fm_mod_ratio": torch.full((1,), 0.0, device=device),
        "fm_mod_index": torch.full((1,), 0.0, device=device),
        "lfo_rate": torch.full((1,), 0.5, device=device),
        "lfo_depth": torch.full((1,), 0.0, device=device),
        "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device),
        "lfo_target": torch.zeros(1, 4, device=device),
        "lfo_phase": torch.full((1,), 0.0, device=device),
        "amp_attack": torch.full((1,), 0.1, device=device),
        "amp_decay": torch.full((1,), 0.3, device=device),
        "amp_sustain": torch.full((1,), 0.7, device=device),
        "amp_release": torch.full((1,), 0.2, device=device),
        "filter_cutoff": torch.full((1,), 0.6, device=device),
        "filter_q": torch.full((1,), 0.3, device=device),
        "filter_type": torch.tensor([[1.0, 0.0, 0.0]], device=device),
        "filt_env_attack": torch.full((1,), 0.2, device=device),
        "filt_env_decay": torch.full((1,), 0.3, device=device),
        "filt_env_sustain": torch.full((1,), 0.5, device=device),
        "filt_env_release": torch.full((1,), 0.3, device=device),
        "filt_env_amount": torch.full((1,), 0.6, device=device),
        "dist_amount": torch.full((1,), 0.0, device=device),
        "dist_mix": torch.full((1,), 0.0, device=device),
        "master_gain": torch.full((1,), 0.8, device=device),
        "comp_threshold": torch.full((1,), 0.5, device=device),
        "comp_ratio": torch.full((1,), 0.3, device=device),
        "comp_attack": torch.full((1,), 0.5, device=device),
        "comp_release": torch.full((1,), 0.5, device=device),
        "comp_makeup": torch.full((1,), 0.0, device=device),
        "comp_mix": torch.full((1,), 0.0, device=device),
        "chorus_rate": torch.full((1,), 0.5, device=device),
        "chorus_depth": torch.full((1,), 0.5, device=device),
        "chorus_mix": torch.full((1,), 0.0, device=device),
        "delay_time": torch.full((1,), 0.5, device=device),
        "delay_feedback": torch.full((1,), 0.3, device=device),
        "delay_mix": torch.full((1,), 0.0, device=device),
        "reverb_room_size": torch.full((1,), 0.5, device=device),
        "reverb_decay": torch.full((1,), 0.5, device=device),
        "reverb_damping": torch.full((1,), 0.3, device=device),
        "reverb_mix": torch.full((1,), 0.0, device=device),
        "eq_low_gain": torch.full((1,), 0.5, device=device),
        "eq_mid_gain": torch.full((1,), 0.5, device=device),
        "eq_high_gain": torch.full((1,), 0.5, device=device),
    }


class TestSequencer:
    def test_output_shape(self):
        synth_params = _make_synth_params()
        bpm = 170.0
        step_sec = 60.0 / bpm / 8.0
        total_samples = int(32 * step_sec * SAMPLE_RATE)

        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.full((1, 32), 0.8, device=DEVICE)
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=bpm, sample_rate=SAMPLE_RATE,
        )
        assert out.shape[0] == 1
        assert abs(out.shape[1] - total_samples) < SAMPLE_RATE  # within 1 sec

    def test_silence_when_all_velocity_zero(self):
        synth_params = _make_synth_params()
        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.zeros(1, 32, device=DEVICE)
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        assert out.abs().max().item() < 0.01

    def test_single_step_has_audio_at_correct_position(self):
        """A single active step should place audio at that step's time."""
        synth_params = _make_synth_params()
        seq_pitch = torch.full((1, 32), 0.5, device=DEVICE)
        seq_velocity = torch.zeros(1, 32, device=DEVICE)
        seq_velocity[0, 16] = 0.8  # only step 16 active (middle of bar)
        seq_gate = torch.full((1, 32), 0.5, device=DEVICE)
        seq_timing = torch.zeros(1, 32, device=DEVICE)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        total = out.shape[1]
        first_half_energy = out[0, :total // 2].pow(2).mean()
        second_half_energy = out[0, total // 2:].pow(2).mean()
        # Step 16 is in the second half
        assert second_half_energy > first_half_energy * 5

    def test_no_nan(self):
        synth_params = _make_synth_params()
        seq_pitch = torch.rand(1, 32, device=DEVICE)
        seq_velocity = torch.rand(1, 32, device=DEVICE)
        seq_gate = torch.rand(1, 32, device=DEVICE)
        seq_timing = (torch.rand(1, 32, device=DEVICE) - 0.5)

        out = render_sequence(
            synth_params, seq_pitch, seq_velocity, seq_gate, seq_timing,
            bpm=170.0, sample_rate=SAMPLE_RATE,
        )
        assert not torch.isnan(out).any()
```

- [ ] **Step 2: 运行测试确认失败**

```bash
uv run pytest tests/test_sequencer.py -v
```

- [ ] **Step 3: 实现 sequencer**

`src/loom/sequencer.py`:
```python
import torch
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE


def render_sequence(
    synth_params: dict[str, torch.Tensor],
    seq_pitch: torch.Tensor,
    seq_velocity: torch.Tensor,
    seq_gate: torch.Tensor,
    seq_timing: torch.Tensor,
    bpm: float = 170.0,
    sample_rate: int = SAMPLE_RATE,
    n_steps: int = 32,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Render a 32-step sequence using SubtractiveSynth.

    Args:
        synth_params: Shared synth parameters (batch=1 tensors).
        seq_pitch: (batch, 32) per-step pitch [0,1].
        seq_velocity: (batch, 32) per-step velocity [0,1], 0=rest.
        seq_gate: (batch, 32) per-step gate length [0,1] as fraction of step.
        seq_timing: (batch, 32) per-step micro-timing [-0.5, 0.5] in step units.
        bpm: Tempo in BPM.
        sample_rate: Audio sample rate.
        n_steps: Number of steps (default 32).
        velocity_threshold: Minimum velocity to trigger a note.

    Returns:
        (batch, total_samples) rendered audio.
    """
    batch = seq_pitch.shape[0]
    device = seq_pitch.device
    step_sec = 60.0 / bpm / 8.0  # 32 steps = 4 beats = 1 bar
    total_sec = n_steps * step_sec
    total_samples = int(total_sec * sample_rate)
    output = torch.zeros(batch, total_samples, device=device)

    # Max note length: 2 steps worth of samples (note + release tail)
    max_note_samples = int(step_sec * 2.0 * sample_rate)
    max_note_samples = min(max_note_samples, total_samples)

    for step in range(n_steps):
        # Check which batch items have active notes at this step
        vel = seq_velocity[:, step]  # (batch,)
        active_mask = vel > velocity_threshold  # (batch,)
        if not active_mask.any():
            continue

        # Note timing
        note_start_sec = step * step_sec + seq_timing[:, step] * step_sec
        note_start_sec = note_start_sec.clamp(min=0.0)
        note_start_sample = (note_start_sec * sample_rate).long()

        # Note duration from gate
        gate_sec = seq_gate[:, step] * step_sec
        gate_sec = gate_sec.clamp(min=0.01)

        # Build per-note synth params: override pitch and gain
        note_params = {}
        for key, val in synth_params.items():
            if val.shape[0] == 1 and batch > 1:
                note_params[key] = val.expand(batch, *val.shape[1:])
            else:
                note_params[key] = val.clone()

        note_params["osc_pitch"] = seq_pitch[:, step]
        # Scale master_gain by velocity
        base_gain = synth_params.get("master_gain", torch.tensor([0.8], device=device))
        if base_gain.shape[0] == 1 and batch > 1:
            base_gain = base_gain.expand(batch)
        note_params["master_gain"] = (base_gain * vel).clamp(0.0, 1.0)

        # Render note with appropriate length
        note_synth = SubtractiveSynth(sample_rate, max_note_samples).to(device)
        # Use mean gate for note_on_duration (all batch items share synth structure)
        mean_gate = gate_sec.mean().item()
        # Forward pass
        with torch.no_grad() if not seq_pitch.requires_grad else torch.enable_grad():
            note_audio = note_synth(note_params)  # (batch, max_note_samples)

        # Place each batch item's note at its start position
        for b in range(batch):
            if not active_mask[b]:
                continue
            start = note_start_sample[b].item()
            start = max(0, min(start, total_samples - 1))
            end = min(start + max_note_samples, total_samples)
            length = end - start
            if length > 0:
                output[b, start:end] = output[b, start:end] + note_audio[b, :length]

    return output
```

- [ ] **Step 4: 运行测试确认通过**

```bash
uv run pytest tests/test_sequencer.py -v --timeout=120
```

Expected: 全部 PASS（sequencer 测试可能较慢，每个 step 渲染一次 synth）

- [ ] **Step 5: Commit**

```bash
git add src/loom/sequencer.py tests/test_sequencer.py
git commit -m "feat: 32-step sequencer with per-step pitch/velocity/gate/timing"
```

---

## Task 5: Demo 更新 — 有节奏的 DnB pattern

**Files:**
- Modify: `scripts/demo.py`

- [ ] **Step 1: 添加 sequencer demo**

Read current `scripts/demo.py`. Add import at top:
```python
from loom.sequencer import render_sequence
```

Add after the existing sounds (read the file to find the right spot):

```python
    # --- 9. DnB Drum Pattern (sequencer demo) ---
    print("9. DnB Bass Sequence")
    p = make_base_params()
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    p["filter_cutoff"] = torch.tensor([0.4])
    p["filter_q"] = torch.tensor([0.5])
    p["filt_env_amount"] = torch.tensor([0.7])
    p["dist_amount"] = torch.tensor([0.3])
    p["dist_mix"] = torch.tensor([0.5])

    # Simple DnB bass pattern: hits on 1, 4, 7, 10, 13 (syncopated 8ths)
    seq_pitch = torch.full((1, 32), 0.2)  # low bass
    seq_velocity = torch.zeros(1, 32)
    for step in [0, 6, 8, 14, 16, 22, 24, 30]:  # syncopated
        seq_velocity[0, step] = 0.9
    seq_pitch[0, 8] = 0.25   # variation
    seq_pitch[0, 24] = 0.18  # variation
    seq_gate = torch.full((1, 32), 0.6)
    seq_timing = torch.zeros(1, 32)

    with torch.no_grad():
        audio = render_sequence(p, seq_pitch, seq_velocity, seq_gate, seq_timing, bpm=174.0)
    save_wav(audio[0], "09_dnb_bass_sequence")

    # --- 10. Wobble Bass (LFO demo) ---
    print("10. Wobble Bass (LFO)")
    p = make_base_params()
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    p["osc_pitch"] = torch.tensor([0.2])
    p["filter_cutoff"] = torch.tensor([0.35])
    p["filter_q"] = torch.tensor([0.6])
    p["filt_env_amount"] = torch.tensor([0.5])
    p["dist_amount"] = torch.tensor([0.4])
    p["dist_mix"] = torch.tensor([0.6])
    p["lfo_rate"] = torch.tensor([0.35])     # ~2Hz wobble
    p["lfo_depth"] = torch.tensor([0.9])
    p["lfo_waveform"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # sine LFO
    p["lfo_target"] = torch.tensor([[1.0, 0.0, 0.3, 0.0]])    # cutoff + slight dist
    p["lfo_phase"] = torch.tensor([0.0])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "10_wobble_bass")
```

Also add LFO params to `make_base_params`:
```python
        "lfo_rate": torch.full((batch,), 0.5),
        "lfo_depth": torch.full((batch,), 0.0),
        "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch),
        "lfo_target": torch.zeros(batch, 4),
        "lfo_phase": torch.full((batch,), 0.0),
```

- [ ] **Step 2: Commit**

```bash
git add scripts/demo.py
git commit -m "feat: add DnB bass sequence and wobble bass demos"
```

---

## Task 6: Integration verification

- [ ] **Step 1: Full test suite**

```bash
uv run pytest tests/ -v
```

Expected: 全部 PASS

- [ ] **Step 2: Commit if any uncommitted changes**

```bash
git add -A
git commit -m "feat: Phase 0 batch 3 complete — LFO + sequencer"
```
