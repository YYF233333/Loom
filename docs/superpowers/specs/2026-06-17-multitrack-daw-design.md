# Multitrack Differentiable DAW — Design Spec

**Date**: 2026-06-17
**Status**: Approved
**Scope**: Complete the Loom differentiable engine from single-note single-track to full multitrack DAW with DAG routing, modulation matrix, sequencer, and arrangement.

## Context

The Loom engine currently renders a single 4-second note through a single SubtractiveSynth (3 oscillator types, SVFilter, ADSR, 6-effect chain with Sinkhorn routing). Training pipeline is validated (curriculum stages 0-3, pool rotation, multi-res STFT + param + signal-chain loss).

This spec extends the engine to a complete multitrack DAW while maintaining end-to-end differentiability. The engine is built first ("engine-first" strategy); the training side controls complexity via curriculum — the engine's full capability is a superset that the model gradually unlocks.

Sampler module is deferred — sampled sounds use the embedding retrieval path (CLAP/FAISS), not the differentiable engine.

## 1. Signal Flow & Routing

### Node Types

| Node | Internal Structure | Input | Output |
|------|-------------------|-------|--------|
| **Track** | Synth → ModMatrix → InsertFX → Fader/Pan | None (self-generates audio) | direct out + send outs |
| **Bus** (Group/Aux) | InsertFX → Fader/Pan | Mixed send/direct from tracks | direct out → higher bus or master |
| **Master** | InsertFX → Fader | All buses + direct-connected tracks | Final output |

### Routing Rules (DAG Constraints)

- **Track → Bus**: Allowed (direct out or send with amount)
- **Track → Master**: Allowed (direct out)
- **Bus → Bus**: Allowed (must be higher-level, no cycles)
- **Bus → Master**: Allowed
- **Track → Track**: **Forbidden** — buses exist as mixing nodes; track interconnection would make buses redundant.

### Routing Data Structures

```python
# Nodes ordered: [Track_0, ..., Track_N, Bus_0, ..., Bus_M, Master]
# Ordering itself enforces DAG constraint — no mask needed.

send_matrix: (N_tracks, M_buses)   # each track's send amount to each bus, [0,1]
direct_to_master: (N_tracks,)      # each track's direct-out level to master
bus_to_master: (M_buses,)          # each bus's output level to master
```

### Sidechain

Sidechain is NOT a routing feature — it is a **Compressor module feature**. The compressor has an optional `sidechain` input for its envelope detector.

Implementation pattern (matches REAPER/Ableton):
1. Track A (kick) sends to a "SC bus" at some send level
2. The SC bus fader is at 0 — it contributes no audio to the mix
3. A compressor on Track B (bass) or a group bus references the SC bus as its `sidechain` input
4. The compressor's RMS envelope detector uses the SC bus signal instead of the main signal

```python
class Compressor(nn.Module):
    def forward(self, signal, threshold, ratio, attack, release, makeup, mix,
                sidechain=None):
        detector_input = sidechain if sidechain is not None else signal
        rms = self._rms_envelope(detector_input)
        # gain reduction applied to `signal`, not `sidechain`
        ...
```

## 2. Modulation Matrix

### Architecture

Multiple LFO sources, each can simultaneously modulate multiple targets with independent amounts.

- **S = 4 sources**: 2 general-purpose LFOs + 1 envelope follower + 1 spare
- **T = all continuous parameters** (~42 per track from `CONTINUOUS_KEYS`)
- **mod_matrix**: `(batch, S, T)` with values in `[-1, 1]`, 0 = no connection

### Per-Source Parameters

```python
lfo_params = {
    "rate":     (batch, S),      # [0,1] → [0.1, 20] Hz per source
    "depth":    (batch, S),      # global depth scaling per source
    "waveform": (batch, S, 4),   # sine/saw/square/tri blend per source
    "phase":    (batch, S),      # initial phase offset per source
}
# 28 scalar params total (4 sources × 7 params)
```

### Runtime Computation

```python
# Generate LFO signals: (batch, S, n_samples)
lfo_signals = render_all_lfos(lfo_params)

# Envelope follower replaces one LFO slot:
lfo_signals[:, ENV_FOLLOWER_IDX, :] = rms_envelope(input_audio)

# Apply modulation matrix: (batch, T, n_samples)
modulation = torch.einsum("bst,bsn->btn", mod_matrix, lfo_signals)

# Add to base parameters (per-sample):
modulated_cutoff = (base_cutoff + modulation[:, CUTOFF_IDX, :]).clamp(0, 1)
```

### Sparsity

Most matrix entries will be ~0 (a track rarely uses all 4 LFOs on all 42 targets). Training data generation samples sparse matrices (e.g., 1-3 active connections per LFO).

## 3. Sequencer & Arrangement

### Hierarchy

```
Arrangement
├── bpm: float
├── time_signature: (int, int)
├── sections: [Section, ...]
│   └── Section
│       ├── start_bar: int
│       ├── length_bars: int
│       └── track_patterns: {track_idx: Pattern}
│           └── Pattern
│               ├── length_bars: int
│               └── events: Tensor (N_max_notes, 4)
```

### Note Event Representation (MIDI-Compatible Tensor)

```python
note_events: (batch, N_max_notes, 4)
# [:, :, 0] = pitch     — MIDI note number [0, 127] normalized to [0, 1]
# [:, :, 1] = velocity  — [0, 1] (MIDI 0-127); velocity=0 marks empty slot
# [:, :, 2] = onset     — beat position (float, 0.0 = first beat)
# [:, :, 3] = duration  — in beats (float)
```

Empty slots: `velocity = 0` (MIDI convention: velocity 0 = note off).

### MIDI Import/Export

```python
def from_midi(midi_path: str) -> NoteEvents:
    """mido MidiFile → tensor representation."""

def to_midi(events: Tensor, bpm: float) -> mido.MidiFile:
    """Tensor → standard MIDI file for REAPER validation."""
```

### Pattern Rendering (Differentiable)

```python
def render_pattern(synth, synth_params, note_events, bpm, total_samples):
    pattern_audio = torch.zeros(batch, total_samples)
    samples_per_beat = 60.0 / bpm * sample_rate

    for note_idx in range(N_max_notes):
        vel = note_events[:, note_idx, 1]          # (batch,)
        if vel.max() < 1e-6:
            continue  # skip empty slots

        pitch = note_events[:, note_idx, 0]
        onset = note_events[:, note_idx, 2]
        dur = note_events[:, note_idx, 3]

        # Override pitch in synth_params for this note
        note_params = {**synth_params, "osc_pitch": pitch}
        note_audio = synth(note_params)             # (batch, note_samples)

        # Time-place using differentiable fractional shift (grid_sample)
        offset_samples = onset * samples_per_beat
        placed = fractional_shift(note_audio, offset_samples, total_samples)
        pattern_audio += placed * vel.unsqueeze(1)

    return pattern_audio
```

Key: notes are summed BEFORE the effects chain — real DAW behavior. Nonlinear effects (distortion, compression) must see the combined signal.

## 4. Full Render Pipeline

### Render Order

```
① Per-track: pattern → note sum → mod_matrix → insert FX → fader/pan
② Aggregate sends to buses
③ Per-bus: sum inputs → insert FX (with optional sidechain) → fader/pan
④ Master: sum direct-outs + bus outputs → insert FX → fader
⑤ Output
```

### Gradient Flow

```
Loss(output, target)
  ↓
Master FX ← gradients through master effects
  ├─ Bus FX ← gradients through bus effects (incl. sidechain compressor)
  │    └─ send_amount ← gradients to send weights
  └─ Track FX ← gradients through per-track effects
       ├─ Fader (level, pan) ← simple multiplication
       ├─ Insert FX chain ← Sinkhorn-routed, already validated
       ├─ Mod Matrix ← einsum, natural gradient flow
       ├─ Note Events (pitch, vel, onset, dur) ← grid_sample shift
       └─ Synth Params ← existing oscillator/filter/envelope
```

Every step is differentiable — no discrete decisions. Gradients flow from final loss back to every note's pitch/velocity and every synth parameter.

## 5. Parameter Space

### Per-Track (~440 params)

| Component | Count | Notes |
|-----------|-------|-------|
| Synth params | ~50 | Existing continuous + categorical |
| Mod matrix | ~196 | S=4 × T=42 amounts + 4×7 source params |
| Effects chain | ~59 | Existing FX params + 6×6 Sinkhorn routing |
| Pattern | ~128 | 32 notes × 4 (pitch/vel/onset/dur) |
| Fader + sends | ~6 | level, pan, send×4 |

### Session Total (8 tracks)

| Component | Count |
|-----------|-------|
| 8 tracks × 440 | ~3520 |
| 3 buses × ~65 (FX + fader) | ~195 |
| Master (FX + fader) | ~65 |
| Global (BPM, time sig, section layout) | ~18 |
| **Total** | **~3800** |

### Curriculum Expansion

| Stage | What's Active | Approx Params |
|-------|--------------|---------------|
| 0-3 | Single note, single track (current) | ~90 |
| 4 | Single track + 2-4 note pattern | ~220 |
| 5 | Single track + full pattern (32 notes) + mod matrix | ~440 |
| 6 | 2-4 tracks + patterns + mixing | ~1800 |
| 7 | Full session: all tracks + buses + sections | ~3800 |

Inactive parameters are frozen at defaults and excluded from loss computation.

## 6. Key Implementation Notes

### What Exists (Reuse)

- `SubtractiveSynth` with 3 oscillator types — becomes the per-track synth
- `EffectsChain` with Sinkhorn routing — one instance per track/bus/master
- All 6 effects (distortion, compressor, chorus, delay, reverb, EQ) — unchanged
- `SVFilter` with time-varying cutoff — unchanged
- `ADSR` envelope — unchanged
- `LFO` module — extended from 1 to S=4 sources
- `grid_sample` fractional shift — reused from Delay/Chorus for note placement
- Training pipeline (pool rotation, curriculum, losses) — extended with new stages

### What's New

- `Session` top-level module orchestrating tracks/buses/master
- `TrackNode` wrapping synth + mod matrix + FX + fader + sends
- `BusNode` wrapping FX + fader + optional sidechain reference
- `MasterNode` wrapping FX + fader
- `ModulationMatrix` — S sources × T targets, runtime `einsum`
- `Sequencer` — pattern rendering with `fractional_shift`
- `Arrangement` — section layout, MIDI import/export
- `Compressor.sidechain` input parameter
- Extended `random_params` / `params_to_vector` for full session
- Curriculum stages 4-7

## 7. Testing Strategy

Every new module must have its numerical invariants defined BEFORE implementation. Tests pass = module done.

### A. Routing Correctness (highest priority — wrong signal flow invalidates everything)

| Test | Method | Assertion |
|------|--------|-----------|
| send=0 → bus silent | Single track send_amount=0 | Bus input RMS == 0 |
| send=1 → full signal | send_amount=1.0 | Bus input ≈ track output (atol=1e-5) |
| send linear superposition | 2 tracks, send=0.5, same-freq sine | Bus input amplitude = single × 2 × 0.5 |
| direct_to_master correct | Single track direct=1, no bus | Master input == track output |
| bus fader=0 silent | Bus level=0 (SC bus scenario) | Master does not contain bus signal |
| Track→Track forbidden | Construct illegal route | Raises exception |
| DAG acyclicity | Bus A→Bus B→Bus A | Raises exception |
| Multi-bus independence | 2 buses, different tracks each | No crosstalk |

### B. Sidechain Compressor

| Test | Method | Assertion |
|------|--------|-----------|
| sidechain=None backward compat | Compare with existing Compressor | Bit-exact |
| sidechain controls gain reduction | Kick SC → bass | Bass gain drops > 3dB at kick transients |
| sidechain doesn't bleed audio | Kick SC → bass | Bass output has no kick frequency content |
| sidechain strength controllable | Different thresholds | Gain reduction monotonically increases as threshold decreases |

### C. Modulation Matrix

| Test | Method | Assertion |
|------|--------|-----------|
| All-zero matrix = no modulation | matrix=0 | Output == unmodulated output |
| Single connection correct | LFO0 → cutoff, amount=0.3 | Spectral centroid oscillates at LFO frequency |
| Multiple connections stack | LFO0 → cutoff + pitch | Both targets modulated |
| Amount polarity | amount=+0.5 vs -0.5 | Opposite modulation direction |
| Clamp within bounds | amount=1.0, base=0.9 | Modulated param ≤ 1.0 |
| Envelope follower source | env follower → cutoff | Cutoff high during signal, low during silence |
| Gradient through matrix | grad w.r.t. mod_matrix | Non-zero gradients |

### D. Sequencer / Note Placement

| Test | Method | Assertion |
|------|--------|-----------|
| Single note at onset=0 | onset=0.0, 170BPM | Energy starts at sample 0 |
| Offset note position | onset=2.0 beats | Energy starts at 2×(60/170)×44100 samples ± 10 |
| Velocity scaling | vel=0.5 vs vel=1.0 | RMS ratio ≈ 0.5 ± 0.05 |
| velocity=0 silent | vel=0 | Note contributes RMS == 0 |
| Multi-note superposition | 2 notes, same pitch, different onsets | Energy at both time points |
| Notes sum before FX | 2 notes + distortion | Output ≠ distort(note1) + distort(note2) |
| MIDI round-trip | tensor → to_midi → from_midi → tensor | pitch/vel/onset/dur error < 1e-4 |

### E. Gradient Health (Multi-Track Level)

| Test | Method | Assertion |
|------|--------|-----------|
| Full pipeline gradients exist | 3 tracks + 1 bus + master, loss.backward() | All track params have non-zero grad |
| No NaN/Inf gradients | Random params, full pipeline | No anomalous values |
| Gradient magnitude ratios | Log grad norms per param group | max/min ratio < 10000 |
| send_amount gradient | grad w.r.t. send_matrix | Non-zero |
| note onset gradient | grad w.r.t. onset | Non-zero (validates grid_sample shift differentiability) |
| Deep chain gradient survival | 8 tracks + 3 buses + master, all FX active | Bottom-layer osc_pitch grad norm > 1e-8 |

### F. Regression Protection (Golden Snapshots)

Fixed-seed renders saved as golden tensors. Any engine change that alters output breaks the test.

```python
def test_session_golden():
    torch.manual_seed(42)
    output = render_session(fixed_session_params)
    golden = torch.load("tests/fixtures/golden/session_3track_1bus.pt")
    assert torch.allclose(output, golden, atol=1e-5)
```

New golden snapshots added as each module is completed. `pytest --update-golden` regenerates.

### FM Frequency Caveat (from Deep Research)

Gradient-based optimization of frequency/FM parameters is fundamentally unreliable (Hayes et al. ICASSP 2023, 6-0 verified). Loom mitigates this because the encoder network predicts pitch (neural network frequency prediction, the DDSP-validated approach), not direct gradient optimization from random init. However, FM `mod_index` gradients are noisy above ~1.83 — training data should bias `fm_mod_index` sampling toward lower values in early curriculum stages.
