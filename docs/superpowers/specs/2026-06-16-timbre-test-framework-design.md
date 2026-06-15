# Timbre Test Framework Design

## Problem

The synth engine produces incorrect timbres across all layers (oscillators, filters, envelopes, effects). The existing test suite only covers smoke tests (shape, no NaN, non-silence). There is no verification that the engine produces acoustically correct output for known parameters.

## Goal

Build a layered test framework that locks down functional correctness at every level of the synthesis chain, using standardized timbres and measurable acoustic properties.

## Verification Strategy — Three Tiers

1. **Mathematically exact**: For modules with closed-form solutions (oscillator frequencies, harmonic amplitudes, envelope timing, LFO waveforms), assert numerical values with tight tolerances.
2. **Golden audio snapshots**: For full-chain presets, save rendered output as golden files. Any change to engine internals that alters output is caught as a regression.
3. **External reference comparison**: Render equivalent patches in Serum/Sylenth1, compare mel-spectrogram distance and spectral features.

## File Structure

```
tests/
  fixtures/
    reference/           # Serum/Sylenth1 exported wav files
    golden/              # Engine golden snapshots (.pt tensors)
  timbre_helpers.py      # Spectral analysis utility functions
  test_timbres_osc.py    # Oscillator layer acoustic tests
  test_timbres_filter.py # Filter layer acoustic tests
  test_timbres_envelope.py # Envelope & LFO layer tests
  test_timbres_effects.py  # Effects chain tests
  test_timbres_presets.py  # Full-chain preset golden + reference tests
```

## Part 1: Test Utilities — `timbre_helpers.py`

Core functions for acoustic assertions. All accept numpy array or torch tensor, compute in numpy (no gradients needed). Dependencies: `numpy`, `scipy.signal` only.

| Function | Signature | Returns | Purpose |
|----------|-----------|---------|---------|
| `fundamental_freq` | `(audio, sr)` | `float` (Hz) | FFT peak extraction |
| `harmonic_amplitudes` | `(audio, sr, f0, n)` | `(n,)` dB | First n harmonic levels |
| `spectral_centroid` | `(audio, sr)` | `float` (Hz) | Spectral center of mass |
| `spectral_rolloff` | `(audio, sr, pct=0.85)` | `float` (Hz) | Energy cutoff frequency |
| `envelope_shape` | `(audio, sr)` | `(attack_ms, peak, sustain_level, release_ms)` | ADSR shape from amplitude |
| `mel_spectrogram_distance` | `(a, b, sr)` | `float` | Mel-spec L1 distance |
| `freq_response` | `(filter_fn, sr, n_points)` | `(freqs, magnitudes_db)` | White noise frequency response |
| `rms_envelope` | `(audio, hop)` | `(n_frames,)` | Frame-level RMS |
| `thd` | `(audio, sr, f0)` | `float` | Total harmonic distortion ratio |

## Part 2: Oscillator Layer — `test_timbres_osc.py`

Test each oscillator in isolation, bypassing filter/envelope/effects.

### AdditiveOscillator

| Test | Parameters | Assertion | Precision |
|------|-----------|-----------|-----------|
| Pure sine A4 | pitch=A4, waveform=[1,0,0,0] | `fundamental_freq` == 440Hz +/- 1Hz, harmonics 2-8 < -60dB | exact |
| Pure sine C2 | pitch=C2 | `fundamental_freq` == 65.41Hz +/- 0.5Hz | exact |
| Saw harmonic decay | waveform=[0,1,0,0] | Harmonic n amplitude ~ 1/n, error < 1dB | exact |
| Square odd-only harmonics | waveform=[0,0,1,0] | Even harmonics < -50dB, odd ~ 1/n | exact |
| Triangle harmonic decay | waveform=[0,0,0,1] | Odd harmonics ~ 1/n^2, even < -50dB | exact |
| Detune beat frequency | detune=0.6 (20 cents) | Frequency shift matches theory +/- 0.5Hz | exact |
| Amplitude range | all waveforms | peak in [-1.05, 1.05] | exact |

### WavetableOscillator

| Test | Parameters | Assertion |
|------|-----------|-----------|
| position=0.0 ~ saw | pos=0 | Harmonic distribution matches additive saw < 2dB |
| position=1.0 ~ square | pos=1 | Harmonic distribution matches additive square < 2dB |
| Mid-position smooth morph | pos=0.5 | spectral_centroid between saw and square |
| Correct fundamental | multiple pitches | `fundamental_freq` error < 1Hz |

### FMOscillator

| Test | Parameters | Assertion |
|------|-----------|-----------|
| mod_index=0 degenerates to sine | idx=0, c_ratio=1 | Only fundamental, THD < -40dB |
| 1:1 ratio symmetric sidebands | c=1, m=1, idx=0.1 | Sidebands at f0 +/- fm, amplitude ~ J1 approximation |
| 1:2 ratio harmonic series | c=1, m=2 | Harmonics at f0, 2f0, 3f0... |
| High mod_index spectral spread | idx=0.5 vs idx=0.1 | Higher spectral_centroid, more harmonics |

## Part 3: Filter Layer — `test_timbres_filter.py`

Test SVFilter using white noise input, measure frequency response.

### Frequency Response Tests

| Test | Parameters | Assertion | Precision |
|------|-----------|-----------|-----------|
| LP cutoff accuracy | cutoff=1000Hz, Q=0.707 | -3dB point == 1000Hz +/- 50Hz | spectral |
| LP rolloff slope | cutoff=1000Hz | -12dB/oct +/- 2dB above cutoff (2nd order SVF) | spectral |
| HP cutoff accuracy | type=HP, cutoff=1000Hz | -3dB point == 1000Hz +/- 50Hz | spectral |
| HP rolloff slope | type=HP | -12dB/oct +/- 2dB below cutoff | spectral |
| BP center frequency | type=BP, cutoff=1000Hz | Peak == 1000Hz +/- 50Hz | spectral |
| BP bandwidth vs Q | Q=0.3 vs Q=0.8 | Higher Q = narrower -3dB bandwidth | spectral |
| Q resonance peak | Q=0.9 | Gain > 3dB at cutoff | spectral |
| Time-varying cutoff sweep | cutoff linear 0.1->0.9 | Spectral centroid rises over time | spectral |
| Extreme params stable | cutoff=0.01, Q=0.99 | No NaN, bounded energy | stability |

### Filter Envelope Interaction

| Test | Parameters | Assertion |
|------|-----------|-----------|
| env amount=0 no modulation | filt_env_amount=0.5 (neutral) | Output matches static cutoff |
| Positive env modulation | amount=0.8 | Attack spectral_centroid > sustain spectral_centroid |
| Negative env modulation | amount=0.2 | Attack spectral_centroid < sustain spectral_centroid |

### Parameter Denormalization

| Test | Input | Assertion |
|------|-------|-----------|
| cutoff=0.0 -> 20Hz | norm=0.0 | `_denorm_cutoff` == 20.0 +/- 0.1 |
| cutoff=1.0 -> 20000Hz | norm=1.0 | `_denorm_cutoff` == 20000.0 +/- 1.0 |
| cutoff=0.5 -> ~632Hz | norm=0.5 | Log midpoint sqrt(20*20000) +/- 5Hz |
| Q=0.0 -> 0.5 | norm=0.0 | `_denorm_q` == 0.5 +/- 0.01 |
| Q=1.0 -> 20.0 | norm=1.0 | `_denorm_q` == 20.0 +/- 0.1 |

## Part 4: Envelope & LFO Layer — `test_timbres_envelope.py`

### ADSR Envelope

| Test | Parameters | Assertion | Precision |
|------|-----------|-----------|-----------|
| Attack time accurate | attack=0.5 (->54ms) | Time 0->0.95 == 54ms +/- 5ms | exact |
| attack=0.0 fastest | attack=0.0 (->1ms) | Peak time < 3ms | exact |
| attack=1.0 slowest | attack=1.0 (->2000ms) | Peak time == 2000ms +/- 50ms | exact |
| Decay to sustain | decay=0.5, sustain=0.6 | Reaches 0.6 +/- 0.05 after decay | exact |
| Sustain holds flat | sustain=0.7 | Sustain segment std < 0.02 | exact |
| Release to zero | release=0.3 | < 0.05 after release time | exact |
| sustain=0 AD-only | sustain=0.0 | Envelope ~ 0 after decay | exact |
| sustain=1 no decay | sustain=1.0 | Holds 1.0 until release | exact |
| Denorm mapping table | multiple values | `_denorm_time` matches manual calculation | exact |

### LFO

| Test | Parameters | Assertion | Precision |
|------|-----------|-----------|-----------|
| Sine LFO frequency | rate=0.5, sine | FFT peak == theory +/- 0.1Hz | exact |
| LFO amplitude range | depth=0.8 | Output in [-0.8, 0.8] +/- 0.01 | exact |
| depth=0 silent | depth=0.0 | All zeros | exact |
| Saw LFO shape | saw waveform | Linear ramp + drop, verified by derivative | exact |
| Square LFO shape | square waveform | Only +depth and -depth values | exact |
| Triangle LFO shape | tri waveform | Symmetric linear rise/fall | exact |
| Phase offset | phase=0.5 (->pi) | Sine starts at ~0, half-cycle offset from phase=0 | exact |
| Rate denorm range | rate=0.0, rate=1.0 | 0.1Hz +/- 0.01, 20Hz +/- 0.1 | exact |

### LFO Target Routing

| Test | lfo_target | Assertion |
|------|-----------|-----------|
| Cutoff only | [1,0,0,0] | Spectral centroid oscillates at LFO frequency |
| Pitch only | [0,1,0,0] | Fundamental has periodic vibrato, cutoff unchanged |
| Drive only | [0,0,1,0] | THD oscillates over time, fundamental stable |
| Zero target | [0,0,0,0] | Output identical to depth=0 |

## Part 5: Effects Chain — `test_timbres_effects.py`

Each effect tested in isolation with known input (sine wave or white noise).

### Distortion

| Test | Parameters | Assertion |
|------|-----------|-----------|
| amount=0, mix=0 passthrough | drive=0, mix=0 | Bit-exact input == output |
| mix=0 bypass | any drive, mix=0 | Output == input |
| Tanh soft clip | drive=0.5, mix=1, sine input | THD > 5%, harmonics present |
| Heavy clip | drive=0.9, mix=1 | Output approaches square, peak clamped near +/-1 |
| Drive monotonic THD | drive=0.2 vs 0.5 vs 0.8 | THD monotonically increasing |

### Compressor

| Test | Parameters | Assertion |
|------|-----------|-----------|
| mix=0 bypass | mix=0 | Output == input |
| Dynamic range reduction | threshold=0.3, ratio=0.6 | Output dynamic range < input dynamic range |
| Higher ratio = more compression | ratio=0.2 vs 0.8 | Higher ratio = smaller dynamic range |
| Makeup gain | makeup=0.5, mix=1 | Output RMS > input RMS |
| No spectral coloring | mix=1 | spectral_centroid change < 10% |

### Chorus

| Test | Parameters | Assertion |
|------|-----------|-----------|
| mix=0 bypass | mix=0 | Output == input |
| Spectral widening | mix=0.5, depth=0.6 | Sideband spread around fundamental |
| Preserves fundamental | mix=0.5 | fundamental_freq unchanged +/- 2Hz |

### Delay

| Test | Parameters | Assertion |
|------|-----------|-----------|
| mix=0 bypass | mix=0 | Output == input |
| Echo position correct | time=0.5, fb=0.3, mix=0.5 | RMS envelope peak at expected delay time |
| Feedback = multiple echoes | feedback=0.6 | At least 2+ echo peaks detected |
| feedback=0 single echo | feedback=0 | Only one echo |

### Reverb

| Test | Parameters | Assertion |
|------|-----------|-----------|
| mix=0 bypass | mix=0 | Output == input |
| Adds tail energy | mix=0.5 | Energy persists after input ends |
| Room size affects tail length | room=0.3 vs 0.8 | Larger room = longer RT60 |
| Damping affects high freq | damping=0.2 vs 0.8 | Higher damping = lower tail spectral_centroid |

### EQ

| Test | Parameters | Assertion |
|------|-----------|-----------|
| Neutral passthrough | low/mid/high=0.5 | Output ~ input, error < -40dB |
| Low boost | low=0.8 | < 300Hz energy increase > 3dB |
| High cut | high=0.2 | > 5kHz energy decrease > 3dB |
| Band independence | change mid only | Low and high band energy change < 1dB |

## Part 6: Full-Chain Presets — `test_timbres_presets.py`

### Preset Onboarding Workflow

Presets are NOT bulk-defined upfront. Each preset goes through an individual onboarding process:

1. **Design preset parameters** — define the target timbre and set synth params
2. **Render & listen** — render from the engine, audition for correctness
3. **Serum recreation** — recreate the equivalent patch in Serum, render reference wav
4. **Lock both standards** — save engine golden snapshot (.pt) + Serum reference (.wav)
5. **Add to test suite** — write golden comparison test + acoustic assertions + reference comparison

This means `test_timbres_presets.py` starts empty and grows as presets are individually validated. Each preset gets its own commit with both reference files and test code.

### Per-Preset Test Template

When a preset is onboarded, it gets three layers of testing:

1. **Golden snapshot**: `torch.allclose(rendered, golden, atol=1e-5)`
2. **Acoustic assertions**: preset-specific spectral/temporal properties (e.g., fundamental range, THD, LFO modulation signature)
3. **Serum reference comparison**: mel-spectrogram distance or spectral feature comparison against the Serum-rendered wav

Example acoustic assertions per preset type:

| Type | Example Assertions |
|------|-------------------|
| Sub bass | fundamental < 80Hz, spectral_centroid < 200Hz |
| Reese bass | fundamental < 120Hz, beat frequency sidebands present |
| Neuro bass | THD > 10%, RMS envelope modulated at LFO frequency |
| Wobble bass | spectral_centroid oscillates periodically |
| FM e-piano | Non-harmonic sidebands, attack < 50ms |
| Pad | spectral_centroid in expected range for waveform type |

### Comparison Metrics (selected per preset during onboarding)

| Type | Metric | Threshold |
|------|--------|-----------|
| Clean timbres (sub bass, pad) | mel-spec L1 distance | < 0.15 |
| Distorted timbres (neuro, reese) | spectral_centroid difference | < 20%, harmonic correlation > 0.8 |
| Modulated timbres (wobble, neuro) | LFO modulation frequency match | +/- 0.5Hz, depth difference < 25% |
| Effect-heavy (delay lead) | Dry portion mel-spec L1 | < 0.2 |

### CLI

```
pytest tests/test_timbres_presets.py --update-golden   # regenerate all golden files
pytest -m "not reference"                               # skip Serum reference tests
```

### File Layout

```
tests/fixtures/
  golden/
    01_sub_bass.pt           # added when preset 01 is onboarded
    02_reese_bass.pt         # added when preset 02 is onboarded
    ...
  reference/
    serum_01_sub_bass.wav    # added when preset 01 is onboarded
    serum_02_reese_bass.wav  # added when preset 02 is onboarded
    ...
```

## Implementation Order

1. `timbre_helpers.py` — utility functions
2. `test_timbres_osc.py` — oscillator layer (find and fix oscillator bugs first)
3. `test_timbres_envelope.py` — envelope and LFO (second layer)
4. `test_timbres_filter.py` — filter layer (depends on known-good input)
5. `test_timbres_effects.py` — effects chain (depends on known-good dry signal)
6. `test_timbres_presets.py` — full-chain golden + reference (after all modules pass)

This order ensures each layer is verified before building on it. Bugs found at lower layers are fixed before testing higher layers.
