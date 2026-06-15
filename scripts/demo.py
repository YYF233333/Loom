"""Demo: render a few synth sounds and save as .wav files.

Usage:
    uv run python scripts/demo.py
"""

import torch
import numpy as np
from scipy.io import wavfile
from loom.synth import SubtractiveSynth
from loom.sequencer import render_sequence
from loom.core import SAMPLE_RATE

N_SAMPLES = SAMPLE_RATE * 4  # 4 seconds


def make_base_params(batch=1):
    """All params at neutral/bypass defaults."""
    return {
        "osc_pitch": torch.full((batch,), 0.5),
        "osc_waveform": torch.tensor([[0.0, 1.0, 0.0, 0.0]] * batch),  # saw
        "osc_detune": torch.full((batch,), 0.5),  # no detune
        "osc_type": torch.tensor([[1.0, 0.0, 0.0]] * batch),
        "wt_position": torch.full((batch,), 0.5),
        "fm_carrier_ratio": torch.full((batch,), 0.0),
        "fm_mod_ratio": torch.full((batch,), 0.0),
        "fm_mod_index": torch.full((batch,), 0.0),
        "lfo_rate": torch.full((batch,), 0.5),
        "lfo_depth": torch.full((batch,), 0.0),
        "lfo_waveform": torch.tensor([[1.0, 0.0, 0.0, 0.0]] * batch),
        "lfo_target": torch.zeros(batch, 4),
        "lfo_phase": torch.full((batch,), 0.0),
        "amp_attack": torch.full((batch,), 0.2),
        "amp_decay": torch.full((batch,), 0.3),
        "amp_sustain": torch.full((batch,), 0.8),
        "amp_release": torch.full((batch,), 0.3),
        "filter_cutoff": torch.full((batch,), 0.5),
        "filter_q": torch.full((batch,), 0.3),
        "filter_type": torch.tensor([[1.0, 0.0, 0.0]] * batch),  # LP
        "filt_env_attack": torch.full((batch,), 0.2),
        "filt_env_decay": torch.full((batch,), 0.4),
        "filt_env_sustain": torch.full((batch,), 0.3),
        "filt_env_release": torch.full((batch,), 0.3),
        "filt_env_amount": torch.full((batch,), 0.7),
        "dist_amount": torch.full((batch,), 0.0),
        "dist_mix": torch.full((batch,), 0.0),
        "master_gain": torch.full((batch,), 0.85),
        "comp_threshold": torch.full((batch,), 0.5),
        "comp_ratio": torch.full((batch,), 0.3),
        "comp_attack": torch.full((batch,), 0.5),
        "comp_release": torch.full((batch,), 0.5),
        "comp_makeup": torch.full((batch,), 0.0),
        "comp_mix": torch.full((batch,), 0.0),
        "chorus_rate": torch.full((batch,), 0.5),
        "chorus_depth": torch.full((batch,), 0.5),
        "chorus_mix": torch.full((batch,), 0.0),
        "delay_time": torch.full((batch,), 0.5),
        "delay_feedback": torch.full((batch,), 0.3),
        "delay_mix": torch.full((batch,), 0.0),
        "reverb_room_size": torch.full((batch,), 0.5),
        "reverb_decay": torch.full((batch,), 0.5),
        "reverb_damping": torch.full((batch,), 0.3),
        "reverb_mix": torch.full((batch,), 0.0),
        "eq_low_gain": torch.full((batch,), 0.5),
        "eq_mid_gain": torch.full((batch,), 0.5),
        "eq_high_gain": torch.full((batch,), 0.5),
    }


def save_wav(audio: torch.Tensor, name: str):
    audio = audio.detach().cpu().float()
    peak = audio.abs().max().clamp(min=1e-6)
    audio = (audio / peak * 0.9).numpy()
    audio_16 = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    wavfile.write(f"output/{name}.wav", SAMPLE_RATE, audio_16)
    print(f"  -> output/{name}.wav ({len(audio_16) / SAMPLE_RATE:.1f}s)")


def main():
    import os
    os.makedirs("output", exist_ok=True)

    synth = SubtractiveSynth(SAMPLE_RATE, N_SAMPLES)
    print(f"Rendering at {SAMPLE_RATE}Hz, {N_SAMPLES/SAMPLE_RATE:.0f}s per clip\n")

    # --- 1. Reese Bass (DnB staple): detuned saw, low pitch, LP filter ---
    print("1. Reese Bass")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.15])          # ~E1, low
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # saw
    p["osc_detune"] = torch.tensor([0.65])          # slight detune up -> beating
    p["filter_cutoff"] = torch.tensor([0.35])       # low cutoff
    p["filter_q"] = torch.tensor([0.5])             # moderate resonance
    p["filt_env_amount"] = torch.tensor([0.65])     # filter opens with envelope
    p["dist_amount"] = torch.tensor([0.2])
    p["dist_mix"] = torch.tensor([0.3])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "01_reese_bass")

    # --- 2. Supersaw Lead: saw wave, mid pitch, chorus ---
    print("2. Supersaw Lead")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.55])           # ~A3
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # saw
    p["osc_detune"] = torch.tensor([0.6])
    p["filter_cutoff"] = torch.tensor([0.65])
    p["filt_env_amount"] = torch.tensor([0.6])
    p["chorus_rate"] = torch.tensor([0.4])
    p["chorus_depth"] = torch.tensor([0.6])
    p["chorus_mix"] = torch.tensor([0.5])
    p["reverb_room_size"] = torch.tensor([0.4])
    p["reverb_decay"] = torch.tensor([0.5])
    p["reverb_damping"] = torch.tensor([0.3])
    p["reverb_mix"] = torch.tensor([0.3])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "02_supersaw_lead")

    # --- 3. Square Pad: square wave, slow attack, reverb ---
    print("3. Square Pad")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.45])           # ~F3
    p["osc_waveform"] = torch.tensor([[0.0, 0.0, 1.0, 0.0]])  # square
    p["amp_attack"] = torch.tensor([0.6])           # slow attack
    p["amp_decay"] = torch.tensor([0.5])
    p["amp_sustain"] = torch.tensor([0.9])
    p["filter_cutoff"] = torch.tensor([0.4])
    p["filter_q"] = torch.tensor([0.2])
    p["filt_env_amount"] = torch.tensor([0.55])
    p["reverb_room_size"] = torch.tensor([0.7])
    p["reverb_decay"] = torch.tensor([0.7])
    p["reverb_damping"] = torch.tensor([0.4])
    p["reverb_mix"] = torch.tensor([0.5])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "03_square_pad")

    # --- 4. Distorted Neuro Bass: saw + heavy distortion + compression ---
    print("4. Neuro Bass")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.2])            # low
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # saw
    p["filter_cutoff"] = torch.tensor([0.45])
    p["filter_q"] = torch.tensor([0.6])
    p["filt_env_amount"] = torch.tensor([0.7])
    p["dist_amount"] = torch.tensor([0.7])          # heavy distortion
    p["dist_mix"] = torch.tensor([0.8])
    p["comp_threshold"] = torch.tensor([0.3])
    p["comp_ratio"] = torch.tensor([0.6])
    p["comp_makeup"] = torch.tensor([0.4])
    p["comp_mix"] = torch.tensor([0.7])
    p["eq_low_gain"] = torch.tensor([0.7])          # boost lows
    p["eq_high_gain"] = torch.tensor([0.3])         # cut highs
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "04_neuro_bass")

    # --- 5. Clean Sine Sub: pure sine, low ---
    print("5. Sub Bass (sine)")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.1])            # very low
    p["osc_waveform"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])  # sine
    p["filter_cutoff"] = torch.tensor([0.3])
    p["filter_q"] = torch.tensor([0.2])
    p["amp_attack"] = torch.tensor([0.15])
    p["amp_sustain"] = torch.tensor([0.9])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "05_sub_bass")

    # --- 6. Delay Lead: triangle + delay ---
    print("6. Delay Lead")
    p = make_base_params()
    p["osc_pitch"] = torch.tensor([0.6])            # ~C4
    p["osc_waveform"] = torch.tensor([[0.0, 0.0, 0.0, 1.0]])  # triangle
    p["amp_attack"] = torch.tensor([0.1])
    p["amp_decay"] = torch.tensor([0.4])
    p["amp_sustain"] = torch.tensor([0.3])
    p["amp_release"] = torch.tensor([0.5])
    p["filter_cutoff"] = torch.tensor([0.55])
    p["filt_env_amount"] = torch.tensor([0.6])
    p["delay_time"] = torch.tensor([0.5])
    p["delay_feedback"] = torch.tensor([0.6])
    p["delay_mix"] = torch.tensor([0.4])
    p["reverb_room_size"] = torch.tensor([0.3])
    p["reverb_decay"] = torch.tensor([0.4])
    p["reverb_mix"] = torch.tensor([0.2])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "06_delay_lead")

    # --- 7. Wavetable Morph ---
    print("7. Wavetable Morph")
    p = make_base_params()
    p["osc_type"] = torch.tensor([[0.0, 1.0, 0.0]])
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
    p["osc_type"] = torch.tensor([[0.0, 0.0, 1.0]])
    p["osc_pitch"] = torch.tensor([0.55])
    p["fm_carrier_ratio"] = torch.tensor([0.0])
    p["fm_mod_ratio"] = torch.tensor([0.0])
    p["fm_mod_index"] = torch.tensor([0.15])
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

    # --- 9. DnB Bass Sequence ---
    print("9. DnB Bass Sequence")
    p = make_base_params()
    p["osc_waveform"] = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    p["filter_cutoff"] = torch.tensor([0.4])
    p["filter_q"] = torch.tensor([0.5])
    p["filt_env_amount"] = torch.tensor([0.7])
    p["dist_amount"] = torch.tensor([0.3])
    p["dist_mix"] = torch.tensor([0.5])

    seq_pitch = torch.full((1, 32), 0.2)
    seq_velocity = torch.zeros(1, 32)
    for step in [0, 6, 8, 14, 16, 22, 24, 30]:
        seq_velocity[0, step] = 0.9
    seq_pitch[0, 8] = 0.25
    seq_pitch[0, 24] = 0.18
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
    p["lfo_rate"] = torch.tensor([0.35])
    p["lfo_depth"] = torch.tensor([0.9])
    p["lfo_waveform"] = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    p["lfo_target"] = torch.tensor([[1.0, 0.0, 0.3, 0.0]])
    p["lfo_phase"] = torch.tensor([0.0])
    with torch.no_grad():
        audio = synth(p)
    save_wav(audio[0], "10_wobble_bass")

    print("\nDone! Check output/ folder.")


if __name__ == "__main__":
    main()
