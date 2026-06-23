"""Test model on classic synth timbres at various pitches.

Tests whether the model can reproduce low-frequency sounds, or if there's
a systematic bias toward higher frequencies.
"""
import torch, sys, warnings; warnings.filterwarnings('ignore'); sys.path.insert(0, 'src')
from loom.core import SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.training.dataset import params_to_vector, vector_to_params, CONTINUOUS_KEYS
from loom.training.losses import multi_resolution_stft_loss, param_loss
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import sample_euler

n_audio = 44100
synth = SubtractiveSynth(SAMPLE_RATE, n_audio).eval()

state = torch.load('data_flow_id/best_flow.pt', map_location='cpu', weights_only=True)
cond = Conditioner(frontend='cqt', n_bins=192, d_model=256, d_cond=512, n_layers=4)
cond.load_state_dict(state['conditioner']); cond.eval()
flow = FlowNetwork(d_model=256, n_dit_blocks=4, nhead=8, d_cond=512)
flow.load_state_dict(state['flow_net']); flow.eval()

# ── Classic timbre presets ──────────────────────────────────────────────
# Each preset is a hand-crafted parameter dict matching synth expectations.
# We use the ParamTokenizer's understanding of the 97-dim vector.

def make_params(pitch, detune, waveform_idx, cutoff, q_val, osc_type=0):
    """Build a complete parameter vector for stage 1."""
    vec = torch.zeros(1, 97)
    # Continuous (first 43)
    vec[0, 0] = pitch       # osc_pitch
    vec[0, 1] = detune      # osc_detune
    vec[0, 10] = cutoff     # filter_cutoff
    vec[0, 11] = q_val      # filter_q
    # Categorical (43:61)
    vec[0, 43 + waveform_idx] = 1.0  # osc_waveform (4 classes)
    vec[0, 47 + osc_type] = 1.0      # osc_type (3 classes)
    # Filter type: 2=LP
    vec[0, 52] = 1.0  # filter_type class 2 (LP)

    # Fill fixed defaults (same as get_stage_fixed_vector for stage 1)
    vec[0, 2] = 0.5   # wt_position
    vec[0, 3:6] = 0.5 # fm params (0.5, 0.5, 0.0)
    vec[0, 5] = 0.0
    vec[0, 6:10] = torch.tensor([0.05, 0.3, 0.8, 0.3])  # amp ADSR
    vec[0, 12] = 1.0  # filter_mix
    vec[0, 13:18] = torch.tensor([0.1, 0.3, 0.7, 0.3, 0.5])  # filt_env
    vec[0, 20] = 0.7  # master_gain
    vec[0, 21:43] = 0.3  # FX defaults
    vec[0, 26] = 0.0; vec[0, 29] = 0.0; vec[0, 32] = 0.0; vec[0, 36] = 0.0
    vec[0, 37:40] = 0.5  # EQ
    vec[0, 40:43] = torch.tensor([0.5, 0.0, 0.0])  # LFO (off)
    # lfo_waveform, lfo_target → class 0
    vec[0, 53] = 1.0  # lfo_waveform class 0
    vec[0, 57] = 1.0  # lfo_target class 0
    # osc_waveform, osc_type already set above
    return vec

from scipy.io import wavfile
import numpy as np

def save_wav(path, audio_tensor):
    a = audio_tensor.squeeze(0).cpu().numpy()
    peak = abs(a).max()
    if peak > 1.0: a = a / peak * 0.95
    a_int16 = (a.clip(-1.0, 1.0) * 32767).astype(np.int16)
    wavfile.write(path, SAMPLE_RATE, a_int16)

# ── Test cases ──────────────────────────────────────────────────────────

tests = [
    # Name,         pitch, detune, waveform, cutoff, q,    osc_type
    ("sub_bass_C2",  0.15,  0.1,    0,        0.08,  0.6,  0),  # sine, very low
    ("sub_bass_C3",  0.40,  0.1,    0,        0.10,  0.6,  0),  # sine, mid-low
    ("reese_C2",     0.15,  0.85,   1,        0.15,  0.5,  0),  # detuned saws
    ("reese_C3",     0.40,  0.85,   1,        0.20,  0.5,  0),
    ("supersaw_C4",  0.65,  0.95,   1,        0.60,  0.4,  0),  # bright detuned
    ("pluck_C4",     0.60,  0.2,    2,        1.00,  0.3,  0),  # square, open filter
    ("pad_C3",       0.40,  0.4,    1,        0.30,  0.7,  0),  # saw, resonant
]

import os
os.makedirs("demo_audio_classic", exist_ok=True)

print(f"{'Name':<18} {'Target Pitch':>12} {'Flow Pitch':>10} {'T RMS':>8} {'F RMS':>8} {'Spec Loss':>10}")
print("-" * 72)

for name, pitch, detune, wf, cutoff, q, otype in tests:
    target_vec = make_params(pitch, detune, wf, cutoff, q, otype)

    # Render target
    tp = vector_to_params(target_vec); tp.pop('fx_routing', None)
    with torch.no_grad(): target_a = synth(tp)

    # Conditioner
    with torch.no_grad(): cv, al = cond(target_a)

    # Flow ODE
    with torch.no_grad(): pred_vec = sample_euler(flow, cv, al, n_steps=20, stage=1)

    # Render flow
    pp = vector_to_params(pred_vec); pp.pop('fx_routing', None)
    with torch.no_grad(): pred_a = synth(pp)

    spec_l = multi_resolution_stft_loss(pred_a, target_a).item()
    t_rms = target_a.norm().item()
    f_rms = pred_a.norm().item()

    print(f'{name:<18} {pitch:>12.3f} {pred_vec[0,0]:>10.3f} {t_rms:>8.2f} {f_rms:>8.2f} {spec_l:>10.4f}')

    save_wav(f"demo_audio_classic/{name}_target.wav", target_a)
    save_wav(f"demo_audio_classic/{name}_flow.wav", pred_a)

print(f"\nSaved {len(tests)*2} WAVs to demo_audio_classic/")
print("Listen: *_target.wav = ground truth, *_flow.wav = model prediction")
