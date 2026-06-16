"""Gradient magnitude analysis across all synth parameters.

Measures gradient norms for each parameter to identify imbalance.
"""

import torch
import sys
sys.path.insert(0, "src")

from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.core import SAMPLE_RATE, DEVICE

N_SEEDS = 10
N_SAMPLES = 22050  # 0.5s


def measure_gradients():
    synth = SubtractiveSynth(SAMPLE_RATE, N_SAMPLES).to(DEVICE)

    continuous_keys = [
        "osc_pitch", "osc_detune",
        "wt_position",
        "fm_carrier_ratio", "fm_mod_ratio", "fm_mod_index",
        "amp_attack", "amp_decay", "amp_sustain", "amp_release",
        "filter_cutoff", "filter_q",
        "filt_env_attack", "filt_env_decay", "filt_env_sustain",
        "filt_env_release", "filt_env_amount",
        "dist_amount", "dist_mix", "master_gain",
        "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
        "comp_makeup", "comp_mix",
        "chorus_rate", "chorus_depth", "chorus_mix",
        "delay_time", "delay_feedback", "delay_mix",
        "reverb_room_size", "reverb_decay", "reverb_damping", "reverb_mix",
        "eq_low_gain", "eq_mid_gain", "eq_high_gain",
        "lfo_rate", "lfo_depth", "lfo_phase",
    ]

    grad_accum = {k: [] for k in continuous_keys}

    for seed in range(N_SEEDS):
        torch.manual_seed(seed)
        params = random_params(1, device=DEVICE)

        for key in continuous_keys:
            params[key] = params[key].detach().clone().requires_grad_(True)

        audio = synth(params)
        loss = audio.pow(2).mean()
        loss.backward()

        for key in continuous_keys:
            g = params[key].grad
            if g is not None:
                grad_accum[key].append(g.abs().item())

    print(f"{'Parameter':<25} {'Mean |grad|':>15} {'Min':>12} {'Max':>12}")
    print("-" * 68)

    stats = []
    for key in continuous_keys:
        vals = grad_accum[key]
        if vals:
            mean_g = sum(vals) / len(vals)
            min_g = min(vals)
            max_g = max(vals)
            stats.append((key, mean_g, min_g, max_g))

    stats.sort(key=lambda x: x[1], reverse=True)

    for key, mean_g, min_g, max_g in stats:
        print(f"{key:<25} {mean_g:>15.6f} {min_g:>12.6f} {max_g:>12.6f}")

    biggest = stats[0][1]
    smallest = stats[-1][1]
    print(f"\nMax/Min ratio: {biggest / max(smallest, 1e-15):.0f}x")
    print(f"Largest:  {stats[0][0]} = {stats[0][1]:.6f}")
    print(f"Smallest: {stats[-1][0]} = {stats[-1][1]:.6f}")

    # Group by order of magnitude
    print("\n--- Grouped by magnitude ---")
    for exp in range(6, -10, -1):
        group = [(k, m) for k, m, _, _ in stats if 10**(exp-1) <= m < 10**exp]
        if group:
            print(f"\n  1e{exp-1} ~ 1e{exp}:")
            for k, m in group:
                print(f"    {k:<25} {m:.6f}")


if __name__ == "__main__":
    measure_gradients()
