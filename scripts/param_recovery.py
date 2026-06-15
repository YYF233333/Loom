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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    n_samples = 44100  # 1 second for reasonable speed
    synth = SubtractiveSynth(SAMPLE_RATE, n_samples).to(device)

    target_params = random_params(1, device=device)
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
        [pred_params[k] for k in optimize_keys], lr=0.005
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

        loss = torch.tensor(0.0, device=device)
        for fft_size in [512, 1024, 2048]:
            window = torch.hann_window(fft_size, device=device)
            target_stft = torch.stft(
                target_audio[0], fft_size,
                hop_length=fft_size // 4,
                return_complex=True,
                window=window,
            )
            pred_stft = torch.stft(
                pred_audio[0], fft_size,
                hop_length=fft_size // 4,
                return_complex=True,
                window=window,
            )
            loss = loss + (target_stft.abs() - pred_stft.abs()).pow(2).mean()

        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if step % 50 == 0:
            print(f"Step {step:4d} | Loss: {loss.item():.6f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(losses)
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Multi-res STFT Loss")
    axes[0].set_title("Convergence")
    axes[0].set_yscale("log")

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
