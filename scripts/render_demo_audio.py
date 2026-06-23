"""Render demo audio samples for listening comparison.

Generates Target / Flow / Random audio pairs and saves as WAV files.

Usage:
    uv run python scripts/render_demo_audio.py --checkpoint data_flow/best_flow.pt
    uv run python scripts/render_demo_audio.py --train-first
"""

import argparse
import sys
import time

import torch
import torchaudio

sys.path.insert(0, "src")

from loom.core import DEVICE, SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import params_to_vector, vector_to_params
from loom.training.losses import multi_resolution_stft_loss, param_loss
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import sample_euler


def train_model(stage: int = 1):
    """Train a small model (reuses demo_flow logic)."""
    d_model, d_cond, n_blocks = 128, 256, 2
    n_audio = int(SAMPLE_RATE * 1.0)
    n_samples = 800
    gen_batch = 64

    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).eval()
    cond = Conditioner(frontend="gammatone", n_bins=128, d_model=d_model, d_cond=d_cond, n_blocks=n_blocks)
    flow = FlowNetwork(d_model=d_model, n_dit_blocks=n_blocks, nhead=4, d_cond=d_cond)
    opt = torch.optim.AdamW(list(cond.parameters()) + list(flow.parameters()), lr=3e-4, weight_decay=0.01)

    print(f"Training (stage={stage}, {n_samples} samples)...")
    all_conds, all_params = [], []
    for offset in range(0, n_samples, gen_batch):
        bs = min(gen_batch, n_samples - offset)
        params = random_params(bs, stage=stage)
        params.pop("fx_routing", None)
        with torch.no_grad():
            audio = synth(params)
        with torch.no_grad():
            c = cond(audio)
        all_conds.append(c)
        all_params.append(params_to_vector(params))
        del audio

    train_cond = torch.cat(all_conds, dim=0)
    train_params = torch.cat(all_params, dim=0)
    del all_conds, all_params

    B = 64
    for epoch in range(500):
        perm = torch.randperm(n_samples)
        for start in range(0, n_samples, B):
            idx = perm[start:start + B]
            opt.zero_grad()
            loss = flow.compute_loss(train_params[idx], train_cond[idx])
            loss.backward()
            opt.step()
        if epoch % 100 == 0:
            print(f"  epoch {epoch}...")

    cond.eval(); flow.eval()
    return cond, flow, synth


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--out-dir", type=str, default="demo_audio")
    args = parser.parse_args()

    n_audio = int(SAMPLE_RATE * 1.0)  # 1s for CQT frontend
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).eval()

    if args.train_first or args.checkpoint is None:
        cond, flow, synth = train_model(stage=args.stage)
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cond = Conditioner(frontend="multires", n_bins=192, d_model=256, d_cond=512, n_layers=4)
        cond.load_state_dict(state["conditioner"]); cond.eval()
        flow = FlowNetwork(d_model=256, n_dit_blocks=4, nhead=8, d_cond=512)
        flow.load_state_dict(state["flow_net"]); flow.eval()

    import os
    os.makedirs(args.out_dir, exist_ok=True)

    torch.manual_seed(999)
    test_params = random_params(args.n_samples, stage=args.stage)
    test_params.pop("fx_routing", None)
    with torch.no_grad():
        test_audio = synth(test_params)
    test_vec = params_to_vector(test_params)

    with torch.no_grad():
        test_cond, test_lats = cond(test_audio)
        pred_vec = sample_euler(flow, test_cond, test_lats, n_steps=20, stage=args.stage)

    # Random baseline
    rand_vec = torch.rand(args.n_samples, 97)
    rand_vec[:, :43] = rand_vec[:, :43]
    rand_vec[:, 43:61] = torch.softmax(torch.randn(args.n_samples, 18), dim=-1)

    print()
    print(f"{'Sample':<8} {'Flow Spec':>10} {'Rand Spec':>10} {'Win?':>6}")
    print("-" * 38)

    for i in range(args.n_samples):
        # Target
        target_p = vector_to_params(test_vec[i:i+1])
        target_p.pop("fx_routing", None)
        with torch.no_grad():
            target_a = synth(target_p)

        # Flow
        pred_p = vector_to_params(pred_vec[i:i+1])
        pred_p.pop("fx_routing", None)
        with torch.no_grad():
            pred_a = synth(pred_p)

        # Random
        rand_p = vector_to_params(rand_vec[i:i+1])
        rand_p.pop("fx_routing", None)
        with torch.no_grad():
            rand_a = synth(rand_p)

        # Metrics
        flow_s = multi_resolution_stft_loss(pred_a, target_a).item()
        rand_s = multi_resolution_stft_loss(rand_a, target_a).item()
        win = "✓" if flow_s < rand_s else ""

        print(f"{i:<8} {flow_s:>10.4f} {rand_s:>10.4f} {win:>6}")

        # Save WAV files (normalize to avoid clipping)
        from scipy.io import wavfile
        import numpy as np

        def save_wav(path, audio_tensor):
            a = audio_tensor.squeeze(0).cpu().numpy()
            peak = abs(a).max()
            if peak > 1.0:
                a = a / peak * 0.95
            # Convert float32 [-1,1] → int16
            a_int16 = (a.clip(-1.0, 1.0) * 32767).astype(np.int16)
            wavfile.write(path, SAMPLE_RATE, a_int16)

        save_wav(f"{args.out_dir}/sample{i:02d}_target.wav", target_a)
        save_wav(f"{args.out_dir}/sample{i:02d}_flow.wav", pred_a)
        save_wav(f"{args.out_dir}/sample{i:02d}_random.wav", rand_a)

    print(f"\nSaved {args.n_samples * 3} WAV files to {args.out_dir}/")
    print("Files: sampleXX_target.wav (ground truth)")
    print("       sampleXX_flow.wav   (ODE inference)")
    print("       sampleXX_random.wav (random baseline)")
    print()
    print("Listen and compare! If flow sounds more like target than random does,")
    print("the model has learned the audio→parameter mapping.")


if __name__ == "__main__":
    main()
