"""Flow matching inference — ODE sampling + optional gradient refinement.

Usage:
    # Quick: ODE sampling only (fast)
    uv run python -m loom.flow.inference --checkpoint data_flow/best_flow.pt --mode ode --n-steps 20

    # Best quality: ODE + gradient refinement
    uv run python -m loom.flow.inference --checkpoint data_flow/best_flow.pt --mode refine --refine-steps 50

    # Best-of-N: multiple ODE samples, pick best (leverages flow diversity)
    uv run python -m loom.flow.inference --checkpoint data_flow/best_flow.pt --mode best-n --n-candidates 8

    # From target audio (generates random ground truth, tries to match)
    uv run python -m loom.flow.inference --checkpoint data_flow/best_flow.pt --mode ode
"""

import argparse
import sys
import time

import torch

from loom.core import DEVICE, SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import params_to_vector, vector_to_params
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import (
    sample_euler,
    sample_midpoint,
    refine_via_gradient,
    sample_best_of_n,
)
from loom.training.losses import param_loss, multi_resolution_stft_loss


def load_checkpoint(checkpoint_path: str, d_model: int = 256, d_cond: int = 512):
    """Load conditioned flow model from checkpoint."""
    state = torch.load(checkpoint_path, map_location=DEVICE, weights_only=True)

    conditioner = Conditioner(d_model=d_model, d_cond=d_cond).to(DEVICE)
    conditioner.load_state_dict(state["conditioner"])
    conditioner.eval()

    flow_net = FlowNetwork(d_model=d_model, d_cond=d_cond).to(DEVICE)
    flow_net.load_state_dict(state["flow_net"])
    flow_net.eval()

    print(f"Loaded checkpoint (epoch {state.get('epoch', '?')}, val_loss={state.get('val_loss', float('nan')):.4f})")
    return conditioner, flow_net


def run_inference(args):
    conditioner, flow_net = load_checkpoint(args.checkpoint, args.d_model, args.d_cond)

    n_audio = int(SAMPLE_RATE * args.audio_duration)
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
    synth.eval()

    # Generate target audio
    torch.manual_seed(args.seed)
    target_params = random_params(1, device=DEVICE, stage=args.stage)
    target_params.pop("fx_routing", None)
    with torch.no_grad():
        target_audio = synth(target_params)
    target_vec = params_to_vector(target_params)
    print(f"Target audio: {target_audio.shape}, duration={args.audio_duration}s")

    # Compute audio condition
    t0 = time.perf_counter()
    with torch.no_grad():
        audio_cond, audio_lats = conditioner(target_audio)
    cond_time = time.perf_counter() - t0
    print(f"Conditioner: {cond_time*1000:.1f}ms")

    # ── Mode: ODE sampling ──────────────────────────────────────────────
    if args.mode == "ode":
        t0 = time.perf_counter()
        with torch.no_grad():
            if args.solver == "midpoint":
                pred_vec = sample_midpoint(flow_net, audio_cond, n_steps=args.n_steps)
            else:
                pred_vec = sample_euler(flow_net, audio_cond, audio_lats, n_steps=args.n_steps)
        ode_time = time.perf_counter() - t0
        print(f"ODE sampling ({args.solver}, {args.n_steps} steps): {ode_time*1000:.1f}ms")

        p_loss = param_loss(pred_vec, target_vec).item()

        pred_p = vector_to_params(pred_vec)
        pred_p.pop("fx_routing", None)
        with torch.no_grad():
            pred_audio = synth(pred_p)
        s_loss = multi_resolution_stft_loss(pred_audio, target_audio).item()

        print(f"Param loss: {p_loss:.4f}")
        print(f"Spectral loss: {s_loss:.4f}")

    # ── Mode: gradient refinement ───────────────────────────────────────
    elif args.mode == "refine":
        # Step 1: ODE
        t0 = time.perf_counter()
        with torch.no_grad():
            init_vec = sample_euler(flow_net, audio_cond, audio_lats, n_steps=args.n_steps)
        ode_time = time.perf_counter() - t0

        # Step 2: Refine
        refined_vec, loss_history = refine_via_gradient(
            flow_net, synth, audio_cond, target_audio,
            init_vec, n_steps=args.refine_steps, lr=args.refine_lr,
        )
        refine_time = time.perf_counter() - t0 - ode_time

        p_loss = param_loss(refined_vec, target_vec).item()
        refined_p = vector_to_params(refined_vec)
        refined_p.pop("fx_routing", None)
        with torch.no_grad():
            refined_audio = synth(refined_p)
        s_loss = multi_resolution_stft_loss(refined_audio, target_audio).item()

        print(f"ODE ({args.n_steps} steps): {ode_time*1000:.1f}ms")
        print(f"Refine ({args.refine_steps} steps): {refine_time*1000:.1f}ms")
        print(f"Param loss: {p_loss:.4f}")
        print(f"Spectral loss: {s_loss:.4f}")
        print(f"Loss trace: {[f'{x:.4f}' for x in loss_history[:5]]}...")

    # ── Mode: best-of-N ─────────────────────────────────────────────────
    elif args.mode == "best-n":
        t0 = time.perf_counter()
        best_params, best_audio, losses = sample_best_of_n(
            flow_net, synth, audio_cond, target_audio,
            n_candidates=args.n_candidates, n_ode_steps=args.n_steps,
        )
        total_time = time.perf_counter() - t0

        p_loss = param_loss(best_params, target_vec).item()
        s_loss = multi_resolution_stft_loss(best_audio, target_audio).item()

        print(f"Candidates: {args.n_candidates}, ODE steps: {args.n_steps}")
        print(f"Time: {total_time:.1f}s")
        print(f"Losses: {[f'{x:.4f}' for x in losses]}")
        print(f"Best param loss: {p_loss:.4f}")
        print(f"Best spectral loss: {s_loss:.4f}")

    # ── Mode: profile ───────────────────────────────────────────────────
    elif args.mode == "profile":
        """Profile inference speed and quality trade-offs."""
        print(f"\n{'Steps':>6} {'Param Loss':>12} {'Spectral Loss':>14} {'Time (ms)':>10}")
        print("-" * 48)

        for n_steps in [5, 10, 20, 50, 100]:
            t0 = time.perf_counter()
            with torch.no_grad():
                pred_vec = sample_euler(flow_net, audio_cond, n_steps=n_steps)
            elapsed = (time.perf_counter() - t0) * 1000

            p_loss = param_loss(pred_vec, target_vec).item()
            pred_p = vector_to_params(pred_vec)
            pred_p.pop("fx_routing", None)
            with torch.no_grad():
                pred_audio = synth(pred_p)
            s_loss = multi_resolution_stft_loss(pred_audio, target_audio).item()

            print(f"{n_steps:>6} {p_loss:>12.4f} {s_loss:>14.4f} {elapsed:>10.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow matching inference")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to best_flow.pt")
    parser.add_argument("--mode", type=str, default="ode",
                        choices=["ode", "refine", "best-n", "profile"],
                        help="Inference mode")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audio-duration", type=float, default=1.0)
    parser.add_argument("--stage", type=int, default=99)

    # Model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-cond", type=int, default=512)

    # ODE
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--solver", type=str, default="euler",
                        choices=["euler", "midpoint"])

    # Refine
    parser.add_argument("--refine-steps", type=int, default=50)
    parser.add_argument("--refine-lr", type=float, default=0.01)

    # Best-of-N
    parser.add_argument("--n-candidates", type=int, default=8)

    args = parser.parse_args()
    run_inference(args)
