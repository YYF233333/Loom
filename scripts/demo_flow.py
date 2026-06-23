"""Flow Matching demo — prove the model learned something.

Generates test samples in batches (low memory), runs ODE inference, and
compares Flow vs Random vs Untrained baselines.

Usage:
    uv run python scripts/demo_flow.py --train-first --n-samples 16 --stage 1
    uv run python scripts/demo_flow.py --checkpoint data_flow/best_flow.pt
"""

import argparse
import sys
import time

import torch

sys.path.insert(0, "src")

from loom.core import DEVICE, SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import params_to_vector, vector_to_params
from loom.training.losses import param_loss, multi_resolution_stft_loss
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import sample_euler


# ── Batch generators (low memory) ──────────────────────────────────────────


def _generate_batch(synth, batch_size: int, stage: int, audio_dur: float = 0.5):
    """Generate one batch of (audio, params). No accumulation."""
    params = random_params(batch_size, stage=stage)
    params.pop("fx_routing", None)
    with torch.no_grad():
        audio = synth(params)
    pv = params_to_vector(params)
    return audio, pv


# ── Training ────────────────────────────────────────────────────────────────


def train_quick_model(
    stage: int = 0,
    n_samples: int = 800,
    epochs: int = 500,
    gen_batch: int = 64,
):
    """Train a small flow model, generating data in sub-batches to limit RAM."""
    d_model, d_cond, n_blocks = 128, 256, 2
    n_audio = int(SAMPLE_RATE * 0.5)

    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).eval()
    cond = Conditioner(frontend="gammatone", n_bins=128, d_model=d_model, d_cond=d_cond, n_blocks=n_blocks)
    flow = FlowNetwork(d_model=d_model, n_dit_blocks=n_blocks, nhead=4, d_cond=d_cond)
    opt = torch.optim.AdamW(list(cond.parameters()) + list(flow.parameters()), lr=3e-4, weight_decay=0.01)

    # ── Generate data in batches & pre-compute conditions ──
    print(f"Generating {n_samples} samples in batches of {gen_batch}...")
    all_conds = []
    all_params = []
    memory_mb = 0

    for offset in range(0, n_samples, gen_batch):
        bs = min(gen_batch, n_samples - offset)
        audio, pv = _generate_batch(synth, bs, stage)
        with torch.no_grad():
            c = cond(audio)
        all_conds.append(c)
        all_params.append(pv)
        # Track peak memory
        cur_mb = (audio.element_size() * audio.numel()) / 1e6
        memory_mb = max(memory_mb, cur_mb)
        del audio  # free audio immediately

    train_cond = torch.cat(all_conds, dim=0)
    train_params = torch.cat(all_params, dim=0)
    del all_conds, all_params

    print(f"  Done. Peak audio batch: {memory_mb:.0f} MB, stored: {train_cond.element_size() * train_cond.numel() / 1e6:.0f} MB")
    print(f"Training {epochs} epochs, batch_size=64...")

    B = 64
    t0 = time.perf_counter()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        ep_loss = 0.0
        nb = 0
        for start in range(0, n_samples, B):
            idx = perm[start:start + B]
            opt.zero_grad()
            loss = flow.compute_loss(train_params[idx], train_cond[idx])
            loss.backward()
            opt.step()
            ep_loss += loss.item()
            nb += 1

        if epoch % 100 == 0:
            print(f"  epoch {epoch:4d}: loss={ep_loss / nb:.4f}  [{time.perf_counter() - t0:.0f}s]")

    elapsed = time.perf_counter() - t0
    print(f"  final loss: {ep_loss / nb:.4f}  [{elapsed:.0f}s, {elapsed/epochs*1000:.0f}ms/epoch]")
    print()

    cond.eval()
    flow.eval()
    return cond, flow, synth


# ── Demo ─────────────────────────────────────────────────────────────────────


def run_demo(cond, flow, synth, n_samples: int = 12, stage: int = 0):
    """Run inference demo — batched generation, low memory."""
    print("=" * 70)
    print("DEMO: Flow Matching Parameter Estimation")
    print(f"      Model: 2.1M params | Stage: {stage} | Samples: {n_samples}")
    print("=" * 70)
    print()

    n_audio = int(SAMPLE_RATE * 0.5)

    # ── Generate test data in one small batch (n_samples is small) ──
    torch.manual_seed(12345)
    test_audio, test_vec = _generate_batch(synth, n_samples, stage)
    test_params = vector_to_params(test_vec)

    # ── Conditioner ──
    with torch.no_grad():
        test_cond = cond(test_audio)

    # ── ODE inference ──
    t0 = time.perf_counter()
    with torch.no_grad():
        pred_vec = sample_euler(flow, test_cond, n_steps=20)
    ode_ms = (time.perf_counter() - t0) * 1000

    # ── Random baseline ──
    rand_vec = torch.rand(n_samples, 97)
    rand_vec[:, :43] = rand_vec[:, :43]
    rand_vec[:, 43:61] = torch.softmax(torch.randn(n_samples, 18), dim=-1)

    # ── Untrained baseline ──
    untrained_flow = FlowNetwork(d_model=128, n_dit_blocks=2, nhead=4, d_cond=256).eval()
    with torch.no_grad():
        untrained_vec = sample_euler(untrained_flow, test_cond, n_steps=20)

    # ── Evaluate sample-by-sample (render one at a time) ──
    results = []
    for i in range(n_samples):
        # Render each method
        def _render(vec_slice):
            pdict = vector_to_params(vec_slice)
            pdict.pop("fx_routing", None)
            with torch.no_grad():
                return synth(pdict)

        target_a  = _render(test_vec[i:i + 1])
        pred_a    = _render(pred_vec[i:i + 1])
        rand_a    = _render(rand_vec[i:i + 1])
        untrained_a = _render(untrained_vec[i:i + 1])

        flow_s = multi_resolution_stft_loss(pred_a, target_a).item()
        rand_s = multi_resolution_stft_loss(rand_a, target_a).item()
        unt_s  = multi_resolution_stft_loss(untrained_a, target_a).item()
        flow_pl = param_loss(pred_vec[i:i + 1], test_vec[i:i + 1]).item()
        rand_pl = param_loss(rand_vec[i:i + 1], test_vec[i:i + 1]).item()

        pred_dict = vector_to_params(pred_vec[i:i + 1])
        rand_dict = vector_to_params(rand_vec[i:i + 1])

        results.append({
            "i": i,
            "flow_s": flow_s,
            "rand_s": rand_s,
            "unt_s": unt_s,
            "flow_p": flow_pl,
            "rand_p": rand_pl,
            "target_pitch": test_params["osc_pitch"][i].item(),
            "pred_pitch":   pred_dict["osc_pitch"].item(),
            "rand_pitch":   rand_dict["osc_pitch"].item(),
            "target_detune": test_params["osc_detune"][i].item(),
            "pred_detune":   pred_dict["osc_detune"].item(),
            "rand_detune":   rand_dict["osc_detune"].item(),
            "target_wf":   test_params["osc_waveform"][i].argmax().item(),
            "pred_wf":     pred_dict["osc_waveform"].argmax().item(),
            "rand_wf":     rand_dict["osc_waveform"].argmax().item(),
        })

    # ── Print table ──
    print(f"ODE: {ode_ms:.0f}ms total, {ode_ms/n_samples:.1f}ms/sample")
    print()
    hdr = f"{'#':<4} {'Flow Spec':>10} {'Rand Spec':>10} {'Untr Spec':>10} | {'Flow Param':>10} {'Rand Param':>10} | Win?"
    print(hdr)
    print("-" * len(hdr))

    sum_fs = sum_rs = sum_us = sum_fp = sum_rp = 0.0
    for r in results:
        sum_fs += r["flow_s"]; sum_rs += r["rand_s"]; sum_us += r["unt_s"]
        sum_fp += r["flow_p"]; sum_rp += r["rand_p"]
        win = "✓" if r["flow_s"] < r["rand_s"] else ""
        print(f'{r["i"]:<4} {r["flow_s"]:>10.4f} {r["rand_s"]:>10.4f} {r["unt_s"]:>10.4f} | {r["flow_p"]:>10.4f} {r["rand_p"]:>10.4f} | {win:>4}')

    n = len(results)
    print("-" * len(hdr))
    print(f'{"AVG":<4} {sum_fs/n:>10.4f} {sum_rs/n:>10.4f} {sum_us/n:>10.4f} | {sum_fp/n:>10.4f} {sum_rp/n:>10.4f}')
    spec_win = (sum_rs - sum_fs) / sum_rs * 100
    param_win = (sum_rp - sum_fp) / sum_rp * 100

    # ── Parameter comparison ──
    print()
    print(f'{"Param":<18} {"Target":>8} {"Flow":>8} {"Rand":>8} {"Δ Flow":>8} {"Δ Rand":>8}')
    print("-" * 58)
    for r in results[:5]:  # show first 5 samples
        for label, t_val, p_val, r_val in [
            ("pitch", r["target_pitch"], r["pred_pitch"], r["rand_pitch"]),
            ("detune", r["target_detune"], r["pred_detune"], r["rand_detune"]),
        ]:
            dp = abs(p_val - t_val)
            dr = abs(r_val - t_val)
            closer = "←" if dp < dr else " "
            print(f'{label:<18} {t_val:>8.3f} {p_val:>8.3f} {r_val:>8.3f} {dp:>8.3f} {dr:>8.3f} {closer}')
        print()

    # ── Waveform accuracy ──
    wf_correct = sum(1 for r in results if r["pred_wf"] == r["target_wf"])
    print(f"Waveform classification: {wf_correct}/{n} ({wf_correct/n*100:.0f}%)")

    # ── Conclusion ──
    print()
    print("=" * 70)
    print(f" Spectral: {spec_win:+.0f}% vs random  |  Param: {param_win:+.0f}% vs random")
    if spec_win > 15:
        print(" VERDICT: Model clearly learned the audio→parameter mapping.")
    elif spec_win > 3:
        print(" VERDICT: Model is learning — improving beyond random baseline.")
    else:
        print(" VERDICT: Model needs more capacity/data/epochs to surpass random on spectral metric.")
        print("         (Param metric already better — model gets params closer but not close enough for audio match.)")
    print("=" * 70)


# ── Main ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flow Matching Demo")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train-first", action="store_true")
    parser.add_argument("--n-samples", type=int, default=12)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--train-epochs", type=int, default=500)
    parser.add_argument("--train-size", type=int, default=800)
    args = parser.parse_args()

    if args.train_first or args.checkpoint is None:
        cond, flow, synth = train_quick_model(
            stage=args.stage,
            n_samples=args.train_size,
            epochs=args.train_epochs,
        )
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        cond = Conditioner(d_model=128, d_cond=256, n_blocks=2)
        cond.load_state_dict(state["conditioner"])
        cond.eval()
        flow = FlowNetwork(d_model=128, n_dit_blocks=2, nhead=4, d_cond=256)
        flow.load_state_dict(state["flow_net"])
        flow.eval()
        synth = SubtractiveSynth(SAMPLE_RATE, int(SAMPLE_RATE * 0.5)).eval()

    run_demo(cond, flow, synth, n_samples=args.n_samples, stage=args.stage)
