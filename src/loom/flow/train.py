"""Flow matching training script — simulation-free, no synth needed during training.

Usage:
    uv run python -m loom.flow.train --n-samples 50000 --epochs 200 --batch-size 1024
    uv run python -m loom.flow.train --pool-size 20000 --pool-epochs 5 --epochs 500

Key differences from the regression pipeline:
    - No spectral loss during training (pure vector regression)
    - No synth forward pass (simulation-free — extremely fast)
    - No GradNorm balancing, no gradient accumulation
    - No curriculum strictly required (but supported via data stages)
    - 5-10× faster per epoch than regression with spectral loss
"""

import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torchaudio.transforms as T

from loom.core import DEVICE, SAMPLE_RATE
from loom.render import random_params
from loom.synth import SubtractiveSynth
from loom.training.dataset import params_to_vector, generate_dataset, load_dataset
from loom.flow.tokenizer import N_TOKENS
from loom.flow.conditioner import Conditioner
from loom.flow.dit import FlowNetwork
from loom.flow.flow_matching import sample_euler
from loom.flow.frontend import build_frontend


# ── Data utilities ──────────────────────────────────────────────────────────


def generate_pool(
    n_samples: int,
    synth: nn.Module,
    frontend: nn.Module,
    gen_batch_size: int,
    device: str,
    audio_duration: float = 1.0,
    stage: int = 99,
):
    """Generate (audio, params) pool. Audio stays raw for conditioner input."""
    n_audio = int(SAMPLE_RATE * audio_duration)

    with torch.no_grad():
        probe_p = random_params(1, device=device, stage=stage)
        probe_p.pop("fx_routing", None)
        probe_a = synth(probe_p)

    audio_shape = (n_samples, n_audio)
    param_shape = (n_samples, params_to_vector(probe_p).shape[-1])

    audio_mem = torch.empty(audio_shape, device=device)
    params_mem = torch.empty(param_shape, device=device)

    for offset in range(0, n_samples, gen_batch_size):
        bs = min(gen_batch_size, n_samples - offset)
        params = random_params(bs, device=device, stage=stage)
        params.pop("fx_routing", None)
        with torch.no_grad():
            audio = synth(params)
        audio_mem[offset:offset + bs] = audio
        params_mem[offset:offset + bs] = params_to_vector(params)

    return audio_mem, params_mem


# ── Training utilities ──────────────────────────────────────────────────────


def precompute_conditions(
    conditioner: nn.Module,
    audio_batch: torch.Tensor,
    cond_batch_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute audio conditions + latent tokens in sub-batches."""
    cond_vecs, latents_list = [], []
    for i in range(0, len(audio_batch), cond_batch_size):
        chunk = audio_batch[i:i + cond_batch_size]
        with torch.no_grad():
            cv, al = conditioner(chunk)
        cond_vecs.append(cv)
        latents_list.append(al)
    return torch.cat(cond_vecs, dim=0), torch.cat(latents_list, dim=0)


def train_epoch(
    flow_net: nn.Module,
    audio_conds: torch.Tensor,
    audio_latents: torch.Tensor,
    params_all: torch.Tensor,
    batch_size: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
    stage: int = 99,
) -> float:
    """Single training epoch. Returns average loss."""
    flow_net.train()
    n_total = len(params_all)
    perm = torch.randperm(n_total, device=params_all.device)
    total_loss = 0.0
    n_batches = 0

    for start in range(0, n_total, batch_size):
        idx = perm[start:start + batch_size]
        params_batch = params_all[idx]
        cond_batch = audio_conds[idx]
        lat_batch = audio_latents[idx]

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = flow_net.compute_loss(params_batch, cond_batch, lat_batch, stage=stage)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(flow_net.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            nn.utils.clip_grad_norm_(flow_net.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate_flow(
    flow_net: nn.Module,
    audio_conds: torch.Tensor,
    audio_latents: torch.Tensor,
    params_all: torch.Tensor,
    batch_size: int,
    stage: int = 99,
) -> float:
    """Validation: flow matching loss on held-out data."""
    flow_net.eval()
    total_loss = 0.0
    n_batches = 0

    for start in range(0, len(params_all), batch_size):
        cond = audio_conds[start:start + batch_size]
        lat = audio_latents[start:start + batch_size]
        params = params_all[start:start + batch_size]
        loss = flow_net.compute_loss(params, cond, lat, stage=stage)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


@torch.no_grad()
def validate_reconstruction(
    flow_net: nn.Module,
    conditioner: nn.Module,
    synth: nn.Module,
    val_audio: torch.Tensor,
    val_params: torch.Tensor,
    n_samples: int = 8,
) -> tuple[float, float]:
    """Evaluate: sample params from flow, render, compare audio."""
    idx = torch.randperm(len(val_audio))[:n_samples]
    target_audio = val_audio[idx]
    target_params = val_params[idx]

    # Pre-compute conditions
    cond_vec, cond_lats = conditioner(target_audio)

    # Sample params
    pred_params = sample_euler(flow_net, cond_vec, cond_lats, n_steps=20)

    # Render and compare
    pred_dicts = []
    target_dicts = []
    from loom.training.dataset import vector_to_params
    from loom.training.losses import multi_resolution_stft_loss, param_loss

    p_loss = param_loss(pred_params, target_params).item()

    # Render a few for audio comparison
    pred_p = vector_to_params(pred_params[:1])
    pred_p.pop("fx_routing", None)
    pred_audio = synth(pred_p)
    s_loss = multi_resolution_stft_loss(pred_audio, target_audio[:1]).item()

    return p_loss, s_loss


# ── Main ────────────────────────────────────────────────────────────────────


STAGE_NAMES = {0: "osc only", 1: "osc+filter", 2: "osc+filter+env", 3: "mild FX", 99: "full"}


def train(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # ── Build models ────────────────────────────────────────────────────
    print(f"Building conditioner ({args.frontend}, {args.n_bins} bins, {args.n_layers} layers)...")
    conditioner = Conditioner(
        frontend=args.frontend,
        n_bins=args.n_bins,
        d_model=args.d_model,
        d_cond=args.d_cond,
        n_layers=args.n_layers,
        n_queries=args.n_queries,
    ).to(DEVICE)
    n_cond_params = sum(p.numel() for p in conditioner.parameters())
    print(f"  Conditioner params: {n_cond_params:,}")

    print(f"Building flow network (DiT-{args.dit_blocks}, d_model={args.d_model})...")
    flow_net = FlowNetwork(
        d_model=args.d_model,
        nhead=args.nhead,
        n_dit_blocks=args.dit_blocks,
        d_cond=args.d_cond,
        dropout=args.dropout,
    ).to(DEVICE)
    n_flow_params = sum(p.numel() for p in flow_net.parameters())
    print(f"  Flow network params: {n_flow_params:,}")
    print(f"  Total params: {n_cond_params + n_flow_params:,}")

    # torch.compile for speed (optional)
    if args.compile and DEVICE.type == "cuda":
        conditioner = torch.compile(conditioner)
        flow_net = torch.compile(flow_net)
        print("  torch.compile enabled")

    # ── Optimizer ───────────────────────────────────────────────────────
    all_params = list(conditioner.parameters()) + list(flow_net.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr_min,
    )

    use_amp = args.amp and DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("  AMP enabled")

    # ── Synth for data generation ───────────────────────────────────────
    n_audio = int(SAMPLE_RATE * args.audio_duration)
    synth_gen = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
    synth_gen.eval()

    frontend = build_frontend(args.frontend, n_mels=128)
    fronend_name = args.frontend

    # ── Validation set (fixed) ──────────────────────────────────────────
    n_val = max(500, args.pool_size // 20 if args.pool_size > 0 else 1000)
    cur_stage = args.stage
    print(f"Generating validation set ({n_val} samples, stage {cur_stage})...")
    val_audio, val_params = generate_pool(
        n_val, synth_gen, frontend, args.gen_batch_size, DEVICE,
        audio_duration=args.audio_duration, stage=cur_stage,
    )
    print(f"  Pre-computing validation audio conditions...")
    val_conds, val_lats = precompute_conditions(conditioner, val_audio, args.cond_batch_size)
    print(f"  Done. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB" if DEVICE.type == "cuda" else "  Done.")

    # ── Training loop ───────────────────────────────────────────────────
    best_val = float("inf")
    patience_counter = 0
    t0 = time.perf_counter()

    if args.pool_size > 0:
        # ── Pool rotation mode ──────────────────────────────────────────
        print(f"\nPool mode: {args.pool_size} samples/pool, {args.pool_epochs} ep/pool, {args.epochs} epochs total")

        for epoch in range(args.epochs):
            # Regenerate pool periodically
            if epoch % args.pool_epochs == 0:
                pool_num = epoch // args.pool_epochs + 1
                t_pool = time.perf_counter()
                train_audio, train_params = generate_pool(
                    args.pool_size, synth_gen, frontend, args.gen_batch_size, DEVICE,
                    audio_duration=args.audio_duration, stage=cur_stage,
                )
                train_conds, train_lats = precompute_conditions(conditioner, train_audio, args.cond_batch_size)
                gen_time = time.perf_counter() - t_pool
                print(f"  Pool {pool_num}: {args.pool_size} samples in {gen_time:.1f}s")

            # Train one epoch
            train_loss = train_epoch(
                flow_net, train_conds, train_lats, train_params,
                args.batch_size, optimizer, scheduler, use_amp, scaler,
                stage=cur_stage,
            )

            # Validate
            val_loss = validate_flow(flow_net, val_conds, val_lats, val_params, args.batch_size, stage=cur_stage)

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save({
                    "conditioner": conditioner.state_dict(),
                    "flow_net": flow_net.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, data_dir / "best_flow.pt")
                marker = " *"
            else:
                patience_counter += 1

            if (epoch + 1) % args.log_every == 0 or marker:
                elapsed = time.perf_counter() - t0
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:4d}/{args.epochs}"
                    f" | loss={train_loss:.4f}/{val_loss:.4f}"
                    f" | lr={lr:.2e}"
                    f" | {elapsed:.0f}s{marker}"
                )

            if args.patience and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    else:
        # ── Fixed dataset mode ──────────────────────────────────────────
        if args.regenerate or not (data_dir / "dataset_meta.pt").exists():
            print(f"\nGenerating {args.n_samples} samples...")
            mels, param_vecs, audio_all = generate_dataset(
                args.n_samples,
                audio_duration=args.audio_duration,
                gen_batch_size=args.gen_batch_size,
                save_path=str(data_dir / "dataset"),
                device=DEVICE,
            )
        else:
            print(f"\nLoading dataset from {data_dir}")
            mels, param_vecs, audio_all = load_dataset(str(data_dir))

        n_total = len(mels)
        n_train = n_total - n_val

        # Use raw audio from dataset or re-render
        if audio_all is not None:
            train_audio_raw = audio_all[:n_train].to(DEVICE)
        else:
            # Fallback: re-render from params (slow, one-time cost)
            print("  Re-rendering audio from params...")
            train_audio_raw = torch.zeros(n_train, n_audio, device=DEVICE)
            from loom.training.dataset import vector_to_params as v2p
            for i in range(0, n_train, args.gen_batch_size):
                bs = min(args.gen_batch_size, n_train - i)
                p = v2p(param_vecs[i:i + bs].to(DEVICE))
                p.pop("fx_routing", None)
                with torch.no_grad():
                    train_audio_raw[i:i + bs] = synth_gen(p)

        train_params = param_vecs[:n_train].to(DEVICE)

        print(f"  Pre-computing training audio conditions ({n_train} samples)...")
        train_conds, train_lats = precompute_conditions(conditioner, train_audio_raw, args.cond_batch_size)
        print(f"  Done.")

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                flow_net, train_conds, train_lats, train_params,
                args.batch_size, optimizer, scheduler, use_amp, scaler,
                stage=cur_stage,
            )
            val_loss = validate_flow(flow_net, val_conds, val_lats, val_params, args.batch_size, stage=cur_stage)

            marker = ""
            if val_loss < best_val:
                best_val = val_loss
                patience_counter = 0
                torch.save({
                    "conditioner": conditioner.state_dict(),
                    "flow_net": flow_net.state_dict(),
                    "epoch": epoch,
                    "val_loss": val_loss,
                }, data_dir / "best_flow.pt")
                marker = " *"
            else:
                patience_counter += 1

            if (epoch + 1) % args.log_every == 0 or marker:
                elapsed = time.perf_counter() - t0
                lr = optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {epoch + 1:4d}/{args.epochs}"
                    f" | loss={train_loss:.4f}/{val_loss:.4f}"
                    f" | lr={lr:.2e}"
                    f" | {elapsed:.0f}s{marker}"
                )

            if args.patience and patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    # ── Final evaluation ────────────────────────────────────────────────
    elapsed = time.perf_counter() - t0
    print(f"\nTraining complete: {elapsed:.0f}s, best val loss: {best_val:.6f}")
    print(f"Model saved to {data_dir / 'best_flow.pt'}")

    # Quick reconstruction check
    print("\n--- Reconstruction check ---")
    synth_val = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
    synth_val.eval()
    p_loss, s_loss = validate_reconstruction(
        flow_net, conditioner, synth_val,
        val_audio, val_params, n_samples=8,
    )
    print(f"Val param loss: {p_loss:.4f}")
    print(f"Val spectral loss: {s_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train flow matching for synth parameter estimation")

    # Data
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--audio-duration", type=float, default=1.0)
    parser.add_argument("--data-dir", type=str, default="data_flow")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--gen-batch-size", type=int, default=64)

    # Pool rotation
    parser.add_argument("--pool-size", type=int, default=20000)
    parser.add_argument("--pool-epochs", type=int, default=5)

    # Frontend
    parser.add_argument("--frontend", type=str, default="cqt",
                        choices=["cqt", "gammatone", "mel", "multi", "multires"])

    # Training
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-min", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--compile", action="store_true")

    # Model
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-cond", type=int, default=512)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--n-bins", type=int, default=192,
                        help="Frequency bins for frontend")
    parser.add_argument("--n-layers", type=int, default=4,
                        help="Transformer encoder layers in conditioner")
    parser.add_argument("--n-queries", type=int, default=4,
                        help="Learnable query tokens in conditioner pool")
    parser.add_argument("--dit-blocks", type=int, default=6,
                        help="DiT backbone blocks")
    parser.add_argument("--dropout", type=float, default=0.1)

    # Data complexity
    parser.add_argument("--stage", type=int, default=99,
                        help="Curriculum stage (0-3, 99=full)")

    # Batch size for pre-computing conditions
    parser.add_argument("--cond-batch-size", type=int, default=64)

    args = parser.parse_args()
    train(args)
