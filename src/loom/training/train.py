"""Training script for synth parameter estimation.

Usage:
    python -m loom.training.train                    # defaults: 10K samples, 100 epochs
    python -m loom.training.train --n-samples 50000 --epochs 200 --batch-size 64
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

from loom.core import DEVICE, SAMPLE_RATE
from loom.training.dataset import generate_dataset, load_dataset, vector_to_params
from loom.training.encoder import ParamEncoder
from loom.training.losses import param_loss, multi_resolution_stft_loss
from loom.synth import SubtractiveSynth


def train(args):
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.7)
        torch.cuda.empty_cache()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- dataset ---
    if args.regenerate or not (data_dir / "dataset_meta.pt").exists():
        print(f"Generating {args.n_samples} samples (duration={args.audio_duration}s)...")
        mels, param_vecs, target_audio_all = generate_dataset(
            args.n_samples,
            audio_duration=args.audio_duration,
            gen_batch_size=8,
            save_path=str(data_dir / "dataset"),
            device=DEVICE,
        )
    else:
        print(f"Loading dataset from {data_dir}")
        mels, param_vecs, target_audio_all = load_dataset(str(data_dir))

    n_total = len(mels)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    has_cached_audio = target_audio_all is not None and args.spectral

    # Mels + params fit on GPU (~9 GB for 100K); audio stays on CPU
    train_mels = mels[train_idx].to(DEVICE)
    train_params = param_vecs[train_idx].to(DEVICE)
    val_mels = mels[val_idx].to(DEVICE)
    val_params = param_vecs[val_idx].to(DEVICE)
    train_audio = target_audio_all[train_idx] if has_cached_audio else None

    del mels, param_vecs, target_audio_all
    vram_gb = torch.cuda.memory_allocated() / 1e9
    print(f"Train: {n_train}, Val: {n_val} (data on GPU: {vram_gb:.1f} GB)")

    # --- model ---
    model = ParamEncoder(
        d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
    ).to(DEVICE)
    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled")
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {n_model_params:,}")

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            ckpt_path = data_dir / args.resume
        state = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {ckpt_path}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    synth = None
    if args.spectral:
        n_audio = int(SAMPLE_RATE * args.audio_duration)
        synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
        synth.eval()
        print(f"Spectral loss enabled (ratio={args.spectral_ratio})")

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP enabled (mixed precision)")

    ema_gn_param = 0.0
    ema_gn_spectral = 0.0
    ema_beta = 0.99

    best_val = float("inf")
    patience_counter = 0
    t0 = time.perf_counter()

    bs = args.batch_size
    n_train_batches = (n_train + bs - 1) // bs

    for epoch in range(args.epochs):
        # --- train ---
        model.train()
        train_loss_acc = 0.0
        n_batches = 0
        perm = torch.randperm(n_train, device=DEVICE)

        for bi in range(n_train_batches):
            idx = perm[bi * bs : (bi + 1) * bs]
            mel = train_mels[idx]
            target = train_params[idx]
            cached_audio = train_audio[idx].to(DEVICE) if train_audio is not None else None

            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(mel)

            pred = pred.float()
            loss = param_loss(pred, target)

            if synth is not None:
                pred_p = vector_to_params(pred)
                pred_p.pop("fx_routing", None)
                pred_audio = synth(pred_p)
                if cached_audio is not None:
                    target_audio = cached_audio
                else:
                    with torch.no_grad():
                        target_p = vector_to_params(target)
                        target_p.pop("fx_routing", None)
                        target_audio = synth(target_p)
                l_spectral = multi_resolution_stft_loss(pred_audio, target_audio)

                # Two-stage backward with EMA gradient balancing
                optimizer.zero_grad()
                loss.backward(retain_graph=True)
                gn_p = torch.stack([
                    p.grad.norm() for p in model.parameters() if p.grad is not None
                ]).norm().item()
                saved_grads = [
                    p.grad.detach().clone() for p in model.parameters()
                    if p.grad is not None
                ]

                model.zero_grad()
                l_spectral.backward()
                gn_s = torch.stack([
                    p.grad.norm() for p in model.parameters() if p.grad is not None
                ]).norm().item()

                if ema_gn_param == 0.0:
                    ema_gn_param = gn_p
                    ema_gn_spectral = gn_s
                else:
                    ema_gn_param = ema_beta * ema_gn_param + (1 - ema_beta) * gn_p
                    ema_gn_spectral = ema_beta * ema_gn_spectral + (1 - ema_beta) * gn_s

                alpha = args.spectral_ratio * ema_gn_param / (ema_gn_spectral + 1e-8)
                gi = 0
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(alpha).add_(saved_grads[gi])
                        gi += 1

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_acc += (loss.item() + alpha * l_spectral.item())
            else:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                train_loss_acc += loss.item()
            n_batches += 1

        train_loss = train_loss_acc / n_batches

        # --- validate ---
        model.eval()
        val_loss_acc = 0.0
        val_batches = 0
        with torch.no_grad():
            for vi in range(0, len(val_mels), bs):
                val_pred = model(val_mels[vi:vi + bs])
                val_loss_acc += param_loss(val_pred, val_params[vi:vi + bs]).item()
                val_batches += 1
        val_loss = val_loss_acc / val_batches

        scheduler.step()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), data_dir / "best_encoder.pt")
            patience_counter = 0
            marker = " *"
        else:
            patience_counter += 1

        if (epoch + 1) % args.log_every == 0 or marker:
            elapsed = time.perf_counter() - t0
            alpha_str = f" | α {args.spectral_ratio * ema_gn_param / (ema_gn_spectral + 1e-8):.2e}" if synth is not None else ""
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs}"
                f" | train {train_loss:.6f} | val {val_loss:.6f}"
                f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                f"{alpha_str}"
                f" | {elapsed:.0f}s{marker}"
            )

        if args.patience and patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    elapsed = time.perf_counter() - t0
    print(f"\nBest val loss: {best_val:.6f} ({elapsed:.0f}s total)")
    print(f"Model saved to {data_dir / 'best_encoder.pt'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train synth parameter encoder")
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--audio-duration", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--spectral", action="store_true", help="Enable spectral loss")
    parser.add_argument("--spectral-ratio", type=float, default=1.0,
                        help="Spectral/param gradient balance ratio (1.0 = equal)")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume from (or filename in data-dir)")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (fp16)")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    args = parser.parse_args()
    train(args)
