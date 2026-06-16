"""Training script for synth parameter estimation.

Usage:
    # Fixed dataset (old mode):
    python -m loom.training.train --n-samples 50000 --epochs 200 --batch-size 512

    # Pool rotation (infinite unique samples):
    python -m loom.training.train --pool-size 50000 --pool-epochs 5 --epochs 500 --batch-size 1024
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
import torchaudio.transforms as T

from loom.core import DEVICE, SAMPLE_RATE
from loom.render import random_params
from loom.training.dataset import (
    generate_dataset, load_dataset, vector_to_params, params_to_vector,
)
from loom.training.encoder import ParamEncoder
from loom.training.losses import param_loss, multi_resolution_stft_loss
from loom.synth import SubtractiveSynth


def generate_pool(n_samples, synth, mel_transform, amp_to_db, gen_batch_size, device):
    """Generate a pool of (mels, params) directly on GPU. Zero disk I/O."""
    with torch.no_grad():
        probe_p = random_params(1, device=device)
        probe_p.pop("fx_routing", None)
        probe_m = amp_to_db(mel_transform(synth(probe_p)))
    n_mels, n_frames = probe_m.shape[1], probe_m.shape[2]
    n_pvec = params_to_vector(probe_p).shape[1]

    mels = torch.empty(n_samples, n_mels, n_frames, device=device)
    pvecs = torch.empty(n_samples, n_pvec, device=device)

    for offset in range(0, n_samples, gen_batch_size):
        bs = min(gen_batch_size, n_samples - offset)
        params = random_params(bs, device=device)
        params.pop("fx_routing", None)
        with torch.no_grad():
            audio = synth(params)
            mel = mel_transform(audio)
            mel_db = amp_to_db(mel)
            mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
        mels[offset:offset + bs] = mel_norm
        pvecs[offset:offset + bs] = params_to_vector(params)

    return mels, pvecs


def train_epoch(model, train_mels, train_params, bs, optimizer, scaler, use_amp,
                synth, train_audio, args, ema_state):
    """Run one training epoch. Returns average loss."""
    model.train()
    n_train = len(train_mels)
    perm = torch.randperm(n_train, device=train_mels.device)
    loss_acc = 0.0
    n_batches = 0

    for bi in range(0, n_train, bs):
        idx = perm[bi:bi + bs]
        mel = train_mels[idx]
        target = train_params[idx]

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(mel)

        pred = pred.float()
        loss = param_loss(pred, target)

        if synth is not None:
            cached_audio = train_audio[idx].to(mel.device) if train_audio is not None else None
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

            if ema_state["gn_param"] == 0.0:
                ema_state["gn_param"] = gn_p
                ema_state["gn_spectral"] = gn_s
            else:
                ema_state["gn_param"] = 0.99 * ema_state["gn_param"] + 0.01 * gn_p
                ema_state["gn_spectral"] = 0.99 * ema_state["gn_spectral"] + 0.01 * gn_s

            alpha = args.spectral_ratio * ema_state["gn_param"] / (ema_state["gn_spectral"] + 1e-8)
            gi = 0
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(alpha).add_(saved_grads[gi])
                    gi += 1

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.item() + alpha * l_spectral.item()
        else:
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_acc += loss.item()
        n_batches += 1

    return loss_acc / n_batches


def validate(model, val_mels, val_params, bs):
    """Run validation. Returns average loss."""
    model.eval()
    loss_acc = 0.0
    n_batches = 0
    with torch.no_grad():
        for vi in range(0, len(val_mels), bs):
            pred = model(val_mels[vi:vi + bs])
            loss_acc += param_loss(pred, val_params[vi:vi + bs]).item()
            n_batches += 1
    return loss_acc / n_batches


def train(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

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

    synth_for_loss = None
    if args.spectral:
        n_audio = int(SAMPLE_RATE * args.audio_duration)
        synth_for_loss = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
        synth_for_loss.eval()
        print(f"Spectral loss enabled (ratio={args.spectral_ratio})")

    use_amp = args.amp and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP enabled (mixed precision)")

    ema_state = {"gn_param": 0.0, "gn_spectral": 0.0}
    best_val = float("inf")
    patience_counter = 0
    t0 = time.perf_counter()

    # ── Pool rotation mode ──────────────────────────────────────
    if args.pool_size > 0:
        n_audio = int(SAMPLE_RATE * args.audio_duration)
        synth_gen = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
        synth_gen.eval()
        mel_transform = T.MelSpectrogram(
            sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256,
            n_mels=128, power=2.0,
        ).to(DEVICE)
        amp_to_db = T.AmplitudeToDB(top_db=80)

        n_val = max(1000, args.pool_size // 10)
        print(f"Generating fixed validation set ({n_val} samples)...")
        val_mels, val_params = generate_pool(
            n_val, synth_gen, mel_transform, amp_to_db,
            args.gen_batch_size, DEVICE,
        )
        print(f"Pool mode: {args.pool_size} samples/pool, {args.pool_epochs} epochs/pool")

        epoch = 0
        pool_round = 0
        while epoch < args.epochs:
            pool_round += 1
            t_pool = time.perf_counter()
            train_mels, train_params = generate_pool(
                args.pool_size, synth_gen, mel_transform, amp_to_db,
                args.gen_batch_size, DEVICE,
            )
            gen_time = time.perf_counter() - t_pool
            vram_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  pool {pool_round}: generated {args.pool_size} in {gen_time:.1f}s ({vram_gb:.1f} GB)")

            for pe in range(args.pool_epochs):
                if epoch >= args.epochs:
                    break

                train_loss = train_epoch(
                    model, train_mels, train_params, args.batch_size,
                    optimizer, scaler, use_amp, synth_for_loss, None, args, ema_state,
                )
                val_loss = validate(model, val_mels, val_params, args.batch_size)
                scheduler.step()
                epoch += 1

                marker = ""
                if val_loss < best_val:
                    best_val = val_loss
                    torch.save(model.state_dict(), data_dir / "best_encoder.pt")
                    patience_counter = 0
                    marker = " *"
                else:
                    patience_counter += 1

                if epoch % args.log_every == 0 or marker:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"Epoch {epoch:3d}/{args.epochs} (pool {pool_round})"
                        f" | train {train_loss:.6f} | val {val_loss:.6f}"
                        f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                        f" | {elapsed:.0f}s{marker}"
                    )

                if args.patience and patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            del train_mels, train_params
            torch.cuda.empty_cache()

            if args.patience and patience_counter >= args.patience:
                break

    # ── Fixed dataset mode ──────────────────────────────────────
    else:
        if args.regenerate or not (data_dir / "dataset_meta.pt").exists():
            print(f"Generating {args.n_samples} samples (duration={args.audio_duration}s)...")
            mels, param_vecs, target_audio_all = generate_dataset(
                args.n_samples,
                audio_duration=args.audio_duration,
                gen_batch_size=args.gen_batch_size,
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
        train_mels = mels[train_idx].to(DEVICE)
        train_params = param_vecs[train_idx].to(DEVICE)
        val_mels = mels[val_idx].to(DEVICE)
        val_params = param_vecs[val_idx].to(DEVICE)
        train_audio = target_audio_all[train_idx] if has_cached_audio else None

        del mels, param_vecs, target_audio_all
        vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"Train: {n_train}, Val: {n_val} (data on GPU: {vram_gb:.1f} GB)")

        for epoch in range(args.epochs):
            train_loss = train_epoch(
                model, train_mels, train_params, args.batch_size,
                optimizer, scaler, use_amp, synth_for_loss, train_audio, args, ema_state,
            )
            val_loss = validate(model, val_mels, val_params, args.batch_size)
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
                alpha_str = (
                    f" | α {args.spectral_ratio * ema_state['gn_param'] / (ema_state['gn_spectral'] + 1e-8):.2e}"
                    if synth_for_loss is not None else ""
                )
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
    # Data
    parser.add_argument("--n-samples", type=int, default=10000)
    parser.add_argument("--audio-duration", type=float, default=1.0)
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--gen-batch-size", type=int, default=64)
    # Pool rotation
    parser.add_argument("--pool-size", type=int, default=0,
                        help="Pool size for on-the-fly generation (0=use fixed dataset)")
    parser.add_argument("--pool-epochs", type=int, default=5,
                        help="Epochs to train per pool before regeneration")
    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--spectral", action="store_true", help="Enable spectral loss")
    parser.add_argument("--spectral-ratio", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    # Model
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    args = parser.parse_args()
    train(args)
