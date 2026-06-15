"""Training script for synth parameter estimation.

Usage:
    python -m loom.training.train                    # defaults: 10K samples, 100 epochs
    python -m loom.training.train --n-samples 50000 --epochs 200 --batch-size 64
"""

import argparse
import time
from pathlib import Path

import torch

from loom.core import DEVICE, SAMPLE_RATE
from loom.training.dataset import generate_dataset, vector_to_params
from loom.training.encoder import ParamEncoder
from loom.training.losses import param_loss, multi_resolution_stft_loss
from loom.synth import SubtractiveSynth


def train(args):
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    data_path = data_dir / "synth_dataset.pt"

    # --- dataset ---
    if data_path.exists() and not args.regenerate:
        print(f"Loading dataset from {data_path}")
        data = torch.load(data_path, weights_only=True)
        mels, param_vecs = data["mels"], data["params"]
    else:
        print(f"Generating {args.n_samples} samples (duration={args.audio_duration}s)...")
        mels, param_vecs = generate_dataset(
            args.n_samples,
            audio_duration=args.audio_duration,
            gen_batch_size=8,
            save_path=str(data_path),
            device=DEVICE,
        )

    # Move entire dataset to GPU
    n_total = len(mels)
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val

    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    train_mels = mels[train_idx].to(DEVICE)
    train_params = param_vecs[train_idx].to(DEVICE)
    val_mels = mels[val_idx].to(DEVICE)
    val_params = param_vecs[val_idx].to(DEVICE)

    del mels, param_vecs
    dataset_mb = (train_mels.nelement() + val_mels.nelement()) * 4 / 1e6
    print(f"Dataset on GPU: {dataset_mb:.0f} MB | Train: {n_train}, Val: {n_val}")

    # --- model ---
    model = ParamEncoder().to(DEVICE)
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {n_model_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    synth = None
    if args.spectral_weight > 0:
        n_audio = int(SAMPLE_RATE * args.audio_duration)
        synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
        synth.eval()
        print(f"Spectral loss enabled (weight={args.spectral_weight})")

    best_val = float("inf")
    patience_counter = 0
    bs = args.batch_size
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        # --- train: direct tensor indexing, no DataLoader ---
        model.train()
        shuffle = torch.randperm(n_train, device=DEVICE)
        train_loss_acc = torch.tensor(0.0, device=DEVICE)
        n_batches = 0

        for start in range(0, n_train, bs):
            idx = shuffle[start:start + bs]
            mel = train_mels[idx]
            target = train_params[idx]

            pred = model(mel)
            loss = param_loss(pred, target)

            if synth is not None:
                from loom.effects.chain import routing_temperature
                tau = routing_temperature(epoch, args.epochs)
                pred_p = vector_to_params(pred)
                pred_p["fx_routing_tau"] = torch.tensor(tau)
                pred_audio = synth(pred_p)
                with torch.no_grad():
                    target_p = vector_to_params(target)
                    target_p["fx_routing_tau"] = torch.tensor(tau)
                    target_audio = synth(target_p)
                loss = loss + args.spectral_weight * multi_resolution_stft_loss(
                    pred_audio, target_audio,
                )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_acc += loss.detach()
            n_batches += 1

        train_loss = (train_loss_acc / n_batches).item()

        # --- validate ---
        model.eval()
        with torch.no_grad():
            val_pred = model(val_mels)
            val_loss = param_loss(val_pred, val_params).item()

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
            print(
                f"Epoch {epoch + 1:3d}/{args.epochs}"
                f" | train {train_loss:.6f} | val {val_loss:.6f}"
                f" | lr {optimizer.param_groups[0]['lr']:.2e}"
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
    parser.add_argument("--spectral-weight", type=float, default=0.0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    train(args)
