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
from loom.training.losses import weighted_param_loss, multi_resolution_stft_loss, signal_chain_loss
from loom.synth import SubtractiveSynth


def _build_model(args):
    """Construct the encoder model based on --arch flag."""
    arch = args.arch
    if arch == "transformer":
        from loom.training.encoder_v2 import TransformerParamEncoder
        model = TransformerParamEncoder(
            d_model=args.d_model, n_layers=args.n_layers,
        )
    elif arch == "hybrid":
        from loom.training.encoder_v2 import HybridParamEncoder
        model = HybridParamEncoder(
            d_model=args.d_model, d_state=args.d_state,
        )
    elif arch == "v1":
        from loom.training.encoder import ParamEncoder
        model = ParamEncoder(
            d_model=args.d_model, d_state=args.d_state, n_layers=args.n_layers,
        )
    else:
        raise ValueError(f"Unknown arch: {arch!r}")
    return model


def _build_optimizer(model, args):
    """Build AdamW optimizer, using per-group lr when the model supports it."""
    if hasattr(model, "param_groups_for_optimizer"):
        param_groups = model.param_groups_for_optimizer(args.lr)
    else:
        param_groups = model.parameters()
    return torch.optim.AdamW(param_groups, lr=args.lr, weight_decay=0.01)


def generate_pool(n_samples, synth, mel_transform, amp_to_db, gen_batch_size, device, stage=99):
    """Generate a pool of (mels, params) directly on GPU. Zero disk I/O."""
    with torch.no_grad():
        probe_p = random_params(1, device=device, stage=stage)
        probe_p.pop("fx_routing", None)
        probe_m = amp_to_db(mel_transform(synth(probe_p)))
    n_mels, n_frames = probe_m.shape[1], probe_m.shape[2]
    n_pvec = params_to_vector(probe_p).shape[1]

    mels = torch.empty(n_samples, n_mels, n_frames, device=device)
    pvecs = torch.empty(n_samples, n_pvec, device=device)

    for offset in range(0, n_samples, gen_batch_size):
        bs = min(gen_batch_size, n_samples - offset)
        params = random_params(bs, device=device, stage=stage)
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
    p_loss_acc, s_loss_acc = 0.0, 0.0
    n_batches = 0

    for bi in range(0, n_train, bs):
        idx = perm[bi:bi + bs]
        mel = train_mels[idx]
        target = train_params[idx]

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred = model(mel)

        pred = pred.float()
        loss_p = weighted_param_loss(pred, target)

        if synth is not None:
            cached_audio = train_audio[idx].to(mel.device) if train_audio is not None else None

            # --- Step 1: backward param loss, save gradients ---
            optimizer.zero_grad()
            loss_p.backward(retain_graph=True)
            gn_p = torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None
            ]).norm().item()
            saved_grads = [
                p.grad.detach().clone() for p in model.parameters()
                if p.grad is not None
            ]

            # --- Step 2: spectral loss with gradient accumulation over sub-batches ---
            model.zero_grad()
            spectral_batch_size = args.spectral_batch_size
            n_sub = max(1, (len(idx) + spectral_batch_size - 1) // spectral_batch_size)
            l_spectral_total = None

            sub_starts = list(range(0, len(idx), spectral_batch_size))
            for si_idx, sub_start in enumerate(sub_starts):
                sub_slice = slice(sub_start, sub_start + spectral_batch_size)
                pred_sub = pred[sub_slice]
                is_last = (si_idx == len(sub_starts) - 1)

                pred_p_sub = vector_to_params(pred_sub)
                pred_p_sub.pop("fx_routing", None)
                pred_result_sub = synth(pred_p_sub, return_intermediates=True)
                pred_audio_sub, pred_inter_sub = pred_result_sub

                with torch.no_grad():
                    target_sub = target[sub_slice]
                    target_p_sub = vector_to_params(target_sub)
                    target_p_sub.pop("fx_routing", None)
                    target_result_sub = synth(target_p_sub, return_intermediates=True)
                    if cached_audio is not None:
                        target_audio_sub = cached_audio[sub_slice]
                    else:
                        target_audio_sub = target_result_sub[0]
                    target_inter_sub = target_result_sub[1]

                l_spec_sub = multi_resolution_stft_loss(pred_audio_sub, target_audio_sub)
                l_chain_sub = signal_chain_loss(pred_inter_sub, target_inter_sub)
                l_sub = (l_spec_sub + l_chain_sub) / n_sub
                l_sub.backward(retain_graph=not is_last)

                if l_spectral_total is None:
                    l_spectral_total = l_sub.detach()
                else:
                    l_spectral_total = l_spectral_total + l_sub.detach()

            gn_s = torch.stack([
                p.grad.norm() for p in model.parameters() if p.grad is not None
            ]).norm().item()

            # --- Step 3: GradNorm EMA balancing ---
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
            p_loss_acc += loss_p.item()
            s_loss_acc += l_spectral_total.item()
        else:
            optimizer.zero_grad()
            scaler.scale(loss_p).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            p_loss_acc += loss_p.item()
        n_batches += 1

    return p_loss_acc / n_batches, s_loss_acc / n_batches


def validate(model, val_mels, val_params, bs, synth=None):
    """Run validation. Returns (param_loss, spectral_loss) averages."""
    model.eval()
    p_acc, s_acc = 0.0, 0.0
    n_batches = 0
    with torch.no_grad():
        for vi in range(0, len(val_mels), bs):
            pred = model(val_mels[vi:vi + bs])
            target = val_params[vi:vi + bs]
            p_acc += weighted_param_loss(pred, target).item()

            if synth is not None:
                pred_p = vector_to_params(pred)
                pred_p.pop("fx_routing", None)
                pred_audio, pred_inter = synth(pred_p, return_intermediates=True)
                target_p = vector_to_params(target)
                target_p.pop("fx_routing", None)
                target_audio, target_inter = synth(target_p, return_intermediates=True)
                s_acc += (multi_resolution_stft_loss(pred_audio, target_audio)
                          + signal_chain_loss(pred_inter, target_inter)).item()
            n_batches += 1
    return p_acc / n_batches, s_acc / n_batches


STAGE_NAMES = {0: "osc only", 1: "osc+filter", 2: "osc+filter+env", 3: "mild FX", 99: "full"}


def get_curriculum_stage(epoch, args):
    """Determine curriculum stage for the current epoch."""
    if not args.curriculum:
        return args.stage
    stage_len = args.stage_epochs if args.stage_epochs > 0 else max(1, args.epochs // 4)
    stage = min(epoch // stage_len, 3)
    return stage


def train(args):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # --- model ---
    model = _build_model(args).to(DEVICE)
    if args.compile:
        model = torch.compile(model)
        print("torch.compile enabled")
    n_model_params = sum(p.numel() for p in model.parameters())
    print(f"Encoder params: {n_model_params:,} (arch={args.arch})")

    if args.resume:
        ckpt_path = Path(args.resume)
        if not ckpt_path.exists():
            ckpt_path = data_dir / args.resume
        state = torch.load(ckpt_path, weights_only=True)
        model.load_state_dict(state)
        print(f"Resumed from {ckpt_path}")

    optimizer = _build_optimizer(model, args)
    sched_len = args.stage_epochs if args.curriculum and args.stage_epochs > 0 else args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=sched_len, eta_min=args.lr_min,
    )

    synth_for_loss = None
    if args.spectral:
        n_audio = int(SAMPLE_RATE * args.audio_duration)
        synth_for_loss = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
        synth_for_loss.eval()
        print(f"Spectral loss enabled (ratio={args.spectral_ratio}, sub-batch={args.spectral_batch_size})")

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
        cur_stage = get_curriculum_stage(0, args)
        print(f"Generating fixed validation set ({n_val} samples, stage {cur_stage}: {STAGE_NAMES.get(cur_stage, '?')})...")
        val_mels, val_params = generate_pool(
            n_val, synth_gen, mel_transform, amp_to_db,
            args.gen_batch_size, DEVICE, stage=cur_stage,
        )
        if args.curriculum:
            print(f"Curriculum mode: stages 0→3, pool {args.pool_size}, {args.pool_epochs} ep/pool")
        else:
            print(f"Pool mode: {args.pool_size} samples/pool, {args.pool_epochs} epochs/pool, stage={cur_stage}")

        def advance_stage(cur, epoch):
            """Advance curriculum stage: reset lr, scheduler, val set."""
            remaining = args.epochs - epoch
            stage_len = min(
                args.stage_epochs if args.stage_epochs > 0 else max(1, args.epochs // 4),
                remaining,
            )
            print(f"\n>>> CURRICULUM STAGE {cur}: {STAGE_NAMES.get(cur, '?')} (lr {args.lr:.0e}→{args.lr_min:.0e}, {stage_len} ep) <<<")
            # Respect per-group lr multipliers when resetting
            if hasattr(model, "param_groups_for_optimizer"):
                reference_groups = model.param_groups_for_optimizer(args.lr)
                ref_by_name = {g["name"]: g["lr"] for g in reference_groups}
                for pg in optimizer.param_groups:
                    name = pg.get("name")
                    if name is not None and name in ref_by_name:
                        pg["lr"] = ref_by_name[name]
                    else:
                        pg["lr"] = args.lr
            else:
                for pg in optimizer.param_groups:
                    pg["lr"] = args.lr
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=stage_len, eta_min=args.lr_min,
            )
            new_val_mels, new_val_params = generate_pool(
                n_val, synth_gen, mel_transform, amp_to_db,
                args.gen_batch_size, DEVICE, stage=cur,
            )
            return sched, new_val_mels, new_val_params

        epoch = 0
        pool_round = 0
        stage_history = []
        while epoch < args.epochs:
            new_stage = max(cur_stage, get_curriculum_stage(epoch, args))
            if new_stage != cur_stage:
                cur_stage = new_stage
                stage_history = []
                scheduler, val_mels, val_params = advance_stage(cur_stage, epoch)
                ema_state = {"gn_param": 0.0, "gn_spectral": 0.0}
                best_val = float("inf")
                patience_counter = 0

            pool_round += 1
            t_pool = time.perf_counter()
            train_mels, train_params = generate_pool(
                args.pool_size, synth_gen, mel_transform, amp_to_db,
                args.gen_batch_size, DEVICE, stage=cur_stage,
            )
            gen_time = time.perf_counter() - t_pool
            vram_gb = torch.cuda.memory_allocated() / 1e9
            print(f"  pool {pool_round}: generated {args.pool_size} in {gen_time:.1f}s ({vram_gb:.1f} GB)")

            advance_now = False
            for pe in range(args.pool_epochs):
                if epoch >= args.epochs:
                    break

                train_p, train_s = train_epoch(
                    model, train_mels, train_params, args.batch_size,
                    optimizer, scaler, use_amp, synth_for_loss, None, args, ema_state,
                )
                val_p, val_s = validate(model, val_mels, val_params, args.batch_size,
                                        synth=synth_for_loss)
                scheduler.step()
                epoch += 1

                marker = ""
                if val_p < best_val:
                    best_val = val_p
                    torch.save(model.state_dict(), data_dir / "best_encoder.pt")
                    patience_counter = 0
                    marker = " *"
                else:
                    patience_counter += 1

                if epoch % args.log_every == 0 or marker:
                    elapsed = time.perf_counter() - t0
                    stage_str = f" S{cur_stage}" if args.curriculum or args.stage != 99 else ""
                    s_str = f" s={train_s:.4f}/{val_s:.4f}" if synth_for_loss else ""
                    print(
                        f"Epoch {epoch:3d}/{args.epochs} (pool {pool_round}){stage_str}"
                        f" | p={train_p:.4f}/{val_p:.4f}{s_str}"
                        f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                        f" | {elapsed:.0f}s{marker}"
                    )

                # Curriculum: advance when relative improvement stalls
                if args.curriculum and cur_stage < 3:
                    stage_history.append(val_p)
                    window = args.stage_patience
                    if len(stage_history) >= window * 2:
                        old_best = min(stage_history[-window * 2:-window])
                        new_best = min(stage_history[-window:])
                        improvement = (old_best - new_best) / (abs(old_best) + 1e-8)
                        if improvement < args.stage_min_improvement:
                            print(f"  Stage {cur_stage} plateau: improvement {improvement:.4%} < {args.stage_min_improvement:.1%} over {window} ep, advancing...")
                            cur_stage += 1
                            scheduler, val_mels, val_params = advance_stage(cur_stage, epoch)
                            ema_state = {"gn_param": 0.0, "gn_spectral": 0.0}
                            best_val = float("inf")
                            patience_counter = 0
                            stage_history = []
                            advance_now = True
                            break

                # Global early stopping
                if args.patience and patience_counter >= args.patience:
                    print(f"Early stopping at epoch {epoch}")
                    advance_now = False
                    break

            del train_mels, train_params
            torch.cuda.empty_cache()

            if advance_now:
                continue
            if args.patience and patience_counter >= args.patience and not args.curriculum:
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
            train_p, train_s = train_epoch(
                model, train_mels, train_params, args.batch_size,
                optimizer, scaler, use_amp, synth_for_loss, train_audio, args, ema_state,
            )
            val_p, val_s = validate(model, val_mels, val_params, args.batch_size,
                                     synth=synth_for_loss)
            scheduler.step()

            marker = ""
            if val_p < best_val:
                best_val = val_p
                torch.save(model.state_dict(), data_dir / "best_encoder.pt")
                patience_counter = 0
                marker = " *"
            else:
                patience_counter += 1

            if (epoch + 1) % args.log_every == 0 or marker:
                elapsed = time.perf_counter() - t0
                s_str = f" s={train_s:.4f}/{val_s:.4f}" if synth_for_loss else ""
                print(
                    f"Epoch {epoch + 1:3d}/{args.epochs}"
                    f" | p={train_p:.4f}/{val_p:.4f}{s_str}"
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
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--lr-min", type=float, default=3e-5,
                        help="Minimum lr for cosine decay (default 3e-5, i.e. 10x decay)")
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--spectral", action="store_true", help="Enable spectral loss")
    parser.add_argument("--spectral-ratio", type=float, default=1.0)
    parser.add_argument("--spectral-batch-size", type=int, default=32,
                        help="Sub-batch size for spectral loss gradient accumulation (default 32)")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--compile", action="store_true", help="torch.compile the model")
    # Curriculum
    parser.add_argument("--curriculum", action="store_true",
                        help="Enable curriculum learning (stage 0→1→2→3)")
    parser.add_argument("--stage", type=int, default=99,
                        help="Fixed curriculum stage (0-3, 99=full). Ignored with --curriculum")
    parser.add_argument("--stage-epochs", type=int, default=0,
                        help="Epochs per curriculum stage (0=auto: epochs/4)")
    parser.add_argument("--stage-patience", type=int, default=20,
                        help="Window size for relative improvement check (compare best of last N vs previous N)")
    parser.add_argument("--stage-min-improvement", type=float, default=0.01,
                        help="Minimum relative improvement to stay in current stage (default 1%%)")
    # Model
    parser.add_argument("--arch", type=str, default="transformer",
                        choices=["transformer", "hybrid", "v1"],
                        help="Encoder architecture: transformer, hybrid (Mamba+Attn), or v1 (legacy)")
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--d-state", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=6)
    args = parser.parse_args()
    train(args)
