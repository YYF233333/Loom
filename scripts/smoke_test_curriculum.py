"""Curriculum smoke test: can the encoder learn pitch + waveform with effects off?

If loss decreases: problem is data complexity, curriculum learning is the fix.
If loss stays flat: encoder or loss function has a deeper bug.

Usage:
    python scripts/smoke_test_curriculum.py --data-dir /root/autodl-tmp/loom-data
"""

import argparse
import sys
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
sys.stdout.reconfigure(line_buffering=True)

import torch
import torchaudio.transforms as T

from loom.core import DEVICE, SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.training.encoder import ParamEncoder


# ── Stage definitions ───────────────────────────────────────────────
# Each stage: (name, list of param keys to learn, param generator overrides)

STAGES = [
    {
        "name": "pitch+waveform only",
        "learn_keys": ["osc_pitch", "osc_waveform"],
        "desc": "Effects off, filter bypass, envelope fixed. Pure oscillator.",
    },
    {
        "name": "pitch+waveform+filter",
        "learn_keys": ["osc_pitch", "osc_waveform", "filter_cutoff", "filter_q", "filter_type"],
        "desc": "Add filter. Still no effects.",
    },
    {
        "name": "pitch+waveform+filter+envelope",
        "learn_keys": [
            "osc_pitch", "osc_waveform",
            "filter_cutoff", "filter_q", "filter_type",
            "amp_attack", "amp_decay", "amp_sustain", "amp_release",
        ],
        "desc": "Add amplitude envelope.",
    },
]


def random_params_stage(batch, stage_idx, device):
    """Generate params with complexity matching the stage."""
    def _rand(shape):
        return torch.rand(shape, device=device)

    def _one_hot_rand(batch, n):
        idx = torch.randint(0, n, (batch,), device=device)
        return torch.nn.functional.one_hot(idx, n).float()

    # Always randomize core oscillator params
    params = {
        "osc_pitch": _rand((batch,)),
        "osc_waveform": _one_hot_rand(batch, 4),
        "osc_detune": torch.full((batch,), 0.5, device=device),  # no detune
        "osc_type": torch.nn.functional.one_hot(
            torch.zeros(batch, dtype=torch.long, device=device), 3
        ).float(),  # always additive
        "wt_position": torch.full((batch,), 0.5, device=device),
        "fm_carrier_ratio": torch.full((batch,), 0.5, device=device),
        "fm_mod_ratio": torch.full((batch,), 0.5, device=device),
        "fm_mod_index": torch.zeros(batch, device=device),
    }

    # LFO off
    params["lfo_rate"] = torch.full((batch,), 0.5, device=device)
    params["lfo_depth"] = torch.zeros(batch, device=device)
    params["lfo_waveform"] = _one_hot_rand(batch, 4)
    params["lfo_target"] = torch.nn.functional.one_hot(
        torch.zeros(batch, dtype=torch.long, device=device), 4
    ).float()
    params["lfo_phase"] = torch.zeros(batch, device=device)

    # Filter: stage 0 = bypass (cutoff max, mix=0), stage 1+ = random
    if stage_idx >= 1:
        params["filter_cutoff"] = _rand((batch,))
        params["filter_q"] = _rand((batch,))
        params["filter_type"] = _one_hot_rand(batch, 3)
        params["filter_mix"] = torch.ones(batch, device=device)
    else:
        params["filter_cutoff"] = torch.ones(batch, device=device)
        params["filter_q"] = torch.full((batch,), 0.3, device=device)
        params["filter_type"] = torch.nn.functional.one_hot(
            torch.zeros(batch, dtype=torch.long, device=device), 3
        ).float()
        params["filter_mix"] = torch.zeros(batch, device=device)

    # Filter envelope: fixed
    params["filt_env_attack"] = torch.full((batch,), 0.1, device=device)
    params["filt_env_decay"] = torch.full((batch,), 0.3, device=device)
    params["filt_env_sustain"] = torch.full((batch,), 0.7, device=device)
    params["filt_env_release"] = torch.full((batch,), 0.3, device=device)
    params["filt_env_amount"] = torch.full((batch,), 0.5, device=device)

    # Amplitude envelope: stage 0-1 = fixed, stage 2+ = random
    if stage_idx >= 2:
        params["amp_attack"] = _rand((batch,))
        params["amp_decay"] = _rand((batch,))
        params["amp_sustain"] = _rand((batch,))
        params["amp_release"] = _rand((batch,))
    else:
        params["amp_attack"] = torch.full((batch,), 0.05, device=device)
        params["amp_decay"] = torch.full((batch,), 0.3, device=device)
        params["amp_sustain"] = torch.full((batch,), 0.8, device=device)
        params["amp_release"] = torch.full((batch,), 0.3, device=device)

    params["master_gain"] = torch.full((batch,), 0.7, device=device)

    # Effects: ALL OFF (mix = 0)
    params["dist_amount"] = torch.zeros(batch, device=device)
    params["dist_mix"] = torch.zeros(batch, device=device)
    params["comp_threshold"] = torch.full((batch,), 0.5, device=device)
    params["comp_ratio"] = torch.full((batch,), 0.3, device=device)
    params["comp_attack"] = torch.full((batch,), 0.3, device=device)
    params["comp_release"] = torch.full((batch,), 0.3, device=device)
    params["comp_makeup"] = torch.full((batch,), 0.5, device=device)
    params["comp_mix"] = torch.zeros(batch, device=device)
    params["chorus_rate"] = torch.full((batch,), 0.3, device=device)
    params["chorus_depth"] = torch.full((batch,), 0.3, device=device)
    params["chorus_mix"] = torch.zeros(batch, device=device)
    params["delay_time"] = torch.full((batch,), 0.3, device=device)
    params["delay_feedback"] = torch.zeros(batch, device=device)
    params["delay_mix"] = torch.zeros(batch, device=device)
    params["reverb_room_size"] = torch.full((batch,), 0.3, device=device)
    params["reverb_decay"] = torch.full((batch,), 0.3, device=device)
    params["reverb_damping"] = torch.full((batch,), 0.5, device=device)
    params["reverb_mix"] = torch.zeros(batch, device=device)
    params["eq_low_gain"] = torch.full((batch,), 0.5, device=device)
    params["eq_mid_gain"] = torch.full((batch,), 0.5, device=device)
    params["eq_high_gain"] = torch.full((batch,), 0.5, device=device)

    return params


def params_to_subset_vector(params, learn_keys):
    """Extract only the keys we want to learn into a flat vector."""
    from loom.training.dataset import CONTINUOUS_KEYS, CATEGORICAL_KEYS
    parts = []
    for key in CONTINUOUS_KEYS:
        if key in learn_keys:
            parts.append(params[key].unsqueeze(1))
    for key, _ in CATEGORICAL_KEYS:
        if key in learn_keys:
            parts.append(params[key])
    return torch.cat(parts, dim=1)


def subset_loss(pred_full, target_subset, learn_keys):
    """Compute loss only on the subset of params we're learning."""
    from loom.training.dataset import CONTINUOUS_KEYS, CATEGORICAL_KEYS, N_CONTINUOUS

    cont_indices = []
    for i, key in enumerate(CONTINUOUS_KEYS):
        if key in learn_keys:
            cont_indices.append(i)

    cat_groups = []
    idx = N_CONTINUOUS
    for key, n in CATEGORICAL_KEYS:
        if key in learn_keys:
            cat_groups.append((idx, n))
        idx += n

    # Extract predicted subset
    pred_parts = []
    for i in cont_indices:
        pred_parts.append(pred_full[:, i:i+1])

    target_idx = 0
    n_cont = len(cont_indices)

    if n_cont > 0:
        pred_cont = torch.cat(pred_parts, dim=1)
        target_cont = target_subset[:, :n_cont]
        cont_loss = (pred_cont - target_cont).pow(2).mean()
    else:
        cont_loss = torch.tensor(0.0, device=pred_full.device)

    cat_parts = []
    target_offset = n_cont
    for pred_start, n in cat_groups:
        pred_logp = pred_full[:, pred_start:pred_start+n].clamp(1e-7, 1.0).log()
        target_p = target_subset[:, target_offset:target_offset+n]
        cat_parts.append((-target_p * pred_logp).sum(dim=-1).mean())
        target_offset += n

    if cat_parts:
        cat_loss = torch.stack(cat_parts).mean()
    else:
        cat_loss = torch.tensor(0.0, device=pred_full.device)

    return cont_loss + 0.5 * cat_loss


def generate_pool(n_samples, stage_idx, synth, mel_transform, amp_to_db, gen_bs, device):
    learn_keys = STAGES[stage_idx]["learn_keys"]

    with torch.no_grad():
        probe = random_params_stage(1, stage_idx, device)
        probe_mel = amp_to_db(mel_transform(synth(probe)))
    n_mels, n_frames = probe_mel.shape[1], probe_mel.shape[2]
    n_target = params_to_subset_vector(probe, learn_keys).shape[1]

    mels = torch.empty(n_samples, n_mels, n_frames, device=device)
    targets = torch.empty(n_samples, n_target, device=device)

    for offset in range(0, n_samples, gen_bs):
        bs = min(gen_bs, n_samples - offset)
        params = random_params_stage(bs, stage_idx, device)
        with torch.no_grad():
            audio = synth(params)
            mel = amp_to_db(mel_transform(audio))
            mel_norm = ((mel + 80.0) / 80.0).clamp(0.0, 1.0)
        mels[offset:offset+bs] = mel_norm
        targets[offset:offset+bs] = params_to_subset_vector(params, learn_keys)

    return mels, targets


def run_stage(stage_idx, args):
    stage = STAGES[stage_idx]
    learn_keys = stage["learn_keys"]
    print(f"\n{'='*60}")
    print(f"STAGE {stage_idx}: {stage['name']}")
    print(f"  {stage['desc']}")
    print(f"  Learning: {learn_keys}")
    print(f"{'='*60}")

    n_audio = int(SAMPLE_RATE * 1.0)
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
    synth.eval()

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256,
        n_mels=128, power=2.0,
    ).to(DEVICE)
    amp_to_db = T.AmplitudeToDB(top_db=80)

    model = ParamEncoder(d_model=args.d_model, d_state=64, n_layers=args.n_layers).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Encoder: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Generate validation set
    val_mels, val_targets = generate_pool(
        1000, stage_idx, synth, mel_transform, amp_to_db, args.gen_batch_size, DEVICE
    )

    best_val = float("inf")
    t0 = time.perf_counter()

    for epoch in range(args.epochs):
        # Generate fresh training pool each epoch group
        if epoch % args.pool_epochs == 0:
            train_mels, train_targets = generate_pool(
                args.pool_size, stage_idx, synth, mel_transform, amp_to_db,
                args.gen_batch_size, DEVICE,
            )

        # Train
        model.train()
        perm = torch.randperm(len(train_mels), device=DEVICE)
        loss_acc = 0.0
        n_batches = 0
        for bi in range(0, len(train_mels), args.batch_size):
            idx = perm[bi:bi+args.batch_size]
            mel = train_mels[idx]
            target = train_targets[idx]

            pred = model(mel)
            loss = subset_loss(pred, target, learn_keys)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            loss_acc += loss.item()
            n_batches += 1
        scheduler.step()

        train_loss = loss_acc / n_batches

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(val_mels)
            val_loss = subset_loss(val_pred, val_targets, learn_keys).item()

        marker = ""
        if val_loss < best_val:
            best_val = val_loss
            marker = " *"

        if (epoch + 1) % 5 == 0 or marker:
            elapsed = time.perf_counter() - t0
            print(
                f"  Epoch {epoch+1:3d}/{args.epochs}"
                f" | train {train_loss:.6f} | val {val_loss:.6f}"
                f" | lr {optimizer.param_groups[0]['lr']:.2e}"
                f" | {elapsed:.0f}s{marker}"
            )

    print(f"\n  RESULT: best val = {best_val:.6f}")
    if best_val < 0.1:
        print(f"  ✓ Stage {stage_idx} PASSED — model can learn {stage['name']}")
    else:
        print(f"  ✗ Stage {stage_idx} FAILED — loss too high, check encoder/loss")
    return best_val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stages", type=int, default=3, help="How many stages to run (1-3)")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--pool-size", type=int, default=10000)
    parser.add_argument("--pool-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gen-batch-size", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=160)
    parser.add_argument("--n-layers", type=int, default=6)
    parser.add_argument("--data-dir", type=str, default="data")
    args = parser.parse_args()

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")

    for stage_idx in range(min(args.stages, len(STAGES))):
        result = run_stage(stage_idx, args)
        if result > 0.1:
            print(f"\nStopping: stage {stage_idx} didn't converge, fix before proceeding.")
            break


if __name__ == "__main__":
    main()
