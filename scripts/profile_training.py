"""Profile training pipeline: measure per-component timing and memory usage.

Usage:
    python -m scripts.profile_training
    python -m scripts.profile_training --batch-size 32 --device cuda
"""

import argparse
import time
import sys
from contextlib import contextmanager

sys.stdout.reconfigure(line_buffering=True)

import torch
import torchaudio.transforms as T

from loom.core import SAMPLE_RATE, DEVICE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import generate_dataset, vector_to_params, params_to_vector
from loom.training.encoder import ParamEncoder
from loom.training.losses import param_loss, multi_resolution_stft_loss


@contextmanager
def timer(label, results):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    yield
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) * 1000
    results[label] = results.get(label, [])
    results[label].append(elapsed)


def gpu_mem_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1e6
    return 0.0


def gpu_peak_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1e6
    return 0.0


def profile_synth_forward(batch_size, device, audio_duration=1.0, n_warmup=3, n_iter=10):
    """Profile each component of SubtractiveSynth.forward()."""
    print(f"\n{'='*70}")
    print(f"SYNTH FORWARD PASS PROFILING (batch={batch_size}, device={device})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * audio_duration)
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(device)
    synth.eval()

    results = {}

    for iteration in range(n_warmup + n_iter):
        params = random_params(batch_size, device=device)
        is_warmup = iteration < n_warmup

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        if is_warmup:
            with torch.no_grad():
                _ = synth(params)
            continue

        with torch.no_grad():
            # --- LFO ---
            with timer("lfo", results):
                lfo_signal = synth.lfo(
                    params["lfo_rate"], params["lfo_depth"],
                    params["lfo_waveform"], params["lfo_phase"],
                )
                lfo_target = params["lfo_target"]
                pitch_lfo = lfo_target[:, 1:2] * lfo_signal * 0.05
                freq_mod = 1.0 + pitch_lfo

            # --- Oscillators ---
            with timer("additive_osc", results):
                additive_out = synth.oscillator(
                    params["osc_pitch"], params["osc_waveform"], params["osc_detune"],
                )

            with timer("wavetable_osc", results):
                wavetable_out = synth.wavetable_osc(
                    params["osc_pitch"], params["osc_detune"], params["wt_position"],
                )

            with timer("fm_osc", results):
                fm_out = synth.fm_osc(
                    params["osc_pitch"], params["osc_detune"],
                    params["fm_carrier_ratio"], params["fm_mod_ratio"], params["fm_mod_index"],
                )

            with timer("osc_blend", results):
                osc_type = params["osc_type"]
                audio = (
                    osc_type[:, 0:1] * additive_out
                    + osc_type[:, 1:2] * wavetable_out
                    + osc_type[:, 2:3] * fm_out
                )

            # --- Filter envelope ---
            with timer("filter_envelope", results):
                filt_env = synth.filter_envelope(
                    params["filt_env_attack"], params["filt_env_decay"],
                    params["filt_env_sustain"], params["filt_env_release"],
                )
                amount = (params["filt_env_amount"] - 0.5) * 2.0
                base_cutoff = params["filter_cutoff"].unsqueeze(1)
                env_mod = amount.unsqueeze(1) * filt_env * 0.3
                lfo_cutoff = lfo_target[:, 0:1] * lfo_signal * 0.3
                cutoff_signal = (base_cutoff + env_mod + lfo_cutoff).clamp(0.0, 1.0)

            # --- SVFilter ---
            with timer("svfilter", results):
                audio = synth.filter(audio, cutoff_signal, params["filter_q"],
                                     params["filter_type"], mix=params.get("filter_mix"))

            # --- Amplitude envelope + VCA ---
            with timer("amp_env_vca", results):
                amp_env = synth.amp_envelope(
                    params["amp_attack"], params["amp_decay"],
                    params["amp_sustain"], params["amp_release"],
                )
                audio = synth.vca(audio, amp_env, params["master_gain"])

            # --- Effects chain (canonical order, no routing) ---
            fx_params = {
                "dist_drive": params["dist_amount"].unsqueeze(1),
                "dist_mix": params["dist_mix"],
                "comp_threshold": params["comp_threshold"],
                "comp_ratio": params["comp_ratio"],
                "comp_attack": params["comp_attack"],
                "comp_release": params["comp_release"],
                "comp_makeup": params["comp_makeup"],
                "comp_mix": params["comp_mix"],
                "chorus_rate": params["chorus_rate"],
                "chorus_depth": params["chorus_depth"],
                "chorus_mix": params["chorus_mix"],
                "delay_time": params["delay_time"],
                "delay_feedback": params["delay_feedback"],
                "delay_mix": params["delay_mix"],
                "reverb_room_size": params["reverb_room_size"],
                "reverb_decay": params["reverb_decay"],
                "reverb_damping": params["reverb_damping"],
                "reverb_mix": params["reverb_mix"],
                "eq_low_gain": params["eq_low_gain"],
                "eq_mid_gain": params["eq_mid_gain"],
                "eq_high_gain": params["eq_high_gain"],
            }

            with timer("fx_distortion", results):
                audio_dist = synth.effects_chain.distortion(audio, fx_params["dist_drive"], fx_params["dist_mix"])

            with timer("fx_compressor", results):
                audio_comp = synth.effects_chain.compressor(audio, fx_params["comp_threshold"],
                    fx_params["comp_ratio"], fx_params["comp_attack"], fx_params["comp_release"],
                    fx_params["comp_makeup"], fx_params["comp_mix"])

            with timer("fx_chorus", results):
                audio_chorus = synth.effects_chain.chorus(audio, fx_params["chorus_rate"],
                    fx_params["chorus_depth"], fx_params["chorus_mix"])

            with timer("fx_delay", results):
                audio_delay = synth.effects_chain.delay(audio, fx_params["delay_time"],
                    fx_params["delay_feedback"], fx_params["delay_mix"])

            with timer("fx_reverb", results):
                audio_reverb = synth.effects_chain.reverb(audio, fx_params["reverb_room_size"],
                    fx_params["reverb_decay"], fx_params["reverb_damping"], fx_params["reverb_mix"])

            with timer("fx_eq", results):
                audio_eq = synth.effects_chain.eq(audio, fx_params["eq_low_gain"],
                    fx_params["eq_mid_gain"], fx_params["eq_high_gain"])

            # --- Full effects chain (canonical order) ---
            with timer("effects_chain_canonical", results):
                audio2 = audio.clone()
                audio2 = synth.effects_chain(audio2, fx_params, routing_logits=None)

            # --- Full effects chain (Sinkhorn routing) ---
            routing_logits = torch.randn(batch_size, 6, 6, device=device)
            with timer("effects_chain_sinkhorn", results):
                audio3 = audio.clone()
                audio3 = synth.effects_chain(audio3, fx_params, routing_logits=routing_logits, tau=1.0)

            # --- Full synth forward ---
            with timer("synth_full_forward", results):
                _ = synth(params)

    print_results(results)
    return results


def profile_synth_backward(batch_size, device, audio_duration=1.0, n_warmup=2, n_iter=5):
    """Profile backward pass through synth (as used in spectral loss)."""
    print(f"\n{'='*70}")
    print(f"SYNTH BACKWARD PASS PROFILING (batch={batch_size}, device={device})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * audio_duration)
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(device)
    synth.eval()

    results = {}

    for iteration in range(n_warmup + n_iter):
        params = random_params(batch_size, device=device)
        for v in params.values():
            if isinstance(v, torch.Tensor) and v.is_floating_point():
                v.requires_grad_(True)

        is_warmup = iteration < n_warmup

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        with timer("forward_with_grad" if not is_warmup else "_warmup_fwd", results):
            audio = synth(params)
            loss = audio.pow(2).mean()

        if is_warmup:
            loss.backward()
            continue

        mem_after_fwd = gpu_mem_mb()

        with timer("backward", results):
            loss.backward()

        mem_after_bwd = gpu_peak_mb()
        if not is_warmup:
            results.setdefault("mem_fwd_mb", []).append(mem_after_fwd)
            results.setdefault("mem_peak_mb", []).append(mem_after_bwd)

    print_results(results)
    return results


def profile_training_step(batch_size, device, audio_duration=1.0, spectral_weight=0.1,
                          n_warmup=2, n_iter=5):
    """Profile a full training step as in train.py."""
    print(f"\n{'='*70}")
    print(f"TRAINING STEP PROFILING (batch={batch_size}, spectral_weight={spectral_weight})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * audio_duration)
    model = ParamEncoder().to(device)
    model.train()

    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(device)
    synth.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    mel_transform = T.MelSpectrogram(
        sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=128, power=2.0,
    ).to(device)
    amp_to_db = T.AmplitudeToDB(top_db=80)

    # Generate one batch of data
    params = random_params(batch_size, device=device)
    with torch.no_grad():
        audio = synth(params)
        mel = mel_transform(audio)
        mel_db = amp_to_db(mel)
        mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
    target = params_to_vector(params).detach()

    results = {}

    for iteration in range(n_warmup + n_iter):
        is_warmup = iteration < n_warmup
        tag = "_warmup" if is_warmup else ""

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # --- Encoder forward ---
        with timer(f"encoder_forward{tag}", results):
            pred = model(mel_norm)

        # --- Param loss ---
        with timer(f"param_loss{tag}", results):
            loss = param_loss(pred, target)

        # --- Spectral loss (the expensive part) ---
        if spectral_weight > 0:
            with timer(f"vector_to_params{tag}", results):
                pred_p = vector_to_params(pred)
                pred_p.pop("fx_routing", None)

            with timer(f"synth_pred_audio{tag}", results):
                pred_audio = synth(pred_p)

            with timer(f"synth_target_audio{tag}", results):
                with torch.no_grad():
                    target_p = vector_to_params(target)
                    target_p.pop("fx_routing", None)
                    target_audio = synth(target_p)

            with timer(f"stft_loss{tag}", results):
                spectral_loss = multi_resolution_stft_loss(pred_audio, target_audio)
                loss = loss + spectral_weight * spectral_loss

        # --- Backward ---
        with timer(f"backward{tag}", results):
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # --- Optimizer step ---
        with timer(f"optimizer_step{tag}", results):
            optimizer.step()

        if not is_warmup:
            results.setdefault("peak_mem_mb", []).append(gpu_peak_mb())

        # --- Full training step (end-to-end) ---
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        pred2 = model(mel_norm)
        loss2 = param_loss(pred2, target)
        if spectral_weight > 0:
            p2 = vector_to_params(pred2)
            p2.pop("fx_routing", None)
            pa2 = synth(p2)
            with torch.no_grad():
                tp2 = vector_to_params(target)
                tp2.pop("fx_routing", None)
                ta2 = synth(tp2)
            loss2 = loss2 + spectral_weight * multi_resolution_stft_loss(pa2, ta2)
        optimizer.zero_grad()
        loss2.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        if not is_warmup:
            results.setdefault("full_step", []).append(elapsed)

    print_results(results)
    return results


def profile_loss_functions(batch_size, device, audio_duration=1.0, n_iter=10):
    """Profile loss computation."""
    print(f"\n{'='*70}")
    print(f"LOSS FUNCTION PROFILING (batch={batch_size})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * audio_duration)
    pred = torch.randn(batch_size, 97, device=device, requires_grad=True)
    target = torch.randn(batch_size, 97, device=device)
    audio_pred = torch.randn(batch_size, n_audio, device=device, requires_grad=True)
    audio_target = torch.randn(batch_size, n_audio, device=device)

    results = {}
    for _ in range(n_iter):
        with timer("param_loss", results):
            l1 = param_loss(pred, target)

        with timer("stft_loss_512", results):
            _ = multi_resolution_stft_loss(audio_pred, audio_target, fft_sizes=[512])

        with timer("stft_loss_1024", results):
            _ = multi_resolution_stft_loss(audio_pred, audio_target, fft_sizes=[1024])

        with timer("stft_loss_2048", results):
            _ = multi_resolution_stft_loss(audio_pred, audio_target, fft_sizes=[2048])

        with timer("stft_loss_multi", results):
            _ = multi_resolution_stft_loss(audio_pred, audio_target)

    print_results(results)
    return results


def profile_dataset_generation(device, n_samples=100, gen_batch_sizes=[8, 32, 64]):
    """Profile dataset generation with different batch sizes."""
    print(f"\n{'='*70}")
    print(f"DATASET GENERATION PROFILING ({n_samples} samples)")
    print(f"{'='*70}")

    for gbs in gen_batch_sizes:
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

        t0 = time.perf_counter()
        mels, params = generate_dataset(
            n_samples, audio_duration=1.0, gen_batch_size=gbs, device=device,
        )
        elapsed = time.perf_counter() - t0
        peak = gpu_peak_mb()
        print(f"  gen_batch_size={gbs:3d}: {elapsed:.2f}s total, "
              f"{elapsed/n_samples*1000:.1f}ms/sample, peak={peak:.0f}MB")


def profile_sinkhorn_overhead(batch_size, device, n_iter=5):
    """Compare canonical vs Sinkhorn routing overhead."""
    print(f"\n{'='*70}")
    print(f"SINKHORN ROUTING OVERHEAD (batch={batch_size})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * 1.0)
    synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(device)
    synth.eval()

    results = {}
    for _ in range(n_iter):
        params = random_params(batch_size, device=device)
        with torch.no_grad():
            # Run through synth up to effects input
            lfo_signal = synth.lfo(params["lfo_rate"], params["lfo_depth"],
                                   params["lfo_waveform"], params["lfo_phase"])
            lfo_target = params["lfo_target"]
            additive_out = synth.oscillator(params["osc_pitch"], params["osc_waveform"], params["osc_detune"])
            wavetable_out = synth.wavetable_osc(params["osc_pitch"], params["osc_detune"], params["wt_position"])
            fm_out = synth.fm_osc(params["osc_pitch"], params["osc_detune"],
                                  params["fm_carrier_ratio"], params["fm_mod_ratio"], params["fm_mod_index"])
            osc_type = params["osc_type"]
            audio = osc_type[:, 0:1] * additive_out + osc_type[:, 1:2] * wavetable_out + osc_type[:, 2:3] * fm_out
            filt_env = synth.filter_envelope(params["filt_env_attack"], params["filt_env_decay"],
                                             params["filt_env_sustain"], params["filt_env_release"])
            amount = (params["filt_env_amount"] - 0.5) * 2.0
            base_cutoff = params["filter_cutoff"].unsqueeze(1)
            env_mod = amount.unsqueeze(1) * filt_env * 0.3
            lfo_cutoff = lfo_target[:, 0:1] * lfo_signal * 0.3
            cutoff_signal = (base_cutoff + env_mod + lfo_cutoff).clamp(0.0, 1.0)
            audio = synth.filter(audio, cutoff_signal, params["filter_q"], params["filter_type"],
                                 mix=params.get("filter_mix"))
            amp_env = synth.amp_envelope(params["amp_attack"], params["amp_decay"],
                                         params["amp_sustain"], params["amp_release"])
            audio = synth.vca(audio, amp_env, params["master_gain"])

            fx_params = {
                "dist_drive": params["dist_amount"].unsqueeze(1),
                "dist_mix": params["dist_mix"],
                "comp_threshold": params["comp_threshold"],
                "comp_ratio": params["comp_ratio"],
                "comp_attack": params["comp_attack"],
                "comp_release": params["comp_release"],
                "comp_makeup": params["comp_makeup"],
                "comp_mix": params["comp_mix"],
                "chorus_rate": params["chorus_rate"],
                "chorus_depth": params["chorus_depth"],
                "chorus_mix": params["chorus_mix"],
                "delay_time": params["delay_time"],
                "delay_feedback": params["delay_feedback"],
                "delay_mix": params["delay_mix"],
                "reverb_room_size": params["reverb_room_size"],
                "reverb_decay": params["reverb_decay"],
                "reverb_damping": params["reverb_damping"],
                "reverb_mix": params["reverb_mix"],
                "eq_low_gain": params["eq_low_gain"],
                "eq_mid_gain": params["eq_mid_gain"],
                "eq_high_gain": params["eq_high_gain"],
            }

            routing_logits = torch.randn(batch_size, 6, 6, device=device)

            with timer("canonical_6fx", results):
                _ = synth.effects_chain(audio.clone(), fx_params, routing_logits=None)

            with timer("sinkhorn_6x6=36fx", results):
                _ = synth.effects_chain(audio.clone(), fx_params, routing_logits=routing_logits, tau=1.0)

    canonical = sum(results["canonical_6fx"]) / len(results["canonical_6fx"])
    sinkhorn = sum(results["sinkhorn_6x6=36fx"]) / len(results["sinkhorn_6x6=36fx"])
    print(f"\n  Canonical (6 effects):    {canonical:.1f} ms")
    print(f"  Sinkhorn  (36 effects):   {sinkhorn:.1f} ms")
    print(f"  Overhead:                 {sinkhorn/canonical:.1f}x")


def profile_reverb_solve(batch_size, device, n_iter=5):
    """Profile reverb's linalg.solve specifically."""
    print(f"\n{'='*70}")
    print(f"REVERB torch.linalg.solve PROFILING (batch={batch_size})")
    print(f"{'='*70}")

    n_audio = int(SAMPLE_RATE * 1.0)
    from loom.effects.reverb import Reverb
    reverb = Reverb(SAMPLE_RATE, n_audio).to(device)

    signal = torch.randn(batch_size, n_audio, device=device)
    room = torch.rand(batch_size, device=device)
    decay = torch.rand(batch_size, device=device)
    damp = torch.rand(batch_size, device=device)
    mix = torch.ones(batch_size, device=device) * 0.5

    results = {}
    for _ in range(n_iter):
        with timer("reverb_full", results):
            with torch.no_grad():
                _ = reverb(signal, room, decay, damp, mix)

    n_freq = n_audio // 2 + 1
    print(f"\n  n_freq = {n_freq}")
    print(f"  linalg.solve shape: ({batch_size}, {n_freq}, 4, 4)")
    print(f"  Total 4x4 solves per call: {batch_size * n_freq:,}")
    print_results(results)


def profile_memory_scaling(device, batch_sizes=[8, 16, 32, 64, 128]):
    """Profile memory usage at different batch sizes."""
    print(f"\n{'='*70}")
    print(f"MEMORY SCALING BY BATCH SIZE")
    print(f"{'='*70}")

    if not torch.cuda.is_available():
        print("  Skipped (CPU mode)")
        return

    n_audio = int(SAMPLE_RATE * 1.0)

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(device)
        synth.eval()
        params = random_params(bs, device=device)

        try:
            with torch.no_grad():
                _ = synth(params)
            peak = gpu_peak_mb()
            print(f"  batch_size={bs:4d}: peak={peak:.0f} MB")
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"  batch_size={bs:4d}: OOM")
                torch.cuda.empty_cache()
            else:
                raise

        del synth, params


def print_results(results):
    print(f"\n  {'Component':<30s} {'Mean (ms)':>10s} {'Std (ms)':>10s} {'% Total':>10s}")
    print(f"  {'-'*60}")

    total_key = None
    for k in ["synth_full_forward", "full_step"]:
        if k in results:
            total_key = k
            break

    total_mean = None
    if total_key:
        total_mean = sum(results[total_key]) / len(results[total_key])

    for label, times in sorted(results.items()):
        if label.startswith("_") or label.endswith("_mb"):
            continue
        mean = sum(times) / len(times)
        if len(times) > 1:
            std = (sum((t - mean) ** 2 for t in times) / (len(times) - 1)) ** 0.5
        else:
            std = 0.0
        pct = f"{mean / total_mean * 100:.1f}%" if total_mean else ""
        print(f"  {label:<30s} {mean:>10.1f} {std:>10.1f} {pct:>10s}")

    for label in ["mem_fwd_mb", "mem_peak_mb", "peak_mem_mb"]:
        if label in results:
            vals = results[label]
            mean = sum(vals) / len(vals)
            print(f"  {label:<30s} {mean:>10.0f} MB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=str(DEVICE))
    parser.add_argument("--audio-duration", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device(args.device)
    bs = args.batch_size

    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Sample rate: {SAMPLE_RATE}, audio duration: {args.audio_duration}s")
    print(f"Samples per audio: {int(SAMPLE_RATE * args.audio_duration)}")

    profile_synth_forward(bs, device, args.audio_duration)
    profile_reverb_solve(bs, device)
    profile_sinkhorn_overhead(bs, device)
    profile_loss_functions(bs, device, args.audio_duration)
    profile_synth_backward(bs, device, args.audio_duration)
    profile_training_step(bs, device, args.audio_duration, spectral_weight=0.0)
    profile_training_step(bs, device, args.audio_duration, spectral_weight=0.1)
    profile_dataset_generation(device, n_samples=100, gen_batch_sizes=[8, 32, 64])
    profile_memory_scaling(device)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
