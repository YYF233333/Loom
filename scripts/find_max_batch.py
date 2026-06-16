"""Find max batch size for param-only and spectral modes on current GPU."""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import time
import torch
import torchaudio.transforms as T

from loom.core import SAMPLE_RATE
from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.training.dataset import vector_to_params, params_to_vector
from loom.training.encoder import ParamEncoder
from loom.training.losses import param_loss, multi_resolution_stft_loss

DEVICE = torch.device("cuda")
n_audio = int(SAMPLE_RATE * 1.0)
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9

model = ParamEncoder(d_model=160, d_state=64, n_layers=6).to(DEVICE)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
synth.eval()
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=128, power=2.0,
).to(DEVICE)
amp_to_db = T.AmplitudeToDB(top_db=80)

n_params = sum(p.numel() for p in model.parameters())
print(f"GPU: {torch.cuda.get_device_name()}, VRAM: {total_vram:.1f} GB")
print(f"Encoder: {n_params:,} params")
print(f"vCPU: {os.cpu_count()}")
print()


def make_batch(bs):
    params = random_params(bs, device=DEVICE)
    with torch.no_grad():
        audio = synth(params)
        mel = mel_transform(audio)
        mel_db = amp_to_db(mel)
        mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
    target = params_to_vector(params).detach()
    return mel_norm, target, audio


def bench_param_only(bs, n_iter=30):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mel, target, _ = make_batch(bs)
    for _ in range(3):
        pred = model(mel)
        loss = param_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pred = model(mel)
        loss = param_loss(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return ms, mem


def bench_spectral(bs, n_iter=10):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    mel, target, cached_audio = make_batch(bs)
    for _ in range(2):
        pred = model(mel)
        loss = param_loss(pred, target)
        p = vector_to_params(pred)
        p.pop("fx_routing", None)
        pa = synth(p)
        loss = loss + 0.1 * multi_resolution_stft_loss(pa, cached_audio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        pred = model(mel)
        loss = param_loss(pred, target)
        p = vector_to_params(pred)
        p.pop("fx_routing", None)
        pa = synth(p)
        loss = loss + 0.1 * multi_resolution_stft_loss(pa, cached_audio)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iter * 1000
    mem = torch.cuda.max_memory_allocated() / 1e9
    return ms, mem


print("=== Param-only ===")
print(f"{'batch':>6s} {'ms/step':>10s} {'ms/sample':>10s} {'VRAM GB':>10s} {'util%':>8s}")
for bs in [128, 256, 512, 1024, 2048]:
    try:
        ms, mem = bench_param_only(bs)
        print(f"{bs:>6d} {ms:>10.1f} {ms/bs:>10.3f} {mem:>10.2f} {mem/total_vram*100:>7.1f}%")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{bs:>6d}        OOM")
        break

print()
print("=== Spectral (with synth forward+backward) ===")
print(f"{'batch':>6s} {'ms/step':>10s} {'ms/sample':>10s} {'VRAM GB':>10s} {'util%':>8s}")
for bs in [8, 16, 32, 64, 128, 256]:
    try:
        ms, mem = bench_spectral(bs)
        print(f"{bs:>6d} {ms:>10.1f} {ms/bs:>10.3f} {mem:>10.2f} {mem/total_vram*100:>7.1f}%")
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        print(f"{bs:>6d}        OOM")
        break
