"""Smoke test for V100: measure speed and memory."""
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
from loom.training.encoder import ParamEncoder, HAS_MAMBA
from loom.training.losses import param_loss, multi_resolution_stft_loss

DEVICE = torch.device("cuda")
bs = 512
n_audio = int(SAMPLE_RATE * 1.0)

model = ParamEncoder(d_model=160, d_state=64, n_layers=6).to(DEVICE)
model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_params = sum(p.numel() for p in model.parameters())

params = random_params(bs, device=DEVICE)
synth = SubtractiveSynth(SAMPLE_RATE, n_audio).to(DEVICE)
synth.eval()
mel_transform = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=1024, hop_length=256, n_mels=128, power=2.0,
).to(DEVICE)
amp_to_db = T.AmplitudeToDB(top_db=80)
with torch.no_grad():
    audio = synth(params)
    mel = mel_transform(audio)
    mel_db = amp_to_db(mel)
    mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)
target_vec = params_to_vector(params).detach()

backend = "Mamba" if HAS_MAMBA else "S4D"
print(f"Encoder: {n_params:,} params ({backend})")
print(f"Batch: {bs}, GPU: {torch.cuda.get_device_name()}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
print()

# --- param-only benchmark ---
torch.cuda.reset_peak_memory_stats()
for _ in range(5):
    pred = model(mel_norm)
    loss = param_loss(pred, target_vec)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(50):
    pred = model(mel_norm)
    loss = param_loss(pred, target_vec)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
param_ms = (time.perf_counter() - t0) / 50 * 1000
param_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Param-only:  {param_ms:.1f} ms/step, peak {param_mem:.2f} GB")

# --- spectral benchmark (smaller batch for synth VRAM) ---
bs_spec = 64
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
mel_spec = mel_norm[:bs_spec]
target_spec = target_vec[:bs_spec]
with torch.no_grad():
    cached_audio = synth(random_params(bs_spec, device=DEVICE))

for _ in range(3):
    pred = model(mel_spec)
    loss = param_loss(pred, target_spec)
    p = vector_to_params(pred)
    p.pop("fx_routing", None)
    pa = synth(p)
    loss = loss + 0.1 * multi_resolution_stft_loss(pa, cached_audio)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(20):
    pred = model(mel_spec)
    loss = param_loss(pred, target_spec)
    p = vector_to_params(pred)
    p.pop("fx_routing", None)
    pa = synth(p)
    loss = loss + 0.1 * multi_resolution_stft_loss(pa, cached_audio)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
torch.cuda.synchronize()
spec_ms = (time.perf_counter() - t0) / 20 * 1000
spec_mem = torch.cuda.max_memory_allocated() / 1e9
print(f"Spectral:    {spec_ms:.1f} ms/step (batch={bs_spec}), peak {spec_mem:.2f} GB")
print(f"VRAM free:   {torch.cuda.get_device_properties(0).total_memory / 1e9 - spec_mem:.1f} GB")

print()
print("--- vs GTX 1050 Ti baseline ---")
print(f"Param-only:  {param_ms:.1f} ms (was ~90 ms on 1050 Ti) -> {90/param_ms:.1f}x faster")
print(f"Spectral:    {spec_ms:.1f} ms (was ~150 ms on 1050 Ti) -> {150/spec_ms:.1f}x faster")
