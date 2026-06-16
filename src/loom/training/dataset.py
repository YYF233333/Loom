import torch
from torch.utils.data import Dataset
import torchaudio.transforms as T

from loom.synth import SubtractiveSynth
from loom.render import random_params
from loom.core import SAMPLE_RATE

CONTINUOUS_KEYS = [
    "osc_pitch", "osc_detune", "wt_position",
    "fm_carrier_ratio", "fm_mod_ratio", "fm_mod_index",
    "amp_attack", "amp_decay", "amp_sustain", "amp_release",
    "filter_cutoff", "filter_q", "filter_mix",
    "filt_env_attack", "filt_env_decay", "filt_env_sustain",
    "filt_env_release", "filt_env_amount",
    "dist_amount", "dist_mix", "master_gain",
    "comp_threshold", "comp_ratio", "comp_attack", "comp_release",
    "comp_makeup", "comp_mix",
    "chorus_rate", "chorus_depth", "chorus_mix",
    "delay_time", "delay_feedback", "delay_mix",
    "reverb_room_size", "reverb_decay", "reverb_damping", "reverb_mix",
    "eq_low_gain", "eq_mid_gain", "eq_high_gain",
    "lfo_rate", "lfo_depth", "lfo_phase",
]

CATEGORICAL_KEYS = [
    ("osc_waveform", 4),
    ("osc_type", 3),
    ("filter_type", 3),
    ("lfo_waveform", 4),
    ("lfo_target", 4),
]

N_ROUTING = 36  # 6x6 fx_routing logits
N_CONTINUOUS = len(CONTINUOUS_KEYS)
N_CATEGORICAL = sum(n for _, n in CATEGORICAL_KEYS)
N_PARAMS = N_CONTINUOUS + N_CATEGORICAL + N_ROUTING


def params_to_vector(params: dict[str, torch.Tensor]) -> torch.Tensor:
    parts = []
    for key in CONTINUOUS_KEYS:
        parts.append(params[key].unsqueeze(1))
    for key, _ in CATEGORICAL_KEYS:
        parts.append(params[key])
    if "fx_routing" in params:
        batch = parts[0].shape[0]
        parts.append(params["fx_routing"].reshape(batch, N_ROUTING))
    else:
        batch = parts[0].shape[0]
        parts.append(torch.zeros(batch, N_ROUTING, device=parts[0].device))
    return torch.cat(parts, dim=1)


def vector_to_params(vector: torch.Tensor) -> dict[str, torch.Tensor]:
    params = {}
    idx = 0
    for key in CONTINUOUS_KEYS:
        params[key] = vector[:, idx]
        idx += 1
    for key, n in CATEGORICAL_KEYS:
        params[key] = vector[:, idx:idx + n]
        idx += n
    if idx < vector.shape[1]:
        params["fx_routing"] = vector[:, idx:idx + N_ROUTING].reshape(-1, 6, 6)
    return params


def generate_dataset(
    n_samples: int,
    audio_duration: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
    gen_batch_size: int = 8,
    save_path: str | None = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    import numpy as np
    from pathlib import Path

    n_audio = int(sample_rate * audio_duration)
    synth = SubtractiveSynth(sample_rate, n_audio).to(device)

    mel_transform = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=256,
        n_mels=128,
        power=2.0,
    ).to(device)
    amp_to_db = T.AmplitudeToDB(top_db=80)

    # Probe shapes from a single batch
    probe_params = random_params(1, device=device)
    with torch.no_grad():
        probe_audio = synth(probe_params)
        probe_mel = amp_to_db(mel_transform(probe_audio))
    n_mels, n_frames = probe_mel.shape[1], probe_mel.shape[2]
    n_param_vec = params_to_vector(probe_params).shape[1]

    # Pre-allocate memory-mapped files for zero RAM overhead
    if save_path:
        tmp_dir = Path(save_path).parent
    else:
        import tempfile
        tmp_dir = Path(tempfile.gettempdir())

    mel_mm = np.memmap(
        tmp_dir / "_gen_mels.dat", dtype="float32", mode="w+",
        shape=(n_samples, n_mels, n_frames),
    )
    param_mm = np.memmap(
        tmp_dir / "_gen_params.dat", dtype="float32", mode="w+",
        shape=(n_samples, n_param_vec),
    )
    audio_mm = np.memmap(
        tmp_dir / "_gen_audio.dat", dtype="float32", mode="w+",
        shape=(n_samples, n_audio),
    )

    n_batches = (n_samples + gen_batch_size - 1) // gen_batch_size
    offset = 0

    for i in range(n_batches):
        bs = min(gen_batch_size, n_samples - offset)
        params = random_params(bs, device=device)

        with torch.no_grad():
            audio = synth(params)
            mel = mel_transform(audio)
            mel_db = amp_to_db(mel)
            mel_norm = ((mel_db + 80.0) / 80.0).clamp(0.0, 1.0)

        mel_mm[offset:offset + bs] = mel_norm.cpu().numpy()
        param_mm[offset:offset + bs] = params_to_vector(params).cpu().numpy()
        audio_mm[offset:offset + bs] = audio.cpu().numpy()
        offset += bs

        if (i + 1) % 100 == 0:
            print(f"  generated {offset}/{n_samples}")

    # Convert to tensors
    mels = torch.from_numpy(mel_mm[:n_samples].copy())
    param_vecs = torch.from_numpy(param_mm[:n_samples].copy())
    audio_all = torch.from_numpy(audio_mm[:n_samples].copy())

    # Clean up temp files
    del mel_mm, param_mm, audio_mm
    for f in ["_gen_mels.dat", "_gen_params.dat", "_gen_audio.dat"]:
        (tmp_dir / f).unlink(missing_ok=True)

    if save_path:
        torch.save({"mels": mels, "params": param_vecs, "audio": audio_all}, save_path)
        size_mb = (mels.nelement() + audio_all.nelement()) * mels.element_size() / 1e6
        print(f"  saved to {save_path} ({size_mb:.0f} MB)")

    return mels, param_vecs, audio_all


class SynthDataset(Dataset):
    def __init__(self, mels: torch.Tensor, params: torch.Tensor, device: str = "cpu"):
        self.mels = mels.to(device)
        self.params = params.to(device)
        self.on_gpu = device != "cpu"

    def __len__(self):
        return len(self.mels)

    def __getitem__(self, idx):
        return self.mels[idx], self.params[idx]
