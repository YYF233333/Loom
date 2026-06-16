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
) -> tuple[torch.Tensor, torch.Tensor]:
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

    all_mels = []
    all_params = []
    all_audio = []
    n_batches = (n_samples + gen_batch_size - 1) // gen_batch_size

    for i in range(n_batches):
        bs = min(gen_batch_size, n_samples - i * gen_batch_size)
        params = random_params(bs, device=device)

        with torch.no_grad():
            audio = synth(params)
            mel = mel_transform(audio)
            mel_db = amp_to_db(mel)
            mel_norm = (mel_db + 80.0) / 80.0
            mel_norm = mel_norm.clamp(0.0, 1.0)

        all_mels.append(mel_norm.cpu())
        all_params.append(params_to_vector(params).cpu())
        all_audio.append(audio.cpu())

        if (i + 1) % 100 == 0:
            print(f"  generated {(i + 1) * gen_batch_size}/{n_samples}")

    mels = torch.cat(all_mels, dim=0)[:n_samples]
    param_vecs = torch.cat(all_params, dim=0)[:n_samples]
    audio_all = torch.cat(all_audio, dim=0)[:n_samples]

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
