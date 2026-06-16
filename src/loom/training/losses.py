import torch

from loom.training.dataset import N_CONTINUOUS, CATEGORICAL_KEYS


def param_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    continuous_mse = (pred[:, :N_CONTINUOUS] - target[:, :N_CONTINUOUS]).pow(2).mean()

    cat_ce = torch.tensor(0.0, device=pred.device)
    idx = N_CONTINUOUS
    n_groups = 0
    for _, n in CATEGORICAL_KEYS:
        pred_logp = pred[:, idx:idx + n].clamp(1e-7, 1.0).log()
        target_p = target[:, idx:idx + n]
        cat_ce = cat_ce + (-target_p * pred_logp).sum(dim=-1).mean()
        idx += n
        n_groups += 1

    return continuous_mse + 0.5 * (cat_ce / n_groups)


_HANN_CACHE: dict[tuple, torch.Tensor] = {}


def multi_resolution_stft_loss(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    fft_sizes: list[int] = [512, 1024, 2048],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=pred_audio.device)
    combined = torch.cat([pred_audio, target_audio], dim=0)

    for fft_size in fft_sizes:
        key = (fft_size, pred_audio.device)
        if key not in _HANN_CACHE:
            _HANN_CACHE[key] = torch.hann_window(fft_size, device=pred_audio.device)
        window = _HANN_CACHE[key]
        hop = fft_size // 4

        combined_mag = torch.stft(
            combined, fft_size, hop_length=hop,
            window=window, return_complex=True,
        ).abs()
        pred_mag, target_mag = combined_mag.chunk(2, dim=0)

        loss = loss + (pred_mag - target_mag).abs().mean()
        loss = loss + (
            torch.log(pred_mag + 1e-7) - torch.log(target_mag + 1e-7)
        ).abs().mean()

    return loss / len(fft_sizes)
