import torch

from loom.training.dataset import N_CONTINUOUS, CATEGORICAL_KEYS


def param_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    continuous_mse = (pred[:, :N_CONTINUOUS] - target[:, :N_CONTINUOUS]).pow(2).mean()

    cat_parts = []
    idx = N_CONTINUOUS
    for _, n in CATEGORICAL_KEYS:
        pred_logp = pred[:, idx:idx + n].clamp(1e-7, 1.0).log()
        target_p = target[:, idx:idx + n]
        cat_parts.append((-target_p * pred_logp).sum(dim=-1).mean())
        idx += n

    cat_ce = torch.stack(cat_parts).mean()
    return continuous_mse + 0.5 * cat_ce


_HANN_CACHE: dict[tuple, torch.Tensor] = {}


def signal_chain_loss(
    pred_intermediates: dict[str, torch.Tensor],
    target_intermediates: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """DiffMoog-style signal-chain loss: supervise each stage independently."""
    if weights is None:
        weights = {"osc": 1.0, "filter": 0.5, "dry": 0.3}
    parts = []
    for key, w in weights.items():
        if key in pred_intermediates and key in target_intermediates:
            pred = pred_intermediates[key]
            target = target_intermediates[key]
            parts.append(w * multi_resolution_stft_loss(pred, target))
    if not parts:
        return torch.tensor(0.0, device=next(iter(pred_intermediates.values())).device)
    return torch.stack(parts).sum()


def multi_resolution_stft_loss(
    pred_audio: torch.Tensor,
    target_audio: torch.Tensor,
    fft_sizes: list[int] = [512, 1024, 2048],
) -> torch.Tensor:
    combined = torch.cat([pred_audio, target_audio], dim=0)
    parts = []

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

        parts.append((pred_mag - target_mag).abs().mean())
        parts.append(
            (torch.log(pred_mag + 1e-7) - torch.log(target_mag + 1e-7)).abs().mean()
        )

    return torch.stack(parts).mean()
