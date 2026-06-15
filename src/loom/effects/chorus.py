import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Chorus(nn.Module):
    """Differentiable chorus effect via LFO-modulated delay line.

    Uses grid_sample for fractional delay interpolation.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
    """

    RATE_MIN_HZ = 0.1
    RATE_MAX_HZ = 5.0
    BASE_DELAY_MS = 7.0
    MAX_DEPTH_MS = 5.0

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _denorm_rate(self, rate: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.RATE_MIN_HZ)
        log_max = math.log(self.RATE_MAX_HZ)
        return torch.exp(rate * (log_max - log_min) + log_min)

    def forward(
        self,
        signal: torch.Tensor,
        rate: torch.Tensor,
        depth: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply chorus.

        Args:
            signal: (batch, n_samples) input audio.
            rate: (batch,) normalized LFO rate [0,1] -> [0.1, 5] Hz.
            depth: (batch,) normalized modulation depth [0,1].
            mix: (batch,) dry/wet [0,1].
        """
        # Short-circuit when fully bypassed to avoid polluting gradients.
        if mix.max().item() < 0.02:
            return signal + 0.0 * mix.unsqueeze(1)

        batch = signal.shape[0]
        rate_hz = self._denorm_rate(rate)
        mix = mix.unsqueeze(1)
        depth = depth.unsqueeze(1)

        lfo = torch.sin(
            2.0 * math.pi * rate_hz.unsqueeze(1) * self.t.unsqueeze(0)
        )

        base_delay_samples = self.BASE_DELAY_MS / 1000.0 * self.sample_rate
        mod_samples = depth * self.MAX_DEPTH_MS / 1000.0 * self.sample_rate
        delay_samples = base_delay_samples + lfo * mod_samples

        indices = torch.arange(
            self.n_samples, dtype=torch.float32, device=signal.device
        ).unsqueeze(0)
        read_pos = indices - delay_samples

        # grid_sample expects grid shape (N, H_out, W_out, 2) for 4D input.
        # We model audio as a 1xN "image" (H=1, W=N_samples).
        # x-coordinate (width) maps to sample position in [-1, 1].
        # y-coordinate (height) is always 0 (center of the single row).
        x_norm = (read_pos / (self.n_samples - 1)) * 2.0 - 1.0  # (batch, N)
        y_norm = torch.zeros_like(x_norm)                         # (batch, N)
        # grid shape: (batch, 1, N_samples, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)

        # signal_4d shape: (batch, 1, 1, N_samples)
        signal_4d = signal.unsqueeze(1).unsqueeze(2)
        wet = F.grid_sample(
            signal_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # wet shape: (batch, 1, 1, N_samples) -> (batch, N_samples)
        wet = wet.squeeze(1).squeeze(1)

        return (1.0 - mix) * signal + mix * wet
