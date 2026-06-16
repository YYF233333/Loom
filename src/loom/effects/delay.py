import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Delay(nn.Module):
    """Differentiable delay line with feedback.

    Uses grid_sample for fractional delay and unrolls feedback 8 times.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
        n_taps: Number of feedback iterations to unroll.
    """

    MIN_MS = 10.0
    MAX_MS = 500.0
    MAX_FEEDBACK = 0.9

    def __init__(self, sample_rate: int, n_samples: int, n_taps: int = 8):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_taps = n_taps

    def _denorm_time(self, time: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_MS)
        log_max = math.log(self.MAX_MS)
        ms = torch.exp(time * (log_max - log_min) + log_min)
        return ms / 1000.0 * self.sample_rate

    def _fractional_delay(
        self, signal: torch.Tensor, delay_samples: torch.Tensor
    ) -> torch.Tensor:
        """Shift signal by delay_samples using grid_sample.

        Follows the same pattern as chorus.py:
        - signal_4d shape: (batch, 1, 1, N_samples)
        - grid shape: (batch, 1, N_samples, 2) with (x, y) coordinates
        """
        n = signal.shape[1]
        indices = torch.arange(n, dtype=torch.float32, device=signal.device).unsqueeze(0)
        read_pos = indices - delay_samples.unsqueeze(1)

        # Normalize x to [-1, 1] over the sample dimension (width)
        x_norm = (read_pos / (n - 1)) * 2.0 - 1.0  # (batch, N)
        # y is always 0 (center of the single row height=1)
        y_norm = torch.zeros_like(x_norm)            # (batch, N)
        # grid shape: (batch, 1, N_samples, 2)
        grid = torch.stack([x_norm, y_norm], dim=-1).unsqueeze(1)

        # signal_4d shape: (batch, 1, 1, N_samples)
        signal_4d = signal.unsqueeze(1).unsqueeze(2)
        out = F.grid_sample(
            signal_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # out shape: (batch, 1, 1, N_samples) -> (batch, N_samples)
        return out.squeeze(1).squeeze(1)

    def forward(
        self,
        signal: torch.Tensor,
        time: torch.Tensor,
        feedback: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply delay effect.

        Args:
            signal: (batch, n_samples) input audio.
            time: (batch,) normalized delay time [0,1] -> [10ms, 500ms].
            feedback: (batch,) feedback amount [0,1] -> [0, 0.9].
            mix: (batch,) dry/wet [0,1].
        """
        if mix.max().item() < 0.02:
            return signal + 0.0 * mix.unsqueeze(1)

        delay_samples = self._denorm_time(time)
        fb = feedback * self.MAX_FEEDBACK
        mix = mix.unsqueeze(1)
        n = signal.shape[1]

        indices = torch.arange(n, dtype=torch.float32, device=signal.device).unsqueeze(0)
        tap_mul = torch.arange(1, self.n_taps + 1, dtype=torch.float32, device=signal.device)
        read_pos = indices.unsqueeze(1) - delay_samples.unsqueeze(1).unsqueeze(1) * tap_mul.unsqueeze(0).unsqueeze(2)
        x_norm = (read_pos / (n - 1)) * 2.0 - 1.0
        y_norm = torch.zeros_like(x_norm)
        grid = torch.stack([x_norm, y_norm], dim=-1)

        signal_4d = signal.unsqueeze(1).unsqueeze(2)
        delayed = F.grid_sample(signal_4d, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
        delayed = delayed.squeeze(1)

        fb_weights = fb.unsqueeze(1).pow(tap_mul - 1).unsqueeze(-1)
        wet = (fb_weights * delayed).sum(dim=1)

        return (1.0 - mix) * signal + mix * wet
