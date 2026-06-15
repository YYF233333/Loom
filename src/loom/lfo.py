import torch
import torch.nn as nn
import math


class LFO(nn.Module):
    """Low-frequency oscillator for parameter modulation.

    Generates a time-varying signal in [-1, 1] range, scaled by depth.
    Supports sine/saw/square/tri waveforms via continuous blending.
    """

    RATE_MIN_HZ = 0.1
    RATE_MAX_HZ = 20.0

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

    def forward(self, rate, depth, waveform, phase):
        """Generate LFO modulation signal.

        Args:
            rate: (batch,) normalized [0,1] -> [0.1, 20] Hz.
            depth: (batch,) modulation depth [0,1].
            waveform: (batch, 4) blend weights [sine, saw, square, tri].
            phase: (batch,) initial phase offset [0,1] -> [0, 2pi].
        Returns:
            (batch, n_samples) modulation signal in [-depth, +depth].
        """
        rate_hz = self._denorm_rate(rate)
        t = self.t.unsqueeze(0)
        phase_rad = phase.unsqueeze(1) * 2.0 * math.pi

        theta = 2.0 * math.pi * rate_hz.unsqueeze(1) * t + phase_rad
        theta_norm = theta % (2.0 * math.pi)
        frac = theta_norm / (2.0 * math.pi)

        sine = torch.sin(theta)
        saw = 2.0 * frac - 1.0
        square = torch.sign(torch.sin(theta))
        tri = 4.0 * torch.abs(frac - 0.5) - 1.0

        waves = torch.stack([sine, saw, square, tri], dim=1)
        w = waveform.unsqueeze(2)
        blended = (w * waves).sum(dim=1)

        return blended * depth.unsqueeze(1)
