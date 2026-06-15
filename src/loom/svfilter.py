import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class SVFilter(nn.Module):
    """Chunk-based State Variable Filter with per-sample cutoff modulation.

    Processes audio in chunks. Each chunk uses the cutoff value at its midpoint.
    Supports LP, HP, BP via continuous blend weights.
    """

    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    MIN_Q = 0.5
    MAX_Q = 20.0

    def __init__(self, sample_rate: int, chunk_size: int = 64):
        super().__init__()
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size

    def _denorm_cutoff(self, cutoff: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_HZ)
        log_max = math.log(self.MAX_HZ)
        return torch.exp(cutoff * (log_max - log_min) + log_min)

    def _denorm_q(self, q: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.MIN_Q)
        log_max = math.log(self.MAX_Q)
        return torch.exp(q * (log_max - log_min) + log_min)

    def forward(self, signal, cutoff, q, filter_type):
        """Apply SVF with time-varying cutoff.

        Args:
            signal: (batch, n_samples) input audio.
            cutoff: (batch, n_samples) normalized cutoff [0,1] per sample.
            q: (batch,) normalized Q [0,1].
            filter_type: (batch, 3) blend weights [LP, HP, BP].
        """
        batch, n_samples = signal.shape
        device = signal.device
        q_val = self._denorm_q(q)

        n_chunks = (n_samples + self.chunk_size - 1) // self.chunk_size
        output_chunks = []

        for c in range(n_chunks):
            start = c * self.chunk_size
            end = min(start + self.chunk_size, n_samples)
            chunk = signal[:, start:end]

            mid = min((start + end) // 2, n_samples - 1)
            chunk_cutoff_hz = self._denorm_cutoff(cutoff[:, mid])
            g = torch.tan(math.pi * chunk_cutoff_hz / self.sample_rate).clamp(max=10.0)
            R = 1.0 / (2.0 * q_val)

            g2 = g * g
            a0 = 1.0 + 2.0 * R * g + g2

            lp_b = torch.stack([g2/a0, 2.0*g2/a0, g2/a0], dim=-1)
            hp_b = torch.stack([1.0/a0, -2.0/a0, 1.0/a0], dim=-1)
            bp_b = torch.stack([2.0*R*g/a0, torch.zeros_like(a0), -2.0*R*g/a0], dim=-1)
            common_a = torch.stack([
                torch.ones_like(a0),
                (2.0 * (g2 - 1.0)) / a0,
                (1.0 - 2.0 * R * g + g2) / a0,
            ], dim=-1)

            chunk_out = torch.zeros_like(chunk)
            for j, b in enumerate([lp_b, hp_b, bp_b]):
                w = filter_type[:, j]
                if (w.abs() < 1e-6).all():
                    continue
                filtered = torch.zeros_like(chunk)
                for i in range(batch):
                    f = AF.lfilter(chunk[i:i+1], common_a[i], b[i], clamp=False)
                    filtered[i] = f.squeeze(0)
                chunk_out = chunk_out + w.unsqueeze(1) * filtered

            output_chunks.append(chunk_out)

        return torch.cat(output_chunks, dim=1)
