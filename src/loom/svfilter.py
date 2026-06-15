import torch
import torch.nn as nn
import torch.nn.functional as Fn
import torchaudio.functional as AF
import math


class SVFilter(nn.Module):
    """Chunk-based State Variable Filter with per-sample cutoff modulation.

    Processes audio in chunks. Each chunk uses the cutoff value at its midpoint.
    All chunks are batched into a single lfilter call for performance.

    Args:
        sample_rate: Audio sample rate in Hz.
        chunk_size: Samples per chunk. Controls modulation time resolution.
    """

    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    MIN_Q = 0.5
    MAX_Q = 20.0

    def __init__(self, sample_rate: int, chunk_size: int = 256):
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
        cs = self.chunk_size

        # Pad signal to multiple of chunk_size
        pad_len = (cs - n_samples % cs) % cs
        if pad_len > 0:
            signal_padded = Fn.pad(signal, (0, pad_len))
            cutoff_padded = Fn.pad(cutoff, (0, pad_len), value=cutoff[:, -1:].mean().item())
        else:
            signal_padded = signal
            cutoff_padded = cutoff

        n_padded = signal_padded.shape[1]
        n_chunks = n_padded // cs

        # Reshape: (batch, n_chunks, cs)
        sig_chunks = signal_padded.reshape(batch, n_chunks, cs)

        # Per-chunk cutoff at midpoint: (batch, n_chunks)
        mid_indices = torch.arange(n_chunks, device=device) * cs + cs // 2
        mid_indices = mid_indices.clamp(max=n_padded - 1)
        chunk_cutoff = cutoff_padded[:, mid_indices]  # (batch, n_chunks)
        chunk_cutoff_hz = self._denorm_cutoff(chunk_cutoff)

        # SVF coefficients for all chunks at once
        g = torch.tan(math.pi * chunk_cutoff_hz / self.sample_rate).clamp(max=10.0)
        R = (1.0 / (2.0 * q_val)).unsqueeze(1).expand_as(g)  # (batch, n_chunks)
        g2 = g * g
        a0 = 1.0 + 2.0 * R * g + g2

        # Coefficients: (batch, n_chunks, 3)
        lp_b = torch.stack([g2 / a0, 2.0 * g2 / a0, g2 / a0], dim=-1)
        hp_b = torch.stack([1.0 / a0, -2.0 / a0, 1.0 / a0], dim=-1)
        bp_b = torch.stack([2.0 * R * g / a0, torch.zeros_like(a0), -2.0 * R * g / a0], dim=-1)
        common_a = torch.stack([
            torch.ones_like(a0),
            (2.0 * (g2 - 1.0)) / a0,
            (1.0 - 2.0 * R * g + g2) / a0,
        ], dim=-1)

        # Flatten batch*chunks into mega-batch for single lfilter call
        # (batch * n_chunks, cs)
        sig_flat = sig_chunks.reshape(batch * n_chunks, cs)
        a_flat = common_a.reshape(batch * n_chunks, 3)

        # Process each filter type with one batched lfilter call
        result = torch.zeros_like(sig_flat)
        for j, b_coeffs in enumerate([lp_b, hp_b, bp_b]):
            w = filter_type[:, j]  # (batch,)
            if (w.abs() < 1e-6).all():
                continue
            b_flat = b_coeffs.reshape(batch * n_chunks, 3)
            filtered = AF.lfilter(sig_flat, a_flat, b_flat, clamp=False)
            # Weight: expand (batch,) -> (batch, n_chunks, 1) -> (batch*n_chunks, 1)
            w_expanded = w.unsqueeze(1).unsqueeze(2).expand(-1, n_chunks, 1).reshape(batch * n_chunks, 1)
            result = result + w_expanded * filtered

        # Reshape back and trim padding
        output = result.reshape(batch, n_padded)[:, :n_samples]
        return output
