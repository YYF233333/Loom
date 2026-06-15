import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class BiquadFilter(nn.Module):
    """Differentiable biquad filter using torchaudio.lfilter.

    Supports LP, HP, BP filter types via continuous blending.
    Coefficients computed from Audio EQ Cookbook formulas.

    Args:
        sample_rate: Audio sample rate in Hz.
    """

    MIN_HZ = 20.0
    MAX_HZ = 20000.0
    MIN_Q = 0.5
    MAX_Q = 20.0

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def _denorm_cutoff(self, cutoff: torch.Tensor) -> torch.Tensor:
        """[0,1] -> Hz via log scale."""
        log_min = math.log(self.MIN_HZ)
        log_max = math.log(self.MAX_HZ)
        return torch.exp(cutoff * (log_max - log_min) + log_min)

    def _denorm_q(self, q: torch.Tensor) -> torch.Tensor:
        """[0,1] -> Q via log scale."""
        log_min = math.log(self.MIN_Q)
        log_max = math.log(self.MAX_Q)
        return torch.exp(q * (log_max - log_min) + log_min)

    def _compute_coeffs(
        self, cutoff_hz: torch.Tensor, q: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        """Compute biquad coefficients for LP, HP, BP (Audio EQ Cookbook).

        Returns dict with keys 'lp', 'hp', 'bp', each containing
        (b0, b1, b2, a0, a1, a2) as a (batch, 6) tensor.
        """
        w0 = 2.0 * math.pi * cutoff_hz / self.sample_rate
        alpha = torch.sin(w0) / (2.0 * q)
        cos_w0 = torch.cos(w0)

        lp_b0 = (1.0 - cos_w0) / 2.0
        lp_b1 = 1.0 - cos_w0
        lp_b2 = (1.0 - cos_w0) / 2.0

        hp_b0 = (1.0 + cos_w0) / 2.0
        hp_b1 = -(1.0 + cos_w0)
        hp_b2 = (1.0 + cos_w0) / 2.0

        bp_b0 = alpha
        bp_b1 = torch.zeros_like(alpha)
        bp_b2 = -alpha

        a0 = 1.0 + alpha
        a1 = -2.0 * cos_w0
        a2 = 1.0 - alpha

        return {
            "lp": torch.stack([lp_b0, lp_b1, lp_b2, a0, a1, a2], dim=-1),
            "hp": torch.stack([hp_b0, hp_b1, hp_b2, a0, a1, a2], dim=-1),
            "bp": torch.stack([bp_b0, bp_b1, bp_b2, a0, a1, a2], dim=-1),
        }

    def forward(
        self,
        signal: torch.Tensor,
        cutoff: torch.Tensor,
        q: torch.Tensor,
        filter_type: torch.Tensor,
    ) -> torch.Tensor:
        """Apply biquad filter.

        Args:
            signal: (batch, n_samples) input audio.
            cutoff: (batch,) normalized cutoff [0,1].
            q: (batch,) normalized Q [0,1].
            filter_type: (batch, 3) blend weights for [LP, HP, BP].

        Returns:
            (batch, n_samples) filtered audio.
        """
        cutoff_hz = self._denorm_cutoff(cutoff)
        q_val = self._denorm_q(q)
        all_coeffs = self._compute_coeffs(cutoff_hz, q_val)

        results = []
        for i in range(signal.shape[0]):
            sample_out = torch.zeros_like(signal[i])
            for j, key in enumerate(["lp", "hp", "bp"]):
                coeffs = all_coeffs[key][i]
                b = coeffs[:3] / coeffs[3]
                a = torch.cat(
                    [torch.ones(1, device=signal.device), coeffs[4:6] / coeffs[3]]
                )
                filtered = AF.lfilter(signal[i].unsqueeze(0), a, b, clamp=False)
                sample_out = sample_out + filter_type[i, j] * filtered.squeeze(0)
            results.append(sample_out)

        return torch.stack(results, dim=0)
