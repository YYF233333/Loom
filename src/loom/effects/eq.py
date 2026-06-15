import torch
import torch.nn as nn
import torchaudio.functional as AF
import math


class EQ(nn.Module):
    """3-band parametric EQ: low shelf, mid peak, high shelf.

    Uses biquad filters via torchaudio.lfilter. Gain range +/-12dB.

    Args:
        sample_rate: Audio sample rate in Hz.
    """

    LOW_FREQ = 200.0
    MID_FREQ = 1000.0
    HIGH_FREQ = 5000.0
    MID_Q = 1.0
    GAIN_RANGE_DB = 12.0

    def __init__(self, sample_rate: int):
        super().__init__()
        self.sample_rate = sample_rate

    def _denorm_gain(self, gain: torch.Tensor) -> torch.Tensor:
        """[0,1] -> [-12dB, +12dB] -> linear amplitude A for shelf/peak."""
        db = (gain - 0.5) * 2.0 * self.GAIN_RANGE_DB
        return torch.pow(10.0, db / 40.0)

    def _low_shelf_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.LOW_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * math.sqrt(2.0))

        Ap1 = A + 1.0
        Am1 = A - 1.0
        sqrt_A_alpha = 2.0 * torch.sqrt(A) * alpha

        b0 = A * (Ap1 - Am1 * cos_w0 + sqrt_A_alpha)
        b1 = 2.0 * A * (Am1 - Ap1 * cos_w0)
        b2 = A * (Ap1 - Am1 * cos_w0 - sqrt_A_alpha)
        a0 = Ap1 + Am1 * cos_w0 + sqrt_A_alpha
        a1 = -2.0 * (Am1 + Ap1 * cos_w0)
        a2 = Ap1 + Am1 * cos_w0 - sqrt_A_alpha

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _peak_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.MID_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * self.MID_Q)

        b0 = 1.0 + alpha * A
        b1 = -2.0 * cos_w0 * torch.ones_like(A)
        b2 = 1.0 - alpha * A
        a0 = 1.0 + alpha / A
        a1 = -2.0 * cos_w0 * torch.ones_like(A)
        a2 = 1.0 - alpha / A

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _high_shelf_coeffs(self, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        w0 = 2.0 * math.pi * self.HIGH_FREQ / self.sample_rate
        cos_w0 = math.cos(w0)
        sin_w0 = math.sin(w0)
        alpha = sin_w0 / (2.0 * math.sqrt(2.0))

        Ap1 = A + 1.0
        Am1 = A - 1.0
        sqrt_A_alpha = 2.0 * torch.sqrt(A) * alpha

        b0 = A * (Ap1 + Am1 * cos_w0 + sqrt_A_alpha)
        b1 = -2.0 * A * (Am1 + Ap1 * cos_w0)
        b2 = A * (Ap1 + Am1 * cos_w0 - sqrt_A_alpha)
        a0 = Ap1 - Am1 * cos_w0 + sqrt_A_alpha
        a1 = 2.0 * (Am1 - Ap1 * cos_w0)
        a2 = Ap1 - Am1 * cos_w0 - sqrt_A_alpha

        b = torch.stack([b0 / a0, b1 / a0, b2 / a0], dim=-1)
        a = torch.stack([torch.ones_like(a0), a1 / a0, a2 / a0], dim=-1)
        return a, b

    def _apply_biquad(
        self, signal: torch.Tensor, a: torch.Tensor, b: torch.Tensor
    ) -> torch.Tensor:
        results = []
        for i in range(signal.shape[0]):
            filtered = AF.lfilter(
                signal[i].unsqueeze(0), a[i], b[i], clamp=False
            )
            results.append(filtered.squeeze(0))
        return torch.stack(results, dim=0)

    def forward(
        self,
        signal: torch.Tensor,
        low_gain: torch.Tensor,
        mid_gain: torch.Tensor,
        high_gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 3-band EQ.

        Args:
            signal: (batch, n_samples) input audio.
            low_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 200Hz.
            mid_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 1kHz.
            high_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 5kHz.
        """
        A_low = self._denorm_gain(low_gain).clamp(min=0.01)
        A_mid = self._denorm_gain(mid_gain).clamp(min=0.01)
        A_high = self._denorm_gain(high_gain).clamp(min=0.01)

        a_l, b_l = self._low_shelf_coeffs(A_low)
        a_m, b_m = self._peak_coeffs(A_mid)
        a_h, b_h = self._high_shelf_coeffs(A_high)

        out = self._apply_biquad(signal, a_l, b_l)
        out = self._apply_biquad(out, a_m, b_m)
        out = self._apply_biquad(out, a_h, b_h)
        return out
