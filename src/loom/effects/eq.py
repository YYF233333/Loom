import torch
import torch.nn as nn
import math


class EQ(nn.Module):
    """3-band parametric EQ: low shelf, mid peak, high shelf.

    Uses frequency-domain multiplication for fully differentiable operation
    with stable gradient flow regardless of signal length.

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
        return torch.pow(10.0, db / 20.0)

    def _biquad_freq_response(
        self, a: torch.Tensor, b: torch.Tensor, n_fft: int
    ) -> torch.Tensor:
        """Compute frequency response H(w) of a biquad filter.

        Args:
            a: (batch, 3) denominator coefficients [1, a1, a2].
            b: (batch, 3) numerator coefficients [b0, b1, b2].
            n_fft: FFT size (number of frequency bins = n_fft // 2 + 1).

        Returns:
            (batch, n_fft // 2 + 1) complex frequency response.
        """
        n_bins = n_fft // 2 + 1
        w = torch.linspace(0, math.pi, n_bins, device=a.device)  # (n_bins,)

        # e^{-jw}, e^{-j2w}
        ej1 = torch.exp(-1j * w)  # (n_bins,)
        ej2 = torch.exp(-2j * w)  # (n_bins,)

        # B(w) = b0 + b1*e^{-jw} + b2*e^{-j2w}
        b0 = b[:, 0:1]  # (batch, 1)
        b1 = b[:, 1:2]
        b2 = b[:, 2:3]
        B = b0 + b1 * ej1.unsqueeze(0) + b2 * ej2.unsqueeze(0)

        # A(w) = 1 + a1*e^{-jw} + a2*e^{-j2w}
        a1 = a[:, 1:2]
        a2 = a[:, 2:3]
        A = 1.0 + a1 * ej1.unsqueeze(0) + a2 * ej2.unsqueeze(0)

        return B / (A + 1e-8)

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

    def forward(
        self,
        signal: torch.Tensor,
        low_gain: torch.Tensor,
        mid_gain: torch.Tensor,
        high_gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply 3-band EQ via frequency-domain filtering.

        Args:
            signal: (batch, n_samples) input audio.
            low_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 200Hz.
            mid_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 1kHz.
            high_gain: (batch,) normalized [0,1] -> [-12dB, +12dB] at 5kHz.
        """
        # Short-circuit when EQ is near flat (all gains ~ 0.5 = 0dB).
        deviation = (
            (low_gain - 0.5).abs().max()
            + (mid_gain - 0.5).abs().max()
            + (high_gain - 0.5).abs().max()
        )
        if deviation.item() < 0.05:
            return signal + 0.0 * (low_gain.sum() + mid_gain.sum() + high_gain.sum())

        A_low = self._denorm_gain(low_gain).clamp(min=0.01)
        A_mid = self._denorm_gain(mid_gain).clamp(min=0.01)
        A_high = self._denorm_gain(high_gain).clamp(min=0.01)

        a_l, b_l = self._low_shelf_coeffs(A_low)
        a_m, b_m = self._peak_coeffs(A_mid)
        a_h, b_h = self._high_shelf_coeffs(A_high)

        n_samples = signal.shape[1]
        n_fft = 2 ** (n_samples - 1).bit_length()  # next power of 2

        # Compute combined frequency response
        H_l = self._biquad_freq_response(a_l, b_l, n_fft)
        H_m = self._biquad_freq_response(a_m, b_m, n_fft)
        H_h = self._biquad_freq_response(a_h, b_h, n_fft)
        H = H_l * H_m * H_h  # (batch, n_fft // 2 + 1) complex

        # FFT-based filtering
        X = torch.fft.rfft(signal, n=n_fft)  # (batch, n_fft // 2 + 1)
        Y = X * H
        out = torch.fft.irfft(Y, n=n_fft)[:, :n_samples]

        return out
