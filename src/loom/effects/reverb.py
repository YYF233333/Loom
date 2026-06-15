import torch
import torch.nn as nn
import math


class Reverb(nn.Module):
    """Differentiable FDN reverb computed in the frequency domain.

    Evaluates the FDN transfer function H(z) at each FFT bin and
    multiplies with the input spectrum. No time-domain loops.

    4 delay lines with mutually coprime lengths, Householder feedback matrix.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of samples in the buffer.
    """

    BASE_DELAYS = [1433, 1601, 1867, 2053]
    ROOM_SCALE_MIN = 0.5
    ROOM_SCALE_MAX = 2.0
    MAX_DECAY = 0.95

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_delays = len(self.BASE_DELAYS)

        # Householder feedback matrix: H = I - 2vvT/(vTv) with v=[1,1,1,1]
        v = torch.ones(self.n_delays)
        H = torch.eye(self.n_delays) - 2.0 * torch.outer(v, v) / torch.dot(v, v)
        self.register_buffer("feedback_matrix", H)

        base = torch.tensor(self.BASE_DELAYS, dtype=torch.float32)
        self.register_buffer("base_delays", base)

    def forward(
        self,
        signal: torch.Tensor,
        room_size: torch.Tensor,
        decay: torch.Tensor,
        damping: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply FDN reverb.

        Args:
            signal: (batch, n_samples) input audio.
            room_size: (batch,) normalized [0,1] -> scales delay lengths.
            decay: (batch,) normalized [0,1] -> feedback gain.
            damping: (batch,) normalized [0,1] -> high-freq absorption.
            mix: (batch,) dry/wet [0,1].
        """
        # Short-circuit when fully bypassed to avoid polluting gradients.
        if mix.max().item() < 0.02:
            return signal + 0.0 * mix.unsqueeze(1)

        batch = signal.shape[0]
        device = signal.device
        mix_expand = mix.unsqueeze(1)

        # Scale delay lengths by room size (keep as float for differentiability)
        scale = room_size * (self.ROOM_SCALE_MAX - self.ROOM_SCALE_MIN) + self.ROOM_SCALE_MIN
        delays = self.base_delays.unsqueeze(0) * scale.unsqueeze(1)  # (batch, 4)

        # Decay gain clamped to max 0.95
        g = decay * self.MAX_DECAY  # (batch,)
        # Damping in [0.05, 0.95]
        damp = damping * 0.9 + 0.05  # (batch,)

        n_fft = self.n_samples
        X = torch.fft.rfft(signal, n=n_fft)  # (batch, n_freq)
        n_freq = X.shape[1]

        # Angular frequencies for each FFT bin
        omega = 2.0 * math.pi * torch.arange(n_freq, device=device, dtype=torch.float32) / n_fft

        # z^{-m_i} = exp(-j * omega * m_i) for each delay line at each freq bin
        # delays: (batch, 4), omega: (n_freq,)
        z_neg_m = torch.exp(
            -1j * omega.unsqueeze(0).unsqueeze(0) * delays.unsqueeze(2)
        )  # (batch, 4, n_freq)

        # Constant gain per round-trip: larger rooms have proportionally
        # longer RT60 (RT60 = -3 * delay_time / log10(gamma)).
        # Compress g into high range so moderate decay produces audible tails.
        gamma = g.clamp(min=1e-6).pow(0.15).unsqueeze(1).expand_as(delays)  # (batch, 4)

        # One-pole lowpass damping filter: H_lp(z) = (1-d) / (1 - d*z^{-1})
        z_inv = torch.exp(-1j * omega)  # (n_freq,)
        damp_expand = damp.unsqueeze(1)  # (batch, 1)
        lp = (1.0 - damp_expand) / (1.0 - damp_expand * z_inv.unsqueeze(0))  # (batch, n_freq)

        # Combined per-delay filter: gamma_i * lp(z)
        filt = gamma.unsqueeze(2) * lp.unsqueeze(1)  # (batch, 4, n_freq)

        # Build per-frequency 4x4 system matrix: M = D - A @ diag(filt)
        # where D[i,i] = z^{-m_i} (diagonal) and A is the feedback matrix
        A = self.feedback_matrix.to(torch.cfloat)  # (4, 4)

        eye = torch.eye(self.n_delays, device=device, dtype=torch.cfloat)
        # D: (batch, 4, 4, n_freq) - diagonal matrices with z_neg_m on diag
        D = eye.unsqueeze(0).unsqueeze(3) * z_neg_m.unsqueeze(2)
        # A @ diag(filt): (batch, 4, 4, n_freq)
        AG = A.unsqueeze(0).unsqueeze(3) * filt.unsqueeze(1)

        system = D - AG  # (batch, 4, 4, n_freq)

        # Add small regularization for numerical stability
        reg = 1e-6 * eye.unsqueeze(0).unsqueeze(3)
        system = system + reg

        # Input/output vectors
        B = torch.ones(self.n_delays, 1, device=device, dtype=torch.cfloat)
        C = torch.ones(1, self.n_delays, device=device, dtype=torch.cfloat) / self.n_delays

        # Solve: system @ x = B for each (batch, freq)
        # Permute to (batch, n_freq, 4, 4) for batched solve
        system_perm = system.permute(0, 3, 1, 2).contiguous()
        B_expand = B.unsqueeze(0).unsqueeze(0).expand(batch, n_freq, -1, -1)

        x = torch.linalg.solve(system_perm, B_expand)  # (batch, n_freq, 4, 1)
        H = (C.unsqueeze(0).unsqueeze(0) @ x).squeeze(-1).squeeze(-1)  # (batch, n_freq)

        # Apply transfer function and IFFT back
        wet = torch.fft.irfft(X * H, n=n_fft)
        wet = wet[:, :self.n_samples]

        return (1.0 - mix_expand) * signal + mix_expand * wet
