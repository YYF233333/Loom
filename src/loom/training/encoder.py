import math

import torch
import torch.nn as nn

from loom.training.dataset import N_CONTINUOUS, N_PARAMS, N_ROUTING, CATEGORICAL_KEYS

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ── S4D fallback (pure PyTorch, any GPU) ─────────────────────────────


class S4DKernel(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.log_A_real = nn.Parameter(
            torch.log(0.5 * torch.ones(d_model, d_state))
        )
        A_imag = math.pi * torch.arange(d_state).float()
        self.A_imag = nn.Parameter(A_imag.unsqueeze(0).expand(d_model, -1).clone())
        self.C = nn.Parameter(
            torch.randn(d_model, d_state, 2) * (0.5 / d_state) ** 0.5
        )
        self.log_dt = nn.Parameter(torch.randn(d_model) * 0.1 - 4.0)

    def forward(self, L):
        dt = torch.exp(self.log_dt).unsqueeze(-1)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag
        C = self.C[..., 0] + 1j * self.C[..., 1]
        dtA = A * dt
        k = torch.arange(L, device=dtA.device, dtype=torch.float32)
        vandermonde = torch.exp(dtA.unsqueeze(-1) * k)
        return torch.einsum("dn,dnl->dl", C, vandermonde).real


class S4DLayer(nn.Module):
    def __init__(self, d_model, d_state=64):
        super().__init__()
        self.kernel = S4DKernel(d_model, d_state)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, u):
        L = u.shape[-1]
        K = self.kernel(L)
        n_fft = 2 * L
        y = torch.fft.irfft(
            torch.fft.rfft(u, n=n_fft) * torch.fft.rfft(K, n=n_fft).unsqueeze(0),
            n=n_fft,
        )[..., :L]
        return y + self.D.unsqueeze(0).unsqueeze(-1) * u


class S4DBlock(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.s4d = S4DLayer(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        h = self.s4d(self.norm1(x).transpose(1, 2)).transpose(1, 2)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ── Mamba block (requires mamba-ssm, cc >= 7.0) ─────────────────────


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ── Encoder ──────────────────────────────────────────────────────────


def _make_blocks(d_model, d_state, n_layers, dropout):
    Block = MambaBlock if HAS_MAMBA else S4DBlock
    return nn.Sequential(*[Block(d_model, d_state, dropout=dropout) for _ in range(n_layers)])


class ParamEncoder(nn.Module):
    """SSM encoder: mel spectrogram -> synth parameter vector.

    Uses Mamba (selective scan) when mamba-ssm is installed, falls back
    to S4D (diagonal SSM via FFT convolution) on older GPUs.
    """

    def __init__(self, n_mels=128, d_model=160, d_state=64, n_layers=6, dropout=0.1):
        super().__init__()
        self.n_continuous = N_CONTINUOUS
        self.n_routing = N_ROUTING
        self.categorical_groups = CATEGORICAL_KEYS

        self.stem = nn.Sequential(
            nn.Conv1d(n_mels, d_model, 3, padding=1),
            nn.GELU(),
        )
        self.blocks = _make_blocks(d_model, d_state, n_layers, dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, N_PARAMS),
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        x = self.stem(mel)
        x = x.transpose(1, 2)
        x = self.blocks(x)
        x = self.norm(x).mean(dim=1)
        x = self.head(x)

        continuous = torch.sigmoid(x[:, : self.n_continuous])
        cats = []
        idx = self.n_continuous
        for _, n in self.categorical_groups:
            cats.append(torch.softmax(x[:, idx : idx + n], dim=-1))
            idx += n
        routing = x[:, idx : idx + self.n_routing]

        return torch.cat([continuous] + cats + [routing], dim=1)
