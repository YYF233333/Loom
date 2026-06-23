"""Audio frontends — time-frequency representations for the conditioner.

Supported:
    cqt:   Constant-Q Transform (log-frequency, 3 bins/octave, preserves bass pitch)
    mel:   Standard mel spectrogram
    gammatone: ERB-spaced Gammatone filterbank
    multi: Multi-resolution stack

CQT is recommended: log-spaced bins match musical pitch, 192 bins × 3/oct
gives ~18 bins below 200 Hz (vs ~8 for 128-bin mel/gammatone).
"""

import torch
import torch.nn as nn
import torchaudio.transforms as T
from nnAudio.features import CQT1992v2, Gammatonegram


# ── CQT (recommended) ────────────────────────────────────────────────────────


class CQTFrontend(nn.Module):
    """Constant-Q Transform via nnAudio — log-frequency, 3 bins/octave.

    CQT places bins on a musical scale: each octave gets the same number of bins.
    At 3 bins/octave × 8 octaves (C1~32.7 Hz → C9~8372 Hz) = 192 bins.
    Bass range (32-200 Hz, ~2.5 octaves) gets ~8 bins, vs ~1-2 for mel.

    nnAudio implements CQT via efficient STFT-based convolution.
    """

    def __init__(
        self,
        sample_rate: int = 44100,
        hop_length: int = 256,
        n_bins: int = 192,             # 8 octaves × 24 bins/oct (C1 → C9)
        bins_per_octave: int = 24,     # 3 bins per semitone
        fmin: float = 32.7,            # C1 (needs ≥1s audio for low-freq window)
        top_db: float = 80.0,
    ):
        super().__init__()
        self.cqt = CQT1992v2(
            sr=sample_rate,
            hop_length=hop_length,
            n_bins=n_bins,
            bins_per_octave=bins_per_octave,
            fmin=fmin,
            output_format="Magnitude",
            norm=True,
            window="hann",
        )
        self.top_db = top_db

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """audio (B, T_raw) → cqt_norm (B, n_bins, T_frames) in [0, 1]."""
        x = self.cqt(audio)                              # (B, n_bins, T_frames)
        x = 20.0 * torch.log10(x.clamp(1e-7))            # dB
        x = ((x + self.top_db) / self.top_db).clamp(0.0, 1.0)
        return x


# ── Gammatone ─────────────────────────────────────────────────────────────────


class GammatoneFrontend(nn.Module):
    """Gammatonegram via nnAudio — ERB-spaced 4th-order cochlear filterbank."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_bins: int = 128,
        fmin: float = 20.0,
        fmax: float = 16000.0,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.gt = Gammatonegram(
            sr=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_bins=n_bins, fmin=fmin, fmax=fmax, power=1.0, htk=False,
        )
        self.top_db = top_db

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.gt(audio)
        x = 20.0 * torch.log10(x.clamp(1e-7))
        x = ((x + self.top_db) / self.top_db).clamp(0.0, 1.0)
        return x


# ── Mel ───────────────────────────────────────────────────────────────────────


class MelFrontend(nn.Module):
    """Standard mel spectrogram → dB normalization → [0,1]."""

    def __init__(
        self,
        sample_rate: int = 44100,
        n_fft: int = 1024,
        hop_length: int = 256,
        n_mels: int = 128,
        top_db: float = 80.0,
    ):
        super().__init__()
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length,
            n_mels=n_mels, power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(top_db=top_db)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.mel(audio)
        x = self.amp_to_db(x)
        x = ((x + 80.0) / 80.0).clamp(0.0, 1.0)
        return x


# ── Multi-Resolution CQT Stack ────────────────────────────────────────────────


class MultiResCQT(nn.Module):
    """3 CQT frontends at different time resolutions, stacked along frequency axis.

    Fast (hop=64):   ~346 time frames — onset transients, attack detail
    Medium (hop=256): ~87 time frames  — standard timbre, harmonics
    Slow (hop=1024): ~22 time frames  — modulation, LFO, envelope shape

    Each channel: 192 bins × 24 bins/oct, fmin=C1=32.7Hz.
    Output: (B, 576, T_medium) — 3×192 channels, interpolated to medium grid.

    This directly helps with:
      - ADSR timing (fast catches attack, slow catches release)
      - LFO detection (slow sees the modulation cycle)
      - Pitch stability (fast+medium together disambiguate harmonic from noise)
    """

    def __init__(self, sample_rate: int = 44100, n_bins: int = 192, top_db: float = 80.0):
        super().__init__()
        self.fast   = CQTFrontend(sample_rate, hop_length=64,  n_bins=n_bins, top_db=top_db)
        self.medium = CQTFrontend(sample_rate, hop_length=256, n_bins=n_bins, top_db=top_db)
        self.slow   = CQTFrontend(sample_rate, hop_length=1024, n_bins=n_bins, top_db=top_db)
        self.n_bins = n_bins

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Returns (B, n_bins*3, T_medium)."""
        f = self.fast(audio)     # (B, 192, T_fast)
        m = self.medium(audio)   # (B, 192, T_medium)
        s = self.slow(audio)     # (B, 192, T_slow)
        # Interpolate to medium time resolution
        if f.shape[-1] != m.shape[-1]:
            f = nn.functional.interpolate(f, size=m.shape[-1], mode="linear")
        if s.shape[-1] != m.shape[-1]:
            s = nn.functional.interpolate(s, size=m.shape[-1], mode="linear")
        return torch.cat([f, m, s], dim=1)  # (B, 576, T_medium)


# ── Legacy Multi-Resolution ───────────────────────────────────────────────────


class MultiResolutionFrontend(nn.Module):
    """Stack of 3 spectrograms at different time resolutions (generic).

    Output: (B, n_bins*3, T_medium) — concatenated along frequency axis.
    """

    def __init__(
        self, sample_rate: int = 44100, n_bins: int = 64, mode: str = "cqt",
        top_db: float = 80.0,
    ):
        super().__init__()
        if mode == "cqt":
            frontend_cls = CQTFrontend; kwargs = {"n_bins": n_bins}
        elif mode == "mel":
            frontend_cls = MelFrontend; kwargs = {"n_mels": n_bins}
        elif mode == "gammatone":
            frontend_cls = GammatoneFrontend; kwargs = {"n_bins": n_bins}
        else:
            raise ValueError(f"Unknown multi mode: {mode!r}")

        self.fast   = frontend_cls(sample_rate, hop_length=64,  top_db=top_db, **kwargs)
        self.medium = frontend_cls(sample_rate, hop_length=256, top_db=top_db, **kwargs)
        self.slow   = frontend_cls(sample_rate, hop_length=1024, top_db=top_db, **kwargs)
        self.n_bins = n_bins

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        f = self.fast(audio)
        m = self.medium(audio)
        s = self.slow(audio)
        if f.shape[-1] != m.shape[-1]:
            f = nn.functional.interpolate(f, size=m.shape[-1], mode="linear")
        if s.shape[-1] != m.shape[-1]:
            s = nn.functional.interpolate(s, size=m.shape[-1], mode="linear")
        return torch.cat([f, m, s], dim=1)


# ── Factory ────────────────────────────────────────────────────────────────────


def build_frontend(name: str = "cqt", **kwargs) -> nn.Module:
    """Build audio frontend by name.

    Args:
        name: "cqt", "gammatone", "mel", or "multi"
        n_bins: frequency channels (mapped to correct kwarg per frontend type)
    """
    kwargs = dict(kwargs)
    n_bins = kwargs.pop("n_bins", kwargs.pop("n_mels", 192))

    if name == "cqt":
        return CQTFrontend(n_bins=n_bins, **kwargs)
    elif name == "multires":
        return MultiResCQT(n_bins=n_bins, **kwargs)
    elif name == "gammatone":
        return GammatoneFrontend(n_bins=n_bins, **kwargs)
    elif name == "mel":
        return MelFrontend(n_mels=n_bins, **kwargs)
    elif name == "multi":
        return MultiResolutionFrontend(n_bins=n_bins, **kwargs)
    else:
        raise ValueError(f"Unknown frontend: {name!r}")
