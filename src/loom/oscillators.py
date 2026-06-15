import torch
import torch.nn as nn
import math


class AdditiveOscillator(nn.Module):
    """Bandlimited oscillator via additive harmonic synthesis.

    Waveforms are weighted sums of sinusoidal harmonics. The 4 waveform types
    (sine, saw, square, triangle) are blended via a continuous weight vector,
    making waveform selection differentiable.

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
        max_harmonics: Maximum number of harmonics to sum.
    """

    MIDI_MIN = 24
    MIDI_MAX = 96

    def __init__(self, sample_rate: int, n_samples: int, max_harmonics: int = 128):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.max_harmonics = max_harmonics
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0  # [0,1] -> [-100, +100] cents

    def _harmonic_amplitudes(self, n_harmonics: int, device: torch.device):
        """Compute per-harmonic amplitudes for each waveform type.

        Returns: (4, n_harmonics) tensor -- rows are [sine, saw, square, tri].
        """
        n = torch.arange(1, n_harmonics + 1, dtype=torch.float32, device=device)

        sine = torch.zeros(n_harmonics, device=device)
        sine[0] = 1.0

        saw = 1.0 / n
        saw = saw * (2.0 / math.pi)

        square = torch.where(n % 2 == 1, 1.0 / n, torch.zeros_like(n))
        square = square * (4.0 / math.pi)

        tri = torch.where(
            n % 2 == 1,
            ((-1.0) ** ((n - 1) / 2.0)) / (n * n),
            torch.zeros_like(n),
        )
        tri = tri * (8.0 / (math.pi**2))

        return torch.stack([sine, saw, square, tri], dim=0)

    def forward(
        self,
        pitch: torch.Tensor,
        waveform: torch.Tensor,
        detune: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render audio from oscillator parameters.

        Args:
            pitch: (batch,) normalized pitch [0,1] -> MIDI [24,96].
            waveform: (batch, 4) waveform blend weights (sine, saw, square, tri).
            detune: (batch,) optional normalized detune [0,1] -> [-100, +100] cents.

        Returns:
            (batch, n_samples) audio tensor in roughly [-1, 1].
        """
        batch = pitch.shape[0]
        f0 = self._denorm_pitch(pitch)

        if detune is not None:
            cents = self._denorm_detune(detune)
            f0 = f0 * torch.pow(2.0, cents / 1200.0)

        nyquist = self.sample_rate / 2.0
        max_h = torch.clamp(
            torch.floor(nyquist / f0).long(), min=1, max=self.max_harmonics
        )
        n_h = max_h.max().item()

        harm_amps = self._harmonic_amplitudes(n_h, pitch.device)
        blended = torch.einsum("bw,wh->bh", waveform, harm_amps)

        harmonic_n = torch.arange(1, n_h + 1, device=pitch.device).float()
        mask = harmonic_n.unsqueeze(0) <= max_h.unsqueeze(1)
        blended = blended * mask.float()

        freqs = f0.unsqueeze(1) * harmonic_n.unsqueeze(0)
        phases = (
            2.0 * math.pi * freqs.unsqueeze(2) * self.t.unsqueeze(0).unsqueeze(0)
        )
        harmonics = torch.sin(phases)

        audio = torch.einsum("bh,bht->bt", blended, harmonics)
        return audio
