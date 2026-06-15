import torch
import torch.nn as nn
import math


class FMOscillator(nn.Module):
    """FM oscillator with fixed carrier/modulator frequency ratios.

    Single carrier + single modulator. Uses frequency ratios (not absolute
    frequencies) to avoid the FM frequency non-convergence problem (DDX7).
    """

    MIDI_MIN = 24
    MIDI_MAX = 96
    RATIO_MIN = 1.0
    RATIO_MAX = 8.0
    MOD_INDEX_MAX = 20.0

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0

    def _denorm_ratio(self, ratio: torch.Tensor) -> torch.Tensor:
        return ratio * (self.RATIO_MAX - self.RATIO_MIN) + self.RATIO_MIN

    def _denorm_mod_index(self, mod_index: torch.Tensor) -> torch.Tensor:
        return mod_index * self.MOD_INDEX_MAX

    def forward(self, pitch, detune, carrier_ratio, mod_ratio, mod_index):
        """Render FM audio.

        Args:
            pitch: (batch,) normalized [0,1] -> MIDI [24,96].
            detune: (batch,) normalized [0,1] -> [-100, +100] cents.
            carrier_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_index: (batch,) normalized [0,1] -> [0, 20].
        Returns:
            (batch, n_samples) audio tensor.
        """
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        c_ratio = self._denorm_ratio(carrier_ratio)
        m_ratio = self._denorm_ratio(mod_ratio)
        m_idx = self._denorm_mod_index(mod_index)

        t = self.t.unsqueeze(0)
        f0 = f0.unsqueeze(1)
        c_ratio = c_ratio.unsqueeze(1)
        m_ratio = m_ratio.unsqueeze(1)
        m_idx = m_idx.unsqueeze(1)

        mod_freq = f0 * m_ratio
        mod_phase = 2.0 * math.pi * mod_freq * t
        mod_signal = m_idx * torch.sin(mod_phase)

        carrier_freq = f0 * c_ratio
        carrier_phase = 2.0 * math.pi * carrier_freq * t + mod_signal

        return torch.sin(carrier_phase)
