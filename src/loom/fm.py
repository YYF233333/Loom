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

    def forward(self, pitch, detune, carrier_ratio, mod_ratio, mod_index, freq_mod=None):
        """Render FM audio.

        Args:
            pitch: (batch,) normalized [0,1] -> MIDI [24,96].
            detune: (batch,) normalized [0,1] -> [-100, +100] cents.
            carrier_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_ratio: (batch,) normalized [0,1] -> [1, 8].
            mod_index: (batch,) normalized [0,1] -> [0, 20].
            freq_mod: (batch, n_samples) optional per-sample multiplicative frequency
                modulator centered at 1.0. When provided, f(t) = f0 * freq_mod[t].
        Returns:
            (batch, n_samples) audio tensor.
        """
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        c_ratio = self._denorm_ratio(carrier_ratio)
        m_ratio = self._denorm_ratio(mod_ratio)
        m_idx = self._denorm_mod_index(mod_index)

        c_ratio = c_ratio.unsqueeze(1)
        m_ratio = m_ratio.unsqueeze(1)
        m_idx = m_idx.unsqueeze(1)

        if freq_mod is not None:
            f0_t = f0.unsqueeze(1) * freq_mod
        else:
            f0_t = f0.unsqueeze(1).expand(-1, self.n_samples)

        mod_phase = torch.cumsum(2.0 * math.pi * f0_t * m_ratio / self.sample_rate, dim=1)
        mod_signal = m_idx * torch.sin(mod_phase)
        carrier_phase = torch.cumsum(2.0 * math.pi * f0_t * c_ratio / self.sample_rate, dim=1) + mod_signal

        return torch.sin(carrier_phase)
