import torch
import torch.nn as nn


class VCA(nn.Module):
    """Voltage-controlled amplifier: signal * envelope * gain.

    Gain is normalized [0,1] mapped to [-60dB, 0dB].
    """

    MIN_DB = -60.0
    MAX_DB = 0.0

    def _denorm_gain(self, gain: torch.Tensor) -> torch.Tensor:
        db = gain * (self.MAX_DB - self.MIN_DB) + self.MIN_DB
        return torch.pow(10.0, db / 20.0)

    def forward(
        self,
        signal: torch.Tensor,
        envelope: torch.Tensor,
        gain: torch.Tensor,
    ) -> torch.Tensor:
        """Apply envelope and gain to signal.

        Args:
            signal: (batch, n_samples) input audio.
            envelope: (batch, n_samples) amplitude envelope [0, 1].
            gain: (batch,) normalized gain [0,1] -> [-60dB, 0dB].

        Returns:
            (batch, n_samples) output audio.
        """
        linear_gain = self._denorm_gain(gain).unsqueeze(1)
        return signal * envelope * linear_gain
