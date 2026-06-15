import torch
import torch.nn as nn


class Distortion(nn.Module):
    """Tanh waveshaper distortion with dry/wet mix.

    amount controls pre-gain (drive): [0,1] -> [1x, 50x].
    mix controls dry/wet blend: 0 = fully dry, 1 = fully wet.
    """

    MIN_DRIVE = 1.0
    MAX_DRIVE = 50.0

    def _denorm_drive(self, amount: torch.Tensor) -> torch.Tensor:
        return amount * (self.MAX_DRIVE - self.MIN_DRIVE) + self.MIN_DRIVE

    def forward(self, signal, amount, mix):
        """Apply distortion.

        Args:
            signal: (batch, n_samples) input audio.
            amount: (batch,) or (batch, n_samples) normalized drive [0,1].
            mix: (batch,) dry/wet [0,1].
        """
        drive = self._denorm_drive(amount)
        if drive.dim() == 1:
            drive = drive.unsqueeze(1)
        mix_v = mix
        if mix_v.dim() == 1:
            mix_v = mix_v.unsqueeze(1)
        wet = torch.tanh(signal * drive)
        return (1.0 - mix_v) * signal + mix_v * wet
