import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class Compressor(nn.Module):
    """Differentiable feed-forward compressor.

    RMS envelope detection -> gain reduction -> smoothing -> makeup gain -> dry/wet.

    All parameters normalized [0,1], denormalized internally.
    """

    THRESH_MIN_DB = -40.0
    THRESH_MAX_DB = 0.0
    RATIO_MIN = 1.0
    RATIO_MAX = 20.0
    ATTACK_MIN_MS = 0.1
    ATTACK_MAX_MS = 100.0
    RELEASE_MIN_MS = 10.0
    RELEASE_MAX_MS = 1000.0
    MAKEUP_MIN_DB = 0.0
    MAKEUP_MAX_DB = 30.0
    RMS_WINDOW = 1024

    def _denorm_threshold(self, t: torch.Tensor) -> torch.Tensor:
        return t * (self.THRESH_MAX_DB - self.THRESH_MIN_DB) + self.THRESH_MIN_DB

    def _denorm_ratio(self, r: torch.Tensor) -> torch.Tensor:
        log_min = math.log(self.RATIO_MIN)
        log_max = math.log(self.RATIO_MAX)
        return torch.exp(r * (log_max - log_min) + log_min)

    def _denorm_makeup(self, m: torch.Tensor) -> torch.Tensor:
        db = m * (self.MAKEUP_MAX_DB - self.MAKEUP_MIN_DB) + self.MAKEUP_MIN_DB
        return torch.pow(10.0, db / 20.0)

    def _denorm_time_ms(
        self, normalized: torch.Tensor, min_ms: float, max_ms: float
    ) -> torch.Tensor:
        log_min = math.log(min_ms)
        log_max = math.log(max_ms)
        return torch.exp(normalized * (log_max - log_min) + log_min)

    def _rms_envelope(self, signal: torch.Tensor) -> torch.Tensor:
        x2 = signal.pow(2).unsqueeze(1)
        rms = F.avg_pool1d(
            x2, self.RMS_WINDOW, stride=1, padding=self.RMS_WINDOW // 2
        )
        rms = rms[:, :, : signal.shape[1]]
        return rms.squeeze(1).sqrt().clamp(min=1e-8)

    def forward(
        self,
        signal: torch.Tensor,
        threshold: torch.Tensor,
        ratio: torch.Tensor,
        attack: torch.Tensor,
        release: torch.Tensor,
        makeup: torch.Tensor,
        mix: torch.Tensor,
    ) -> torch.Tensor:
        """Apply compression.

        Args:
            signal: (batch, n_samples) input audio.
            threshold: (batch,) normalized [0,1] -> [-40dB, 0dB].
            ratio: (batch,) normalized [0,1] -> [1:1, 20:1].
            attack: (batch,) normalized [0,1] -> [0.1ms, 100ms].
            release: (batch,) normalized [0,1] -> [10ms, 1000ms].
            makeup: (batch,) normalized [0,1] -> [0dB, 30dB].
            mix: (batch,) dry/wet [0,1].
        """
        # Short-circuit when fully bypassed to avoid polluting gradients.
        if mix.max().item() < 0.02:
            return signal + 0.0 * mix.unsqueeze(1)

        thresh_db = self._denorm_threshold(threshold).unsqueeze(1)
        ratio_val = self._denorm_ratio(ratio).unsqueeze(1)
        makeup_linear = self._denorm_makeup(makeup).unsqueeze(1)
        mix = mix.unsqueeze(1)

        rms = self._rms_envelope(signal)
        rms_db = 20.0 * torch.log10(rms.clamp(min=1e-8))

        gain_db = torch.min(
            torch.zeros_like(rms_db),
            (1.0 - 1.0 / ratio_val) * (thresh_db - rms_db),
        )
        gain = torch.pow(10.0, gain_db / 20.0)

        smooth_ms = self._denorm_time_ms(
            (attack + release) / 2.0, self.ATTACK_MIN_MS, self.RELEASE_MAX_MS
        )
        # Differentiable smoothing: use a fixed-size avg_pool window
        # but blend between unsmoothed and smoothed based on the
        # continuous smooth_ms value to preserve gradients.
        smooth_factor = (smooth_ms / 1000.0 * 44100).unsqueeze(1)  # (batch, 1)
        window = 128  # fixed kernel size
        gain_pooled = F.avg_pool1d(
            gain.unsqueeze(1), window, stride=1, padding=window // 2
        )
        gain_pooled = gain_pooled[:, :, : signal.shape[1]].squeeze(1)
        # Blend: when smooth_factor is small, use unsmoothed gain;
        # when large, use fully smoothed. sigmoid maps to [0,1].
        blend = torch.sigmoid((smooth_factor - window / 2) / (window / 4))
        gain_smooth = (1.0 - blend) * gain + blend * gain_pooled

        wet = signal * gain_smooth * makeup_linear
        return (1.0 - mix) * signal + mix * wet
