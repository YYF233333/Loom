import torch
import torch.nn as nn
import math


class ADSR(nn.Module):
    """Differentiable ADSR envelope using multiplicative ramp decomposition.

    Each ADSR stage is a monotonic ramp computed over the full time axis.
    The final envelope is their product, avoiding hard if/else transitions.

    All time parameters are normalized [0,1] and mapped to physical
    durations via log scale. Sustain is linear [0,1].

    Args:
        sample_rate: Audio sample rate in Hz.
        n_samples: Number of output samples.
        note_on_duration: Duration in seconds the note is held before release.
            Defaults to 3.0 (release starts at t=3.0 for a 4s buffer).
    """

    MIN_MS = 1.0
    MAX_ATTACK_MS = 2000.0
    MAX_DECAY_MS = 2000.0
    MAX_RELEASE_MS = 4000.0

    def __init__(
        self, sample_rate: int, n_samples: int, note_on_duration: float = 3.0
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.note_on_duration = note_on_duration
        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _denorm_time(
        self, normalized: torch.Tensor, max_ms: float
    ) -> torch.Tensor:
        """[0,1] -> seconds via log scale."""
        log_min = math.log(self.MIN_MS)
        log_max = math.log(max_ms)
        ms = torch.exp(normalized * (log_max - log_min) + log_min)
        return ms / 1000.0

    def forward(
        self,
        attack: torch.Tensor,
        decay: torch.Tensor,
        sustain: torch.Tensor,
        release: torch.Tensor,
        note_on_duration=None,
    ) -> torch.Tensor:
        """Generate ADSR envelope.

        Args:
            attack: (batch,) normalized attack time [0,1].
            decay: (batch,) normalized decay time [0,1].
            sustain: (batch,) sustain level [0,1].
            release: (batch,) normalized release time [0,1].
            note_on_duration: Duration in seconds before release begins. If
                None, uses the value set in the constructor.

        Returns:
            (batch, n_samples) envelope in [0, 1].
        """
        if note_on_duration is None:
            note_on_duration = self.note_on_duration
        a_sec = self._denorm_time(attack, self.MAX_ATTACK_MS)
        d_sec = self._denorm_time(decay, self.MAX_DECAY_MS)
        r_sec = self._denorm_time(release, self.MAX_RELEASE_MS)
        s_level = sustain

        t = self.t.unsqueeze(0)

        # Attack ramp: 0 -> 1 over a_sec
        a_sec_safe = a_sec.unsqueeze(1).clamp(min=1e-6)
        attack_ramp = (t / a_sec_safe).clamp(0.0, 1.0)

        # Decay ramp: 1 -> sustain over d_sec, starting at a_sec
        d_sec_safe = d_sec.unsqueeze(1).clamp(min=1e-6)
        a_sec_expanded = a_sec.unsqueeze(1)
        decay_progress = ((t - a_sec_expanded) / d_sec_safe).clamp(0.0, 1.0)
        s_expanded = s_level.unsqueeze(1)
        decay_ramp = 1.0 - (1.0 - s_expanded) * decay_progress

        # Release ramp: sustain -> 0 over r_sec, starting at note_on_duration
        r_sec_safe = r_sec.unsqueeze(1).clamp(min=1e-6)
        release_progress = (
            (t - note_on_duration) / r_sec_safe
        ).clamp(0.0, 1.0)
        release_ramp = 1.0 - release_progress

        envelope = attack_ramp * decay_ramp * release_ramp
        return envelope
