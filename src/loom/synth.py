import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.envelope import ADSR
from loom.filters import BiquadFilter
from loom.amplifier import VCA
from loom.effects.distortion import Distortion


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer.

    Signal flow: Oscillator -> Filter (with envelope) -> VCA (with envelope) -> Distortion

    The filter envelope modulates cutoff: effective_cutoff = cutoff + amount * filt_env.
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.amp_envelope = ADSR(sample_rate, n_samples)
        self.filter_envelope = ADSR(sample_rate, n_samples)
        self.filter = BiquadFilter(sample_rate)
        self.vca = VCA()
        self.distortion = Distortion()

    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary.

        Args:
            params: Dict with keys matching the parameter table in the spec.

        Returns:
            (batch, n_samples) audio tensor.
        """
        # Oscillator
        audio = self.oscillator(
            params["osc_pitch"],
            params["osc_waveform"],
            params["osc_detune"],
        )

        # Filter envelope -> modulate cutoff
        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        # filt_env_amount: [0,1] normalized, treat 0.5 as zero modulation
        amount = (params["filt_env_amount"] - 0.5) * 2.0  # -> [-1, 1]
        filt_env_mean = filt_env.mean(dim=1)  # (batch,)
        modulated_cutoff = (
            params["filter_cutoff"] + amount * filt_env_mean * 0.3
        ).clamp(0.0, 1.0)

        # Filter
        audio = self.filter(
            audio, modulated_cutoff, params["filter_q"], params["filter_type"]
        )

        # Amplitude envelope + VCA
        amp_env = self.amp_envelope(
            params["amp_attack"],
            params["amp_decay"],
            params["amp_sustain"],
            params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        # Distortion
        audio = self.distortion(audio, params["dist_amount"], params["dist_mix"])

        return audio
