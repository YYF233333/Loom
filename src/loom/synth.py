import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.envelope import ADSR
from loom.filters import BiquadFilter
from loom.amplifier import VCA
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer with full effects chain.

    Signal flow:
        Oscillator -> Filter (with envelope) -> VCA (with envelope)
        -> Distortion -> Compressor -> Chorus -> Delay -> Reverb -> EQ
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.amp_envelope = ADSR(sample_rate, n_samples)
        self.filter_envelope = ADSR(sample_rate, n_samples)
        self.filter = BiquadFilter(sample_rate)
        self.vca = VCA()
        self.distortion = Distortion()
        self.compressor = Compressor()
        self.chorus = Chorus(sample_rate, n_samples)
        self.delay = Delay(sample_rate, n_samples)
        self.reverb = Reverb(sample_rate, n_samples)
        self.eq = EQ(sample_rate)

    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary."""
        audio = self.oscillator(
            params["osc_pitch"],
            params["osc_waveform"],
            params["osc_detune"],
        )

        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        filt_env_mean = filt_env.mean(dim=1)
        modulated_cutoff = (
            params["filter_cutoff"] + amount * filt_env_mean * 0.3
        ).clamp(0.0, 1.0)

        audio = self.filter(
            audio, modulated_cutoff, params["filter_q"], params["filter_type"]
        )

        amp_env = self.amp_envelope(
            params["amp_attack"],
            params["amp_decay"],
            params["amp_sustain"],
            params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        audio = self.distortion(audio, params["dist_amount"], params["dist_mix"])
        audio = self.compressor(
            audio,
            params["comp_threshold"],
            params["comp_ratio"],
            params["comp_attack"],
            params["comp_release"],
            params["comp_makeup"],
            params["comp_mix"],
        )
        audio = self.chorus(
            audio, params["chorus_rate"], params["chorus_depth"], params["chorus_mix"]
        )
        audio = self.delay(
            audio, params["delay_time"], params["delay_feedback"], params["delay_mix"]
        )
        audio = self.reverb(
            audio,
            params["reverb_room_size"],
            params["reverb_decay"],
            params["reverb_damping"],
            params["reverb_mix"],
        )
        audio = self.eq(
            audio, params["eq_low_gain"], params["eq_mid_gain"], params["eq_high_gain"]
        )

        return audio
