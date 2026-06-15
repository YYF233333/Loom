import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.wavetable import WavetableOscillator
from loom.fm import FMOscillator
from loom.envelope import ADSR
from loom.filters import BiquadFilter
from loom.amplifier import VCA
from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ
from loom.lfo import LFO


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer with full effects chain.

    Signal flow:
        Oscillator -> Filter (with envelope) -> VCA (with envelope)
        -> Distortion -> Compressor -> Chorus -> Delay -> Reverb -> EQ
    """

    def __init__(self, sample_rate: int, n_samples: int):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.wavetable_osc = WavetableOscillator(sample_rate, n_samples)
        self.fm_osc = FMOscillator(sample_rate, n_samples)
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
        self.lfo = LFO(sample_rate, n_samples)

    def forward(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Render audio from parameter dictionary."""
        # LFO modulation
        lfo_signal = self.lfo(
            params["lfo_rate"],
            params["lfo_depth"],
            params["lfo_waveform"],
            params["lfo_phase"],
        )
        lfo_target = params["lfo_target"]  # (batch, 4)

        # Modulated pitch (LFO uses mean of time-varying signal for scalar params)
        pitch_mod = (
            params["osc_pitch"]
            + lfo_target[:, 1] * lfo_signal.mean(dim=1) * 0.05
        ).clamp(0.0, 1.0)

        # Oscillators with modulated pitch
        additive_out = self.oscillator(
            pitch_mod,
            params["osc_waveform"],
            params["osc_detune"],
        )
        wavetable_out = self.wavetable_osc(
            pitch_mod,
            params["osc_detune"],
            params["wt_position"],
        )
        fm_out = self.fm_osc(
            pitch_mod,
            params["osc_detune"],
            params["fm_carrier_ratio"],
            params["fm_mod_ratio"],
            params["fm_mod_index"],
        )
        osc_type = params["osc_type"]
        audio = (
            osc_type[:, 0:1] * additive_out
            + osc_type[:, 1:2] * wavetable_out
            + osc_type[:, 2:3] * fm_out
        )

        # Filter with envelope + LFO modulation on cutoff
        filt_env = self.filter_envelope(
            params["filt_env_attack"],
            params["filt_env_decay"],
            params["filt_env_sustain"],
            params["filt_env_release"],
        )
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        filt_env_mean = filt_env.mean(dim=1)
        lfo_cutoff_mod = lfo_target[:, 0] * lfo_signal.mean(dim=1) * 0.3
        modulated_cutoff = (
            params["filter_cutoff"] + amount * filt_env_mean * 0.3 + lfo_cutoff_mod
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

        # Distortion with LFO modulation
        dist_mod = (
            params["dist_amount"]
            + lfo_target[:, 2] * lfo_signal.mean(dim=1) * 0.3
        ).clamp(0.0, 1.0)
        audio = self.distortion(audio, dist_mod, params["dist_mix"])

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
