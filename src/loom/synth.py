import torch
import torch.nn as nn

from loom.oscillators import AdditiveOscillator
from loom.wavetable import WavetableOscillator
from loom.fm import FMOscillator
from loom.envelope import ADSR
from loom.svfilter import SVFilter
from loom.amplifier import VCA
from loom.effects.chain import EffectsChain
from loom.lfo import LFO


class SubtractiveSynth(nn.Module):
    """Complete subtractive synthesizer with Sinkhorn-routed effects chain.

    Signal flow:
        Oscillator -> Filter (with envelope) -> VCA (with envelope)
        -> EffectsChain (Sinkhorn-routed or fixed canonical order)
    """

    def __init__(self, sample_rate: int, n_samples: int, note_on_duration: float = 3.0):
        super().__init__()
        self.oscillator = AdditiveOscillator(sample_rate, n_samples)
        self.wavetable_osc = WavetableOscillator(sample_rate, n_samples)
        self.fm_osc = FMOscillator(sample_rate, n_samples)
        self.amp_envelope = ADSR(sample_rate, n_samples, note_on_duration=note_on_duration)
        self.filter_envelope = ADSR(sample_rate, n_samples, note_on_duration=note_on_duration)
        self.filter = SVFilter(sample_rate)
        self.vca = VCA()
        self.effects_chain = EffectsChain(sample_rate, n_samples)
        self.lfo = LFO(sample_rate, n_samples)

    def forward(self, params: dict[str, torch.Tensor], return_intermediates: bool = False) -> torch.Tensor:
        n_samples = self.oscillator.n_samples

        # LFO signal: (batch, n_samples)
        lfo_signal = self.lfo(
            params["lfo_rate"],
            params["lfo_depth"],
            params["lfo_waveform"],
            params["lfo_phase"],
        )
        lfo_target = params["lfo_target"]  # (batch, 4)

        # Per-sample frequency modulation for oscillators
        pitch_lfo = lfo_target[:, 1:2] * lfo_signal * 0.05
        freq_mod = 1.0 + pitch_lfo
        has_pitch_mod = (params["lfo_depth"].abs() > 1e-4).any() and (lfo_target[:, 1].abs() > 1e-4).any()
        fm_arg = freq_mod if has_pitch_mod else None

        osc_type = params["osc_type"]

        if torch.is_grad_enabled():
            additive_out = self.oscillator(
                params["osc_pitch"], params["osc_waveform"], params["osc_detune"],
                freq_mod=fm_arg,
            )
            wavetable_out = self.wavetable_osc(
                params["osc_pitch"], params["osc_detune"], params["wt_position"],
                freq_mod=fm_arg,
            )
            fm_out = self.fm_osc(
                params["osc_pitch"], params["osc_detune"],
                params["fm_carrier_ratio"], params["fm_mod_ratio"], params["fm_mod_index"],
                freq_mod=fm_arg,
            )
            audio = (
                osc_type[:, 0:1] * additive_out
                + osc_type[:, 1:2] * wavetable_out
                + osc_type[:, 2:3] * fm_out
            )
        else:
            batch = osc_type.shape[0]
            audio = torch.zeros(batch, n_samples, device=osc_type.device)
            mask_add = osc_type[:, 0] > 0.5
            mask_wt = osc_type[:, 1] > 0.5
            mask_fm = osc_type[:, 2] > 0.5
            if mask_add.any():
                idx = mask_add.nonzero(as_tuple=True)[0]
                audio[idx] = self.oscillator(
                    params["osc_pitch"][idx], params["osc_waveform"][idx],
                    params["osc_detune"][idx],
                    freq_mod=fm_arg[idx] if fm_arg is not None else None,
                )
            if mask_wt.any():
                idx = mask_wt.nonzero(as_tuple=True)[0]
                audio[idx] = self.wavetable_osc(
                    params["osc_pitch"][idx], params["osc_detune"][idx],
                    params["wt_position"][idx],
                    freq_mod=fm_arg[idx] if fm_arg is not None else None,
                )
            if mask_fm.any():
                idx = mask_fm.nonzero(as_tuple=True)[0]
                audio[idx] = self.fm_osc(
                    params["osc_pitch"][idx], params["osc_detune"][idx],
                    params["fm_carrier_ratio"][idx], params["fm_mod_ratio"][idx],
                    params["fm_mod_index"][idx],
                    freq_mod=fm_arg[idx] if fm_arg is not None else None,
                )

        osc_out = audio

        # Filter: time-varying cutoff (envelope + LFO)
        filt_env = self.filter_envelope(
            params["filt_env_attack"], params["filt_env_decay"],
            params["filt_env_sustain"], params["filt_env_release"],
        )
        amount = (params["filt_env_amount"] - 0.5) * 2.0
        base_cutoff = params["filter_cutoff"].unsqueeze(1)
        env_mod = amount.unsqueeze(1) * filt_env * 0.3
        lfo_cutoff = lfo_target[:, 0:1] * lfo_signal * 0.3
        cutoff_signal = (base_cutoff + env_mod + lfo_cutoff).clamp(0.0, 1.0)

        audio = self.filter(audio, cutoff_signal, params["filter_q"], params["filter_type"],
                            mix=params.get("filter_mix"))

        filter_out = audio

        # Amplitude envelope + VCA
        amp_env = self.amp_envelope(
            params["amp_attack"], params["amp_decay"],
            params["amp_sustain"], params["amp_release"],
        )
        audio = self.vca(audio, amp_env, params["master_gain"])

        # Effects chain (Sinkhorn-routed when fx_routing provided)
        dist_lfo = lfo_target[:, 2:3] * lfo_signal * 0.3
        dist_drive = (params["dist_amount"].unsqueeze(1) + dist_lfo).clamp(0.0, 1.0)

        fx_params = {
            "dist_drive": dist_drive,
            "dist_mix": params["dist_mix"],
            "comp_threshold": params["comp_threshold"],
            "comp_ratio": params["comp_ratio"],
            "comp_attack": params["comp_attack"],
            "comp_release": params["comp_release"],
            "comp_makeup": params["comp_makeup"],
            "comp_mix": params["comp_mix"],
            "chorus_rate": params["chorus_rate"],
            "chorus_depth": params["chorus_depth"],
            "chorus_mix": params["chorus_mix"],
            "delay_time": params["delay_time"],
            "delay_feedback": params["delay_feedback"],
            "delay_mix": params["delay_mix"],
            "reverb_room_size": params["reverb_room_size"],
            "reverb_decay": params["reverb_decay"],
            "reverb_damping": params["reverb_damping"],
            "reverb_mix": params["reverb_mix"],
            "eq_low_gain": params["eq_low_gain"],
            "eq_mid_gain": params["eq_mid_gain"],
            "eq_high_gain": params["eq_high_gain"],
        }

        routing = params.get("fx_routing")
        tau = params.get("fx_routing_tau", 1.0)
        if isinstance(tau, torch.Tensor):
            tau = tau.item()
        dry_out = audio

        audio = self.effects_chain(audio, fx_params, routing_logits=routing, tau=tau)

        if return_intermediates:
            return audio, {"osc": osc_out, "filter": filter_out, "dry": dry_out}
        return audio
