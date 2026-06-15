import torch
import torch.nn as nn

from loom.effects.distortion import Distortion
from loom.effects.compressor import Compressor
from loom.effects.chorus import Chorus
from loom.effects.delay import Delay
from loom.effects.reverb import Reverb
from loom.effects.eq import EQ

N_EFFECTS = 6
FX_DIST, FX_COMP, FX_CHORUS, FX_DELAY, FX_REVERB, FX_EQ = range(N_EFFECTS)


def sinkhorn_normalize(logits, tau=1.0, iters=10):
    """Sinkhorn normalization: logits (batch, N, N) -> doubly-stochastic matrix."""
    s = logits / max(tau, 1e-6)
    for _ in range(iters):
        s = s - torch.logsumexp(s, dim=-1, keepdim=True)
        s = s - torch.logsumexp(s, dim=-2, keepdim=True)
    return torch.softmax(s, dim=-1)


class EffectsChain(nn.Module):
    def __init__(self, sample_rate, n_samples):
        super().__init__()
        self.distortion = Distortion()
        self.compressor = Compressor()
        self.chorus = Chorus(sample_rate, n_samples)
        self.delay = Delay(sample_rate, n_samples)
        self.reverb = Reverb(sample_rate, n_samples)
        self.eq = EQ(sample_rate)

    def _run_effect(self, idx, audio, params):
        if idx == FX_DIST:
            return self.distortion(audio, params["dist_drive"], params["dist_mix"])
        elif idx == FX_COMP:
            return self.compressor(audio, params["comp_threshold"], params["comp_ratio"],
                                   params["comp_attack"], params["comp_release"],
                                   params["comp_makeup"], params["comp_mix"])
        elif idx == FX_CHORUS:
            return self.chorus(audio, params["chorus_rate"], params["chorus_depth"], params["chorus_mix"])
        elif idx == FX_DELAY:
            return self.delay(audio, params["delay_time"], params["delay_feedback"], params["delay_mix"])
        elif idx == FX_REVERB:
            return self.reverb(audio, params["reverb_room_size"], params["reverb_decay"],
                               params["reverb_damping"], params["reverb_mix"])
        elif idx == FX_EQ:
            return self.eq(audio, params["eq_low_gain"], params["eq_mid_gain"], params["eq_high_gain"])

    def forward(self, audio, params, routing_logits=None, tau=1.0):
        if routing_logits is None:
            for fx_idx in range(N_EFFECTS):
                audio = self._run_effect(fx_idx, audio, params)
            return audio

        P = sinkhorn_normalize(routing_logits, tau=tau)
        for slot_idx in range(N_EFFECTS):
            weights = P[:, slot_idx, :]
            outputs = []
            for fx_idx in range(N_EFFECTS):
                outputs.append(self._run_effect(fx_idx, audio, params))
            stacked = torch.stack(outputs, dim=1)
            audio = (weights.unsqueeze(-1) * stacked).sum(dim=1)
        return audio
