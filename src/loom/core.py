import torch
import torch.nn as nn

SAMPLE_RATE = 44100
DURATION = 4.0
N_SAMPLES = int(SAMPLE_RATE * DURATION)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SynthModule(nn.Module):
    def forward(self, *args, **kwargs):
        raise NotImplementedError
