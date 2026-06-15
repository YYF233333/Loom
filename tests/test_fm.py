import torch
import pytest
from loom.fm import FMOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestFMOscillator:
    def setup_method(self):
        self.osc = FMOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        out = self.osc(
            pitch=torch.full((batch,), 0.5, device=DEVICE),
            detune=torch.full((batch,), 0.5, device=DEVICE),
            carrier_ratio=torch.full((batch,), 0.0, device=DEVICE),
            mod_ratio=torch.full((batch,), 0.0, device=DEVICE),
            mod_index=torch.full((batch,), 0.3, device=DEVICE),
        )
        assert out.shape == (batch, N_SAMPLES)

    def test_zero_mod_is_sine(self):
        """With mod_index=0, FM reduces to a pure carrier sine."""
        midi_note = 69
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        out = self.osc(
            pitch=pitch,
            detune=torch.tensor([0.5], device=DEVICE),
            carrier_ratio=torch.tensor([0.0], device=DEVICE),
            mod_ratio=torch.tensor([0.0], device=DEVICE),
            mod_index=torch.tensor([0.0], device=DEVICE),
        )
        fft = torch.fft.rfft(out[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert abs(peak_freq.item() - 440.0) < 2.0

        peak_idx = torch.argmax(magnitudes[1:]) + 1
        fundamental_energy = magnitudes[peak_idx].item()
        total_energy = magnitudes[1:].sum().item()
        assert fundamental_energy / total_energy > 0.9

    def test_mod_index_adds_harmonics(self):
        """Higher mod_index should introduce more harmonics."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        carrier = torch.tensor([0.0], device=DEVICE)
        mod = torch.tensor([0.0], device=DEVICE)

        out_low = self.osc(pitch, detune, carrier, mod,
                           torch.tensor([0.05], device=DEVICE))
        out_high = self.osc(pitch, detune, carrier, mod,
                            torch.tensor([0.8], device=DEVICE))

        fft_low = torch.abs(torch.fft.rfft(out_low[0]))
        fft_high = torch.abs(torch.fft.rfft(out_high[0]))

        peak_low = torch.argmax(fft_low[1:]) + 1
        peak_high = torch.argmax(fft_high[1:]) + 1

        ratio_low = fft_low[peak_low] / fft_low[1:].sum()
        ratio_high = fft_high[peak_high] / fft_high[1:].sum()
        assert ratio_high < ratio_low

    def test_no_nan(self):
        out = self.osc(
            pitch=torch.tensor([0.99], device=DEVICE),
            detune=torch.tensor([0.99], device=DEVICE),
            carrier_ratio=torch.tensor([0.99], device=DEVICE),
            mod_ratio=torch.tensor([0.99], device=DEVICE),
            mod_index=torch.tensor([0.99], device=DEVICE),
        )
        assert not torch.isnan(out).any()
