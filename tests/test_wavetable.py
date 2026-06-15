import torch
import pytest
from loom.wavetable import WavetableOscillator
from loom.core import SAMPLE_RATE, N_SAMPLES, DEVICE


class TestWavetableOscillator:
    def setup_method(self):
        self.osc = WavetableOscillator(
            sample_rate=SAMPLE_RATE, n_samples=N_SAMPLES
        ).to(DEVICE)

    def test_output_shape(self):
        batch = 4
        pitch = torch.full((batch,), 0.5, device=DEVICE)
        detune = torch.full((batch,), 0.5, device=DEVICE)
        position = torch.full((batch,), 0.5, device=DEVICE)
        audio = self.osc(pitch, detune, position)
        assert audio.shape == (batch, N_SAMPLES)

    def test_frequency(self):
        """Should produce correct fundamental frequency."""
        midi_note = 69  # A4
        pitch = torch.tensor([(midi_note - 24) / (96 - 24)], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        position = torch.tensor([0.0], device=DEVICE)
        audio = self.osc(pitch, detune, position)

        fft = torch.fft.rfft(audio[0])
        magnitudes = torch.abs(fft)
        freqs = torch.fft.rfftfreq(N_SAMPLES, 1.0 / SAMPLE_RATE).to(DEVICE)
        peak_freq = freqs[torch.argmax(magnitudes[1:]) + 1]
        assert abs(peak_freq.item() - 440.0) < 2.0

    def test_position_changes_timbre(self):
        """Different wt_position should produce different waveforms."""
        pitch = torch.tensor([0.5], device=DEVICE)
        detune = torch.tensor([0.5], device=DEVICE)
        audio_a = self.osc(pitch, detune, torch.tensor([0.0], device=DEVICE))
        audio_b = self.osc(pitch, detune, torch.tensor([1.0], device=DEVICE))
        assert not torch.allclose(audio_a, audio_b)

    def test_no_nan(self):
        pitch = torch.tensor([0.01], device=DEVICE)
        detune = torch.tensor([0.99], device=DEVICE)
        position = torch.tensor([0.99], device=DEVICE)
        audio = self.osc(pitch, detune, position)
        assert not torch.isnan(audio).any()
