import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WavetableOscillator(nn.Module):
    """Wavetable oscillator with frame morphing.

    Reads from a 2D wavetable (n_frames, frame_size) using grid_sample
    for phase interpolation and linear blending for frame morphing.

    Built-in wavetable: 16 frames morphing from saw to square.
    """

    MIDI_MIN = 24
    MIDI_MAX = 96

    def __init__(self, sample_rate: int, n_samples: int, n_frames: int = 16, frame_size: int = 2048):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_samples = n_samples
        self.n_frames = n_frames
        self.frame_size = frame_size

        wavetable = self._build_default_wavetable(n_frames, frame_size)
        self.register_buffer("wavetable", wavetable)

        t = torch.arange(n_samples, dtype=torch.float32) / sample_rate
        self.register_buffer("t", t)

    def _build_default_wavetable(self, n_frames: int, frame_size: int) -> torch.Tensor:
        """Build saw-to-square morph wavetable."""
        n_harmonics = frame_size // 2
        n = torch.arange(1, n_harmonics + 1, dtype=torch.float32)
        phase = torch.linspace(0, 2 * math.pi, frame_size, dtype=torch.float32).unsqueeze(0)
        harmonics = torch.sin(n.unsqueeze(1) * phase)

        frames = []
        for i in range(n_frames):
            alpha = i / max(n_frames - 1, 1)
            saw_amps = 1.0 / n * (2.0 / math.pi)
            square_amps = torch.where(n % 2 == 1, 1.0 / n * (4.0 / math.pi), torch.zeros_like(n))
            amps = (1.0 - alpha) * saw_amps + alpha * square_amps
            frame = (amps.unsqueeze(1) * harmonics).sum(dim=0)
            peak = frame.abs().max().clamp(min=1e-6)
            frames.append(frame / peak)

        return torch.stack(frames, dim=0)

    def _midi_to_hz(self, midi: torch.Tensor) -> torch.Tensor:
        return 440.0 * torch.pow(2.0, (midi - 69.0) / 12.0)

    def _denorm_pitch(self, pitch: torch.Tensor) -> torch.Tensor:
        midi = pitch * (self.MIDI_MAX - self.MIDI_MIN) + self.MIDI_MIN
        return self._midi_to_hz(midi)

    def _denorm_detune(self, detune: torch.Tensor) -> torch.Tensor:
        return (detune - 0.5) * 200.0

    def forward(self, pitch: torch.Tensor, detune: torch.Tensor, position: torch.Tensor, freq_mod=None) -> torch.Tensor:
        """Render audio from wavetable.

        Args:
            pitch: (batch,) normalized pitch [0,1] -> MIDI [24,96].
            detune: (batch,) normalized detune [0,1] -> [-100, +100] cents.
            position: (batch,) wavetable position [0,1] for frame morphing.
            freq_mod: (batch, n_samples) optional per-sample multiplicative frequency
                modulator centered at 1.0. When provided, f(t) = f0 * freq_mod[t].
        Returns:
            (batch, n_samples) audio tensor.
        """
        batch = pitch.shape[0]
        f0 = self._denorm_pitch(pitch)
        cents = self._denorm_detune(detune)
        f0 = f0 * torch.pow(2.0, cents / 1200.0)

        # Phase accumulation normalized [0, 1)
        if freq_mod is not None:
            f_t = f0.unsqueeze(1) * freq_mod
            phase_inc = f_t / self.sample_rate
            phase = torch.cumsum(phase_inc, dim=1) % 1.0
        else:
            phase_inc = f0 / self.sample_rate
            phase = torch.cumsum(phase_inc.unsqueeze(1).expand(-1, self.n_samples), dim=1)
            phase = phase % 1.0

        # Frame interpolation
        pos_scaled = position * (self.n_frames - 1)
        frame_lo = pos_scaled.long().clamp(0, self.n_frames - 2)
        frame_hi = (frame_lo + 1).clamp(max=self.n_frames - 1)
        frac = (pos_scaled - frame_lo.float()).unsqueeze(1)

        wt_lo = self.wavetable[frame_lo]
        wt_hi = self.wavetable[frame_hi]
        wt_blended = (1.0 - frac) * wt_lo + frac * wt_hi

        # Read from blended wavetable using grid_sample.
        # Model the 1D wavetable as a (1, frame_size) "image" so grid_sample
        # can do linear interpolation on the phase dimension.
        # wt_4d shape: (batch, 1, 1, frame_size)
        wt_4d = wt_blended.unsqueeze(1).unsqueeze(2)
        # x coordinate maps phase [0,1) -> [-1, 1] (width dimension)
        # y coordinate is always 0 (center of the single-row image)
        grid_x = phase * 2.0 - 1.0          # (batch, n_samples)
        grid_y = torch.zeros_like(grid_x)    # (batch, n_samples)
        # grid shape: (batch, 1, n_samples, 2)
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(1)

        # padding_mode="border" avoids zero-padding artefacts at phase wrap boundary
        audio = F.grid_sample(wt_4d, grid, mode="bilinear", padding_mode="border", align_corners=True)
        # audio shape: (batch, 1, 1, n_samples) -> (batch, n_samples)
        return audio.squeeze(1).squeeze(1)
