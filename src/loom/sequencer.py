import torch
from loom.synth import SubtractiveSynth
from loom.core import SAMPLE_RATE


def render_sequence(
    synth_params: dict[str, torch.Tensor],
    seq_pitch: torch.Tensor,
    seq_velocity: torch.Tensor,
    seq_gate: torch.Tensor,
    seq_timing: torch.Tensor,
    bpm: float = 170.0,
    sample_rate: int = SAMPLE_RATE,
    n_steps: int = 32,
    velocity_threshold: float = 0.05,
) -> torch.Tensor:
    """Render a 32-step sequence using SubtractiveSynth.

    Args:
        synth_params: Shared synth parameters (batch=1 tensors).
        seq_pitch: (batch, 32) per-step pitch [0,1].
        seq_velocity: (batch, 32) per-step velocity [0,1], 0=rest.
        seq_gate: (batch, 32) per-step gate length as fraction of step.
        seq_timing: (batch, 32) micro-timing offset [-0.5, 0.5] in step units.
        bpm: Tempo in BPM.
        sample_rate: Audio sample rate.
        n_steps: Number of steps (default 32).
        velocity_threshold: Minimum velocity to trigger a note.

    Returns:
        (batch, total_samples) rendered audio.
    """
    batch = seq_pitch.shape[0]
    device = seq_pitch.device
    step_sec = 60.0 / bpm / 8.0
    total_samples = int(n_steps * step_sec * sample_rate)
    output = torch.zeros(batch, total_samples, device=device)

    max_note_samples = int(step_sec * 2.0 * sample_rate)
    max_note_samples = min(max_note_samples, total_samples)

    for step in range(n_steps):
        vel = seq_velocity[:, step]
        active_mask = vel > velocity_threshold
        if not active_mask.any():
            continue

        note_start_sec = step * step_sec + seq_timing[:, step] * step_sec
        note_start_sec = note_start_sec.clamp(min=0.0)
        note_start_sample = (note_start_sec * sample_rate).long()

        gate_sec = seq_gate[:, step] * step_sec
        gate_sec = gate_sec.clamp(min=0.01)

        note_params = {}
        for key, val in synth_params.items():
            if val.shape[0] == 1 and batch > 1:
                note_params[key] = val.expand(batch, *val.shape[1:])
            else:
                note_params[key] = val.clone()

        note_params["osc_pitch"] = seq_pitch[:, step]
        base_gain = synth_params.get("master_gain", torch.tensor([0.8], device=device))
        if base_gain.shape[0] == 1 and batch > 1:
            base_gain = base_gain.expand(batch)
        note_params["master_gain"] = (base_gain * vel).clamp(0.0, 1.0)

        note_synth = SubtractiveSynth(sample_rate, max_note_samples).to(device)

        with torch.no_grad():
            note_audio = note_synth(note_params)

        for b in range(batch):
            if not active_mask[b]:
                continue
            start = note_start_sample[b].item()
            start = max(0, min(start, total_samples - 1))
            end = min(start + max_note_samples, total_samples)
            length = end - start
            if length > 0:
                output[b, start:end] = output[b, start:end] + note_audio[b, :length]

    return output
