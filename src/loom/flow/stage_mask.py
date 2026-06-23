"""Stage-based parameter masks — exact match with random_params() in render.py.

Critical: the flow network must know which params are CONSTANT at each stage.
We fix params after ODE sampling by overwriting with correct constant values.
"""

import torch
from loom.training.dataset import N_PARAMS


def get_stage_fixed_vector(stage: int) -> torch.Tensor:
    """Return (N_PARAMS,) vector with NaN=varying, value=fixed.

    Exact match with render.random_params() per-stage behavior.
    Stage 99 = all vary (all NaN).
    """
    vec = torch.full((N_PARAMS,), float("nan"))

    # ── Common fixed defaults (overridden per stage) ──
    # Continuous indices and their fixed values when NOT varying:
    defaults = {
        2: 0.5,   # wt_position
        3: 0.5,   # fm_carrier_ratio
        4: 0.5,   # fm_mod_ratio
        5: 0.0,   # fm_mod_index
        6: 0.05,  # amp_attack
        7: 0.3,   # amp_decay
        8: 0.8,   # amp_sustain
        9: 0.3,   # amp_release
        13: 0.1,  # filt_env_attack
        14: 0.3,  # filt_env_decay
        15: 0.7,  # filt_env_sustain
        16: 0.3,  # filt_env_release
        17: 0.5,  # filt_env_amount
        18: 0.0,  # dist_amount
        19: 0.0,  # dist_mix
        20: 0.7,  # master_gain
        21: 0.5,  # comp_threshold
        22: 0.3,  # comp_ratio
        23: 0.3,  # comp_attack
        24: 0.3,  # comp_release
        25: 0.5,  # comp_makeup
        26: 0.0,  # comp_mix
        27: 0.3,  # chorus_rate
        28: 0.3,  # chorus_depth
        29: 0.0,  # chorus_mix
        30: 0.3,  # delay_time
        31: 0.0,  # delay_feedback
        32: 0.0,  # delay_mix
        33: 0.3,  # reverb_room_size
        34: 0.3,  # reverb_decay
        35: 0.5,  # reverb_damping
        36: 0.0,  # reverb_mix
        37: 0.5,  # eq_low_gain
        38: 0.5,  # eq_mid_gain
        39: 0.5,  # eq_high_gain
        40: 0.5,  # lfo_rate
        41: 0.0,  # lfo_depth
        42: 0.0,  # lfo_phase
    }
    for idx, val in defaults.items():
        vec[idx] = val

    # Categorical defaults (one-hot → fixed class 0)
    # osc_waveform[43:47], osc_type[47:50], filter_type[50:53],
    # lfo_waveform[53:57], lfo_target[57:61]
    vec[47] = 1.0; vec[48] = 0.0; vec[49] = 0.0  # osc_type → class 0
    vec[53] = 1.0; vec[54] = 0.0; vec[55] = 0.0; vec[56] = 0.0  # lfo_waveform → 0
    vec[57] = 1.0; vec[58] = 0.0; vec[59] = 0.0; vec[60] = 0.0  # lfo_target → 0
    # Routing: all zeros
    vec[61:] = 0.0

    # ── Stage-specific overrides: mark varying params as NaN ──

    if stage == 0:
        # Only osc_pitch + osc_waveform vary
        vec[0] = float("nan")    # osc_pitch
        vec[1] = 0.5             # osc_detune fixed
        vec[10] = 1.0            # filter_cutoff=1.0 (fully open)
        vec[11] = 0.3            # filter_q fixed
        vec[12] = 0.0            # filter_mix=0 (bypass)
        vec[43:47] = float("nan")  # osc_waveform varies
        vec[50] = 1.0; vec[51] = 0.0; vec[52] = 0.0  # filter_type → class 0

    elif stage == 1:
        # osc_pitch, osc_detune, osc_waveform vary
        vec[0] = float("nan")    # osc_pitch
        vec[1] = float("nan")    # osc_detune
        vec[43:47] = float("nan")  # osc_waveform
        # osc_type fixed to class 0 (already set above)
        # filter_cutoff, filter_q, filter_type vary
        vec[10] = float("nan")   # filter_cutoff
        vec[11] = float("nan")   # filter_q
        vec[12] = 1.0            # filter_mix=1.0 (fixed)
        vec[50:53] = float("nan")  # filter_type varies

    elif stage == 2:
        # osc + filter + envelope all vary
        vec[0:6] = float("nan")     # osc_pitch, detune, wt_pos, fm params
        vec[6:10] = float("nan")    # amp ADSR
        vec[10:18] = float("nan")   # filter cutoff, q, mix, filt_env
        vec[20] = float("nan")      # master_gain
        vec[43:50] = float("nan")   # osc_waveform, osc_type
        vec[50:53] = float("nan")   # filter_type

    elif stage == 3:
        # All synth params + mild FX vary
        vec[0:43] = float("nan")    # All continuous
        vec[43:61] = float("nan")   # All categorical
        vec[61:] = float("nan")     # Routing

    return vec


def apply_stage_fix(params: torch.Tensor, stage: int) -> torch.Tensor:
    """Overwrite fixed parameters with their correct constant values.

    Args:
        params: (B, N_PARAMS) — ODE output
        stage: curriculum stage
    Returns:
        params with fixed dims overwritten to correct values
    """
    if stage >= 99:
        return params
    fixed = get_stage_fixed_vector(stage).to(params.device)
    mask = torch.isnan(fixed)  # True = varies, False = fixed
    if mask.all():
        return params
    result = params.clone()
    result[:, ~mask] = fixed[~mask].unsqueeze(0).expand(params.shape[0], -1)
    return result


def get_stage_mask(stage: int) -> torch.Tensor:
    """Return (N_PARAMS,) boolean mask: True = varies, False = fixed."""
    fixed = get_stage_fixed_vector(stage)
    return torch.isnan(fixed)
