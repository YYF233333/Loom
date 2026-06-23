"""Flow matching utilities — ODE sampling + gradient refinement.

Rectified Flow (Lipman et al. 2023, "Flow Matching for Generative Modeling"):
    - Probability path: x_t = (1−t)·x_0 + t·x_1
    - Velocity field:   v(x_t, t) = x_1 − x_0  (constant)
    - Training: regress v_θ(x_t, t, c) against v_true
    - Inference: solve ODE dx/dt = v_θ(x, t, c) from t=0 to t=1

This module provides the inference-side utilities:
    sample_ode:   Euler / midpoint ODE integration
    refine_grad:  Gradient-based refinement through the differentiable synth
    sample_best:  ODE sampling with multiple candidates, pick best via synth loss
"""

import torch
import torch.nn as nn

from loom.synth import SubtractiveSynth
from loom.training.losses import multi_resolution_stft_loss
from loom.training.dataset import vector_to_params, N_CONTINUOUS, N_CATEGORICAL, CATEGORICAL_KEYS


# ── Post-processing: raw ODE output → valid parameter vector ────────────────


def _raw_to_params(raw: torch.Tensor) -> torch.Tensor:
    """Convert unbounded ODE output to valid parameter vector.

    Continuous (first 43): sigmoid → [0, 1]
    Categorical (next 18): softmax per group → probability simplex
    Routing (last 36):    raw logits (unconstrained)

    Args:
        raw: (B, 97) — unbounded ODE integration result

    Returns:
        params: (B, 97) — valid parameter vector
    """
    params = torch.empty_like(raw)

    # Continuous: sigmoid
    params[:, :N_CONTINUOUS] = torch.sigmoid(raw[:, :N_CONTINUOUS])

    # Categorical: softmax per group
    cat_start = N_CONTINUOUS
    for _, n_classes in CATEGORICAL_KEYS:
        cat_end = cat_start + n_classes
        params[:, cat_start:cat_end] = torch.softmax(raw[:, cat_start:cat_end], dim=-1)
        cat_start = cat_end

    # Routing: raw (unconstrained)
    params[:, N_CONTINUOUS + N_CATEGORICAL:] = raw[:, N_CONTINUOUS + N_CATEGORICAL:]

    return params


# ── ODE Solvers ─────────────────────────────────────────────────────────────


@torch.no_grad()
def sample_euler(
    flow_net: nn.Module,
    audio_cond: torch.Tensor,
    audio_latents: torch.Tensor | None = None,
    n_steps: int = 20,
    stage: int = 99,
    return_trajectory: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
    """Euler integration of the learned ODE.

    ODE operates in unbounded space. Final output is constrained via
    sigmoid/softmax to produce valid parameter vectors.

    Args:
        flow_net:    FlowNetwork that predicts velocity v(x_t, t, audio_cond)
        audio_cond:  (B, d_cond) pre-computed audio condition
        audio_latents: (B, N_AUDIO, d_model) optional audio tokens
        n_steps:     Number of integration steps
        return_trajectory: If True, also return intermediate x values

    Returns:
        params: (B, N_PARAMS) — valid parameter vector
    """
    B = audio_cond.shape[0]
    device = audio_cond.device

    # In-distribution start: N(0, 0.3²) → sigmoid → valid-ish [0,1] range
    # Much closer to actual params than N(0,1) which produces extreme values
    x = torch.randn(B, 97, device=device) * 0.3
    dt = 1.0 / n_steps
    trajectory = [x.clone()] if return_trajectory else None

    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        v = flow_net(x, t, audio_cond, audio_latents)
        x = x + v * dt

    params = _raw_to_params(x)
    if stage < 99:
        from loom.flow.stage_mask import apply_stage_fix
        params = apply_stage_fix(params, stage)
    if return_trajectory:
        return params, trajectory
    return params


@torch.no_grad()
def sample_midpoint(
    flow_net: nn.Module,
    audio_cond: torch.Tensor,
    audio_latents: torch.Tensor | None = None,
    n_steps: int = 10,
) -> torch.Tensor:
    """Midpoint method — 2nd order."""
    B = audio_cond.shape[0]
    device = audio_cond.device
    x = torch.randn(B, 97, device=device)
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t = torch.full((B,), step * dt, device=device)
        t_mid = torch.full((B,), (step + 0.5) * dt, device=device)
        k1 = flow_net(x, t, audio_cond, audio_latents)
        k2 = flow_net(x + k1 * dt * 0.5, t_mid, audio_cond, audio_latents)
        x = x + k2 * dt
    return _raw_to_params(x)


# ── Gradient Refinement ─────────────────────────────────────────────────────


def refine_via_gradient(
    flow_net: nn.Module,
    synth: nn.Module,
    audio_cond: torch.Tensor,
    target_audio: torch.Tensor,
    init_params: torch.Tensor,
    n_steps: int = 50,
    lr: float = 0.01,
    noise_scale: float = 0.001,
) -> tuple[torch.Tensor, list[float]]:
    """Refine flow sample via gradient descent through differentiable synth.

    This is the "belt and suspenders" approach:
    1. Flow matching gives a good initial guess (in the right basin)
    2. Gradient descent through the synth polishes it to exact match

    Args:
        flow_net:      FlowNetwork (used to get initial guess, not during refinement)
        synth:         SubtractiveSynth for differentiable rendering
        audio_cond:    (B, d_cond) — pre-computed condition
        target_audio:  (B, T) — target audio to match
        init_params:   (B, N_PARAMS) — initial guess (from ODE sampling)
        n_steps:       Number of gradient steps
        lr:            Learning rate for parameter updates
        noise_scale:   Small noise injection to escape shallow local minima

    Returns:
        refined_params: (B, N_PARAMS)
        loss_history:   list of loss values
    """
    params = init_params.detach().clone()
    params.requires_grad_(True)

    # SGD with momentum works better than Adam for short-horizon optimization
    # because we don't need adaptive step sizes for <100 steps
    optimizer = torch.optim.SGD([params], lr=lr, momentum=0.9)

    loss_history = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # Convert flat vector to param dict for synth
        pred_dict = vector_to_params(params)
        pred_dict.pop("fx_routing", None)
        pred_audio = synth(pred_dict)

        # Multi-resolution STFT loss
        loss = multi_resolution_stft_loss(pred_audio, target_audio)

        if not torch.isfinite(loss):
            break

        loss.backward()
        optimizer.step()

        # Clamp parameters to valid range
        with torch.no_grad():
            params[:, :43].clamp_(0.001, 0.999)  # continuous
            # Categorical and routing are unconstrained (logit space)
            if step % 10 == 0:
                loss_history.append(loss.item())

        # Small noise injection every N steps (escaping shallow minima)
        if step > 0 and step % 15 == 0:
            with torch.no_grad():
                params.add_(torch.randn_like(params) * noise_scale)
                params[:, :43].clamp_(0.001, 0.999)

    loss_history.append(loss.item())
    return params.detach(), loss_history


# ── Best-of-N Sampling ──────────────────────────────────────────────────────


@torch.no_grad()
def sample_best_of_n(
    flow_net: nn.Module,
    synth: nn.Module,
    audio_cond: torch.Tensor,
    target_audio: torch.Tensor,
    n_candidates: int = 8,
    n_ode_steps: int = 20,
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """Generate N parameter candidates, pick the one with lowest synth loss.

    This leverages the flow network's ability to sample diverse candidates —
    the "distribution" nature of flow matching at work.

    Args:
        flow_net:     Flow matching network
        synth:        Differentiable synthesizer
        audio_cond:   (1, d_cond) — single audio condition
        target_audio: (1, T) — target to match
        n_candidates: Number of samples to draw
        n_ode_steps:  ODE integration steps per sample

    Returns:
        best_params: (1, N_PARAMS) — best candidate
        best_audio:  (1, T) — rendered audio from best params
        losses:      list of all candidate losses
    """
    best_params = None
    best_audio = None
    best_loss = float("inf")
    losses = []

    for _ in range(n_candidates):
        # Sample from flow
        params = sample_euler(flow_net, audio_cond, n_steps=n_ode_steps)

        # Render and compute loss
        param_dict = vector_to_params(params)
        param_dict.pop("fx_routing", None)
        audio = synth(param_dict)

        loss = multi_resolution_stft_loss(audio, target_audio).item()
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_params = params
            best_audio = audio

    return best_params, best_audio, losses
