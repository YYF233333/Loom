"""DiT (Diffusion Transformer) backbone for flow matching.

Adapted from Peebles & Xie (ICCV 2023) with AdaLN-Zero conditioning.
Operates on 16 parameter-group tokens with audio + timestep conditioning.

Architecture:
    x_t (B, 16, d_model)  ← tokenized params at flow time t
    t_embed (B, d_model)   ← sinusoidal + MLP
    audio_cond (B, d_cond) ← from Conditioner

    For each DiTBlock:
        AdaLN(x) → Self-Attention → AdaLN(x) → FFN
        where AdaLN scale/shift/gate = MLP(t_embed + audio_proj)

    Output head: per-token Linear → per-group param slices

Key design choices:
    - AdaLN-Zero: zero-init final projection in each block for identity at t=0
    - 16 tokens (one per semantic group): osc, filter, env, dist, comp, chorus,
      delay, reverb, eq, global, route0-5
    - SwiGLU FFN (better than GELU for transformers, per Shazeer 2020)
"""

import math

import torch
import torch.nn as nn

from loom.flow.tokenizer import ParamTokenizer, N_TOKENS


# ── Sinusoidal Time Embedding ───────────────────────────────────────────────


class TimestepEmbedding(nn.Module):
    """Sinusoidal timestep embedding → MLP.

    Uses the standard transformer sinusoidal encoding followed by a 2-layer MLP
    with SiLU activation, following the DiT paper.
    """

    def __init__(self, d_model: int = 256, max_period: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_period = max_period
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.SiLU(),
            nn.Linear(d_model * 4, d_model),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t (B,) or (B, 1) → (B, d_model)."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half = self.d_model // 2
        freq = torch.exp(
            -math.log(self.max_period)
            * torch.arange(0, half, dtype=torch.float32, device=t.device)
            / half
        )
        args = t.float() * freq.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.d_model % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return self.mlp(emb)


# ── AdaLN-Zero ──────────────────────────────────────────────────────────────


class AdaLNZeo(nn.Module):
    """Adaptive Layer Normalization with zero-initialized final projection.

    Given a condition vector c, produces (shift, scale, gate) for
    pre-norm residual modulation.

    The gate projection is zero-initialized so that each block starts
    as an identity function — critical for training stability with flows.
    """

    def __init__(self, d_model: int = 256, d_cond: int = 512):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.proj = nn.Linear(d_cond, d_model * 3)
        # Zero-init: output is all zeros at start of training
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """x (B, N, d_model), cond (B, d_cond) → modulated x (B, N, d_model)."""
        shift, scale, gate = self.proj(cond).chunk(3, dim=-1)
        # shape: (B, d_model) each, broadcast to (B, N, d_model)
        x = self.norm(x)
        x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return x * gate.unsqueeze(1).sigmoid()


# ── SwiGLU FFN ──────────────────────────────────────────────────────────────


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network (Shazeer 2020).

    FFN(x) = W2 · (SiLU(W1·x) ⊙ W_g·x)
    """

    def __init__(self, d_model: int = 256, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = int(d_model * expansion * 2 / 3)  # match param count of standard 4×
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w_g = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(nn.functional.silu(self.w1(x)) * self.w_g(x)))


# ── DiT Block ───────────────────────────────────────────────────────────────


class DiTBlock(nn.Module):
    """DiT block with cross-attention to audio latent tokens.

    Sequence: AdaLN → Self-Attn → AdaLN → Cross-Attn(audio) → AdaLN → FFN

    The cross-attention lets each parameter token explicitly query audio features
    for the information it needs (e.g., the osc token queries for pitch).
    This is MUCH stronger conditioning than AdaLN alone.
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, d_cond: int = 512, dropout: float = 0.1):
        super().__init__()
        self.ada_sa = AdaLNZeo(d_model, d_cond)
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ada_ca = AdaLNZeo(d_model, d_cond)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ada_ffn = AdaLNZeo(d_model, d_cond)
        self.ffn = SwiGLU(d_model, dropout=dropout)

    def forward(self, x: torch.Tensor, audio_latents: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # 1. Self-attention among all tokens
        h = self.ada_sa(x, cond)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # 2. Cross-attention: each token queries audio latents
        if audio_latents is not None:
            h = self.ada_ca(x, cond)
            h, _ = self.cross_attn(h, audio_latents, audio_latents)
            x = x + h

        # 3. FFN
        h = self.ada_ffn(x, cond)
        h = self.ffn(h)
        x = x + h

        return x


# ── Full DiT Backbone ───────────────────────────────────────────────────────


class DiTBackbone(nn.Module):
    """DiT backbone with cross-attention to audio latent tokens.

    Input:  param_tokens (B, N_TOKENS, d_model) — parameter tokens
            t (B,) — flow timestamp
            audio_cond (B, d_cond) — compressed condition for AdaLN
            audio_latents (B, N_AUDIO, d_model) — audio tokens for cross-attn

    Output: v (B, N_TOKENS, d_model) — predicted velocity tokens
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        n_blocks: int = 6,
        d_cond: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.time_embed = TimestepEmbedding(d_model)
        self.audio_proj = nn.Sequential(
            nn.Linear(d_cond, d_model * 2),
            nn.SiLU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.cond_proj = nn.Linear(d_model * 2, d_model)  # project concat(audio) back
        self.blocks = nn.ModuleList([
            DiTBlock(d_model, nhead, d_model, dropout)
            for _ in range(n_blocks)
        ])
        self.norm_out = AdaLNZeo(d_model, d_model)

    def forward(
        self,
        param_tokens: torch.Tensor,
        t: torch.Tensor,
        audio_cond: torch.Tensor,
        audio_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """param_tokens (B, N_TOKENS, d_model) → v_tokens (B, N_TOKENS, d_model).

        Audio condition is directly CONCATENATED to each token (not just AdaLN).
        This forces the DiT to use the audio information — it's in the token
        representation from the start, not just modulating normalization stats.
        """
        t_emb = self.time_embed(t)
        a_emb = self.audio_proj(audio_cond)  # (B, d_model)
        cond = t_emb + a_emb                   # (B, d_model) — still used for AdaLN

        # DIRECT conditioning: append audio embedding to each token
        a_expand = a_emb.unsqueeze(1).expand(-1, param_tokens.shape[1], -1)
        x = torch.cat([param_tokens, a_expand], dim=-1)  # (B, N, 2*d_model)
        # Project back to d_model
        x = self.cond_proj(x)                             # (B, N, d_model)

        for block in self.blocks:
            x = block(x, audio_latents, cond)

        x = self.norm_out(x, cond)
        return x


# ── Complete Flow Network (tokenizer + DiT + detokenizer) ───────────────────


class FlowNetwork(nn.Module):
    """Complete flow matching network: params ↔ tokens ↔ DiT ↔ velocity.

    Training:
        x_t = (1-t)·noise + t·true_params     # linear rectified path
        tokens = tokenizer.params_to_tokens(x_t)
        v_tokens = dit(tokens, t, audio_cond)
        v = tokenizer.tokens_to_params(v_tokens)
        loss = Huber(v, true_params - noise)   # constant velocity target

    Inference (Euler ODE):
        x = randn(B, 97)
        for t in linspace(0, 1, steps):
            v = forward(x, t, audio_cond)
            x = x + v * dt
        return x  # sampled params
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        n_dit_blocks: int = 6,
        d_cond: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tokenizer = ParamTokenizer(d_model)
        self.dit = DiTBackbone(
            d_model=d_model,
            nhead=nhead,
            n_blocks=n_dit_blocks,
            d_cond=d_cond,
            dropout=dropout,
        )

    def forward(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        audio_cond: torch.Tensor,
        audio_latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict velocity field at (x_t, t, audio_cond, audio_latents).

        Audio latents go through cross-attention in each DiT block,
        NOT concatenated to param tokens. This gives stronger conditioning.
        """
        param_tokens = self.tokenizer.params_to_tokens(x_t)     # (B, 16, d_model)
        v_tokens = self.dit(param_tokens, t, audio_cond, audio_latents)
        return self.tokenizer.tokens_to_velocity(v_tokens)

    def compute_loss(
        self,
        params_true: torch.Tensor,
        audio_cond: torch.Tensor,
        audio_latents: torch.Tensor | None = None,
        stage: int = 99,
    ) -> torch.Tensor:
        """Simulation-free flow matching with IN-DISTRIBUTION noise.

        Instead of x_0 ~ N(0,I) (which has 10× more variance than actual params),
        we shuffle the batch so x_0 IS another valid parameter vector.
        This makes the audio condition matter — velocity variance comes from
        actual parameter variation, not random noise.
        """
        from loom.flow.stage_mask import get_stage_mask
        B = params_true.shape[0]
        device = params_true.device

        # In-distribution noise: shuffle the batch
        shuffle_idx = torch.randperm(B, device=device)
        x_0 = params_true[shuffle_idx]                    # valid params!
        x_1 = params_true
        t = torch.rand(B, device=device)
        t_exp = t.unsqueeze(-1)
        x_t = (1 - t_exp) * x_0 + t_exp * x_1
        v_true = x_1 - x_0
        v_pred = self.forward(x_t, t, audio_cond, audio_latents)

        mask = get_stage_mask(stage).to(device).unsqueeze(0)
        v_pred_masked = v_pred * mask
        v_true_masked = v_true * mask
        return nn.functional.smooth_l1_loss(v_pred_masked, v_true_masked)

    @torch.no_grad()
    def sample_params(
        self,
        audio_cond: torch.Tensor,
        audio_latents: torch.Tensor | None = None,
        n_steps: int = 20,
    ) -> torch.Tensor:
        """Euler ODE sampling → valid parameter vector."""
        B = audio_cond.shape[0]
        device = audio_cond.device
        x = torch.randn(B, 97, device=device)
        dt = 1.0 / n_steps
        for step in range(n_steps):
            t = torch.full((B,), step * dt, device=device)
            x = x + self.forward(x, t, audio_cond, audio_latents) * dt
        # Final step: convert to valid params
        tokens = self.tokenizer.params_to_tokens(x)
        return self.tokenizer.tokens_to_params(tokens)
