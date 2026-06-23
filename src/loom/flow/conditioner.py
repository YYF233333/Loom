"""Audio conditioner — spectrogram → condition vector + audio latent tokens.

Architecture (revised 2026-06-23):
    1. Frontend:  CQT (168 bins, 24 bins/oct, C2–C9)
    2. FreqEncoder: Conv1d over frequency axis — NO pooling/stride
    3. TemporalBackbone: 4-layer Transformer Encoder (Pre-LN)
    4. MultiQueryPool: 4 learnable queries × Cross-Attention
         → 512d condition vector (L2-normed)
         → 4 audio latent tokens (passed to DiT for fine-grained interaction)

Key insight: never stride/pool the frequency axis. CQT's 168 bins stay 168
all the way through the FreqEncoder. The Transformer sees every frequency bin
at every time frame. This preserves bass fundamental (~65 Hz → ~4 CQT bins)
unlike the old CNN2DStem which compressed 128→8 bins (bass → 0.5 bins).
"""

import torch
import torch.nn as nn

from loom.flow.frontend import build_frontend


# ── Frequency Encoder ───────────────────────────────────────────────────────


class FreqEncoder(nn.Module):
    """Per-frame frequency encoder — Conv1d along freq axis, NO pooling.

    Input:  (B, n_bins, T)
    Output: (B, T, d_model)

    Each time frame is processed independently through 1D convolutions
    along the frequency axis. A kernel_size=7 at 24 bins/oct covers ~2.3 octaves
    — enough to catch harmonic spacing (octave, fifth) and formant structure.
    """

    def __init__(self, n_bins: int = 168, d_model: int = 256):
        super().__init__()
        # Layer 1: (B, n_bins, T) → (B, d_model, T)  — project each freq bin
        self.conv1 = nn.Conv1d(n_bins, d_model, kernel_size=7, padding=3)
        # Layer 2: (B, d_model, T) → (B, d_model, T)  — refine, same shape
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        # spec: (B, n_bins, T)
        x = self.conv1(spec)          # (B, d_model, T)
        x = nn.functional.gelu(x)
        x = self.conv2(x)             # (B, d_model, T)
        x = nn.functional.gelu(x)
        x = x.transpose(1, 2)         # (B, T, d_model)
        return x


# ── Temporal Backbone ───────────────────────────────────────────────────────


class TemporalTransformer(nn.Module):
    """Lightweight Transformer encoder over time frames.

    At T=87–173 (0.5–1s audio), self-attention is cheap (87² ≈ 7.6K pairs).
    Pre-LN with 4 layers provides global temporal context for:
      - ADSR envelope shape (attack/release timing)
      - LFO periodicity
      - Note onset/offset detection
    """

    def __init__(self, d_model: int = 256, nhead: int = 8, n_layers: int = 4, dropout: float = 0.1):
        super().__init__()
        self.pos_enc = nn.Parameter(torch.randn(1, 1024, d_model) * 0.02)  # max 1024 frames
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x (B, T, d_model) → (B, T, d_model)."""
        T = x.shape[1]
        x = x + self.pos_enc[:, :T, :]
        x = self.encoder(x)
        x = self.norm(x)
        return x


# ── Multi-Query Cross-Attention Pool ────────────────────────────────────────


class MultiQueryPool(nn.Module):
    """N_QUERIES learnable tokens cross-attend into the temporal sequence.

    Each query can specialise on different temporal/frequency patterns:
      - Query 0: onset / attack timing
      - Query 1: sustain / harmonic structure
      - Query 2: release timing
      - Query 3: modulation / LFO patterns

    Outputs:
        cond_vec: (B, d_cond) — compressed condition for AdaLN
        audio_latents: (B, N_QUERIES, d_model) — fine-grained tokens for DiT
    """

    def __init__(
        self, d_model: int = 256, d_cond: int = 512, n_queries: int = 4,
        nhead: int = 8, dropout: float = 0.1,
    ):
        super().__init__()
        self.n_queries = n_queries
        self.query_embed = nn.Parameter(torch.randn(1, n_queries, d_model) * 0.02)
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)

        # Project concatenated queries → condition vector
        self.cond_proj = nn.Sequential(
            nn.Linear(n_queries * d_model, d_cond),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_cond, d_cond),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, T, d_model) → cond_vec (B, d_cond), audio_latents (B, N_QUERIES, d_model)."""
        B = x.shape[0]
        queries = self.query_embed.expand(B, -1, -1)      # (B, N_QUERIES, d_model)
        attn_out, _ = self.cross_attn(queries, x, x)       # (B, N_QUERIES, d_model)
        attn_out = self.norm(attn_out)

        # Audio latent tokens: raw normalized query outputs
        audio_latents = attn_out                           # (B, N_QUERIES, d_model)

        # Condition vector: flattened + projected → L2-normed
        cond_vec = self.cond_proj(attn_out.reshape(B, -1))  # (B, d_cond)
        cond_vec = nn.functional.normalize(cond_vec, dim=-1)

        return cond_vec, audio_latents


# ── Conditioner ─────────────────────────────────────────────────────────────


class Conditioner(nn.Module):
    """Full audio conditioner.

    Pipeline:
        audio → CQTFrontend → FreqEncoder → TemporalTransformer → MultiQueryPool
                                                                  ├→ cond_vec (512d, L2-normed)
                                                                  └→ audio_latents (4 tokens × 256d)

    Args:
        frontend:   "cqt", "gammatone", "mel", or "multi"
        n_bins:     frequency bins (168 default for CQT)
        d_model:    internal dimension (256)
        d_cond:     output condition vector dimension (512)
        n_layers:   Transformer encoder layers (4)
        n_queries:  number of learnable query tokens (4)
        nhead:      attention heads (8)
        dropout:    dropout rate
    """

    def __init__(
        self,
        frontend: str = "cqt",
        n_bins: int = 168,
        d_model: int = 256,
        d_cond: int = 512,
        n_layers: int = 4,
        n_queries: int = 4,
        nhead: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_cond = d_cond
        self.n_queries = n_queries

        # 1. Audio frontend
        self.frontend = build_frontend(frontend, n_bins=n_bins)

        # Multi-res stacks 3 CQTs → 3× frequency channels
        n_bins_total = n_bins * 3 if frontend == "multires" else n_bins

        # 2. Frequency encoder (no pooling!)
        self.freq_enc = FreqEncoder(n_bins=n_bins_total, d_model=d_model)

        # 3. Temporal transformer
        self.temporal = TemporalTransformer(d_model=d_model, nhead=nhead, n_layers=n_layers, dropout=dropout)

        # 4. Multi-query pool with audio latent output
        self.pool = MultiQueryPool(d_model=d_model, d_cond=d_cond, n_queries=n_queries, nhead=nhead, dropout=dropout)

    def forward(self, audio: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """audio (B, T_raw) → cond_vec (B, d_cond), audio_latents (B, N_QUERIES, d_model)."""
        spec = self.frontend(audio)               # (B, n_bins, T_frames)
        x = self.freq_enc(spec)                   # (B, T_frames, d_model)
        x = self.temporal(x)                      # (B, T_frames, d_model)
        cond_vec, audio_latents = self.pool(x)    # (B, d_cond), (B, N_QUERIES, d_model)
        return cond_vec, audio_latents


def build_conditioner(
    frontend: str = "cqt",
    d_model: int = 256,
    d_cond: int = 512,
    **kwargs,
) -> Conditioner:
    """Factory with sensible defaults."""
    return Conditioner(
        frontend=frontend,
        d_model=d_model,
        d_cond=d_cond,
        **kwargs,
    )
