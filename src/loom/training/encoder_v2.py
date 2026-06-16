"""encoder_v2.py — 2D CNN stem + DETR decoder + Transformer / Hybrid architectures.

Two encoder architectures are provided:

  TransformerParamEncoder  (~5M params)
    CNN2DStem → 6-layer Transformer encoder → 2-layer DETR decoder → GroupedParamHeads

  HybridParamEncoder  (~4M params)
    CNN2DStem → [Mamba×5, Attention, Mamba, Attention] → DETR decoder → GroupedParamHeads

Shared components:
  CNN2DStem            4-layer 2-D CNN, (B, 128, T) → (B, seq_len, d_model)
  LearnedPosEncoding2D freq + time embeddings
  DETRDecoder          learnable query tokens + TransformerDecoder cross-attention
  GroupedParamHeads    per-group sigmoid / softmax / raw-logit heads → (B, N_PARAMS)
"""

import math

import torch
import torch.nn as nn

from loom.training.dataset import (
    CATEGORICAL_KEYS,
    CONTINUOUS_KEYS,
    N_CONTINUOUS,
    N_PARAMS,
    N_ROUTING,
)
from loom.training.encoder import S4DLayer  # pure-PyTorch SSM fallback

try:
    from mamba_ssm import Mamba

    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False


# ── Parameter group index definitions ────────────────────────────────

# Indices into the CONTINUOUS_KEYS list (43 total):
# osc_pitch(0) osc_detune(1) wt_position(2)
# fm_carrier_ratio(3) fm_mod_ratio(4) fm_mod_index(5)
# amp_attack(6) amp_decay(7) amp_sustain(8) amp_release(9)
# filter_cutoff(10) filter_q(11) filter_mix(12)
# filt_env_attack(13) filt_env_decay(14) filt_env_sustain(15) filt_env_release(16) filt_env_amount(17)
# dist_amount(18) dist_mix(19) master_gain(20)
# comp_threshold(21) comp_ratio(22) comp_attack(23) comp_release(24) comp_makeup(25) comp_mix(26)
# chorus_rate(27) chorus_depth(28) chorus_mix(29)
# delay_time(30) delay_feedback(31) delay_mix(32)
# reverb_room_size(33) reverb_decay(34) reverb_damping(35) reverb_mix(36)
# eq_low_gain(37) eq_mid_gain(38) eq_high_gain(39)
# lfo_rate(40) lfo_depth(41) lfo_phase(42)

OSC_CONT = [0, 1, 2, 3, 4, 5]                       # pitch, detune, wt_pos, fm params
OSC_CAT = [("osc_waveform", 4), ("osc_type", 3)]     # categorical: osc_waveform, osc_type

FILTER_CONT = [10, 11, 12, 13, 14, 15, 16, 17]       # cutoff, q, mix, filt_env ×5
FILTER_CAT = [("filter_type", 3)]                    # categorical: filter_type

ENV_CONT = [6, 7, 8, 9]                              # amp ADSR

FX_CONT = [18, 19, 20, 21, 22, 23, 24, 25, 26,       # dist, master_gain, comp
           27, 28, 29, 30, 31, 32,                   # chorus, delay
           33, 34, 35, 36,                           # reverb
           37, 38, 39]                               # eq

GLOBAL_CONT = [40, 41, 42]                           # lfo rate, depth, phase
GLOBAL_CAT = [("lfo_waveform", 4), ("lfo_target", 4)]  # categorical: lfo_waveform, lfo_target

# Sanity-check: all continuous indices covered
assert sorted(OSC_CONT + FILTER_CONT + ENV_CONT + FX_CONT + GLOBAL_CONT) == list(
    range(N_CONTINUOUS)
), "PARAM_GROUPS do not cover all continuous indices"

# Each query token is responsible for one group:
#   0: OSC (cont + 2 cat)
#   1: FILTER (cont + 1 cat)
#   2: ENV (cont)
#   3: FX (cont)
#   4: GLOBAL (cont + 2 cat)
#   5: routing row 0  |  6: routing row 1  |  7: routing row 2
#   8: routing row 3  |  9: routing row 4  (row 5 also handled by token 9 via head)
N_QUERIES = 10

# PARAM_GROUPS: list of (name, cont_indices, cat_specs, n_routing_logits, loss_weight)
# cat_specs: list of (key_name, n_classes) — order must match CATEGORICAL_KEYS
PARAM_GROUPS = [
    ("osc",    OSC_CONT,    OSC_CAT,    0,  1.5),
    ("filter", FILTER_CONT, FILTER_CAT, 0,  1.2),
    ("env",    ENV_CONT,    [],         0,  1.0),
    ("fx",     FX_CONT,     [],         0,  0.8),
    ("global", GLOBAL_CONT, GLOBAL_CAT, 0,  0.6),
    ("route",  [],          [],         N_ROUTING, 0.5),
]


# ── CNN 2D Stem ───────────────────────────────────────────────────────


class CNN2DStem(nn.Module):
    """4-layer 2-D CNN that maps a mel spectrogram to a sequence of feature vectors.

    Input:  (batch, n_mels=128, T)  — no channel dim; treated as (B, 1, F, T)
    Output: (batch, seq_len, d_model)  where seq_len = T // 4 (two stride-2 layers)
    """

    def __init__(self, n_mels: int = 128, d_model: int = 256):
        super().__init__()
        # We treat the mel as a single-channel 2-D image: (B, 1, n_mels, T)
        self.conv = nn.Sequential(
            # Layer 1: freq-stride=2, time-stride=1 → (B, 32, n_mels//2, T)
            nn.Conv2d(1, 32, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            # Layer 2: freq-stride=2, time-stride=2 → (B, 64, n_mels//4, T//2)
            nn.Conv2d(32, 64, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # Layer 3: freq-stride=2, time-stride=1 → (B, 128, n_mels//8, T//2)
            nn.Conv2d(64, 128, kernel_size=3, stride=(2, 1), padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # Layer 4: freq-stride=2, time-stride=2 → (B, 256, n_mels//16, T//4)
            nn.Conv2d(128, 256, kernel_size=3, stride=(2, 2), padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )
        # After 4 layers with freq strides (2,2,2,2): freq dim = n_mels // 16 = 8
        freq_out = n_mels // 16
        self.proj = nn.Linear(256 * freq_out, d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: (B, F, T)
        x = mel.unsqueeze(1)          # (B, 1, F, T)
        x = self.conv(x)              # (B, C, F', T')
        B, C, Fp, Tp = x.shape
        x = x.permute(0, 3, 1, 2)    # (B, T', C, F')
        x = x.reshape(B, Tp, C * Fp) # (B, T', C*F')
        x = self.proj(x)              # (B, T', d_model)
        return x


# ── Learned 2D Positional Encoding ───────────────────────────────────


class LearnedPosEncoding2D(nn.Module):
    """Separate learned embeddings for frequency and time dimensions, summed.

    The module returns a (seq_len, d_model) additive positional tensor.
    seq_len = time steps after CNN stem.  We allocate up to max_time_len
    time positions and max_freq_bins frequency positions; both are truncated
    or zero-padded at forward time to match the actual tensor.
    """

    def __init__(self, d_model: int = 256, max_time_len: int = 512, max_freq_bins: int = 8):
        super().__init__()
        self.time_embed = nn.Embedding(max_time_len, d_model)
        self.freq_embed = nn.Embedding(max_freq_bins, d_model)
        self.max_time_len = max_time_len
        self.max_freq_bins = max_freq_bins

    def forward(self, seq_len: int) -> torch.Tensor:
        """Return (seq_len, d_model) positional encoding on the correct device."""
        device = self.time_embed.weight.device
        # Time positions
        time_ids = torch.arange(min(seq_len, self.max_time_len), device=device)
        time_pos = self.time_embed(time_ids)  # (seq_len, d_model)
        if seq_len > self.max_time_len:
            pad = torch.zeros(seq_len - self.max_time_len, time_pos.shape[1], device=device)
            time_pos = torch.cat([time_pos, pad], dim=0)

        # Frequency positions are pooled into a single scalar via mean, so we
        # add the mean freq embedding as a constant offset across all time steps.
        freq_ids = torch.arange(self.max_freq_bins, device=device)
        freq_pos = self.freq_embed(freq_ids).mean(dim=0, keepdim=True)  # (1, d_model)

        return time_pos + freq_pos  # (seq_len, d_model)


# ── DETR-style Decoder ────────────────────────────────────────────────


class DETRDecoder(nn.Module):
    """Learnable query tokens attending to encoder memory via TransformerDecoder.

    The decoder produces N_QUERIES output vectors that are used by
    GroupedParamHeads to read off parameter estimates.

    Args:
        n_queries:  number of learnable query tokens (= N_QUERIES = 10)
        d_model:    embedding dimension
        nhead:      attention heads
        n_layers:   number of decoder layers
        dropout:    dropout probability
    """

    def __init__(
        self,
        n_queries: int = N_QUERIES,
        d_model: int = 256,
        nhead: int = 8,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.query_embed = nn.Embedding(n_queries, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.n_queries = n_queries

    def forward(self, memory: torch.Tensor) -> torch.Tensor:
        """
        Args:
            memory: (batch, seq_len, d_model) — encoder output
        Returns:
            queries: (batch, n_queries, d_model)
        """
        B = memory.shape[0]
        # Expand query embeddings to batch
        tgt = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)  # (B, Q, d_model)
        out = self.decoder(tgt, memory)   # (B, Q, d_model)
        return self.norm(out)


# ── Grouped Parameter Heads ───────────────────────────────────────────


class GroupedParamHeads(nn.Module):
    """Per-group linear heads that read from DETR query tokens.

    Each group reads from one or more query tokens and produces:
      - continuous params → sigmoid
      - categorical params → softmax per category
      - routing logits → raw (no activation)

    The output is assembled in the standard (N_CONTINUOUS | N_CATEGORICAL | N_ROUTING) order.

    Args:
        d_model: embedding dimension
    """

    def __init__(self, d_model: int = 256):
        super().__init__()

        # --- continuous heads ---
        # One linear per group, reading the first query token assigned to that group.
        # Query token assignment:  osc=0, filter=1, env=2, fx=3, global=4, route=5..9
        self.osc_head    = nn.Linear(d_model, len(OSC_CONT))
        self.filter_head = nn.Linear(d_model, len(FILTER_CONT))
        self.env_head    = nn.Linear(d_model, len(ENV_CONT))
        self.fx_head     = nn.Linear(d_model, len(FX_CONT))
        self.global_head = nn.Linear(d_model, len(GLOBAL_CONT))

        # --- categorical heads ---
        # osc_waveform (4), osc_type (3)
        self.osc_waveform_head = nn.Linear(d_model, 4)
        self.osc_type_head     = nn.Linear(d_model, 3)
        # filter_type (3)
        self.filter_type_head  = nn.Linear(d_model, 3)
        # lfo_waveform (4), lfo_target (4)
        self.lfo_waveform_head = nn.Linear(d_model, 4)
        self.lfo_target_head   = nn.Linear(d_model, 4)

        # --- routing head: reads query tokens 5-9, outputs N_ROUTING=36 logits ---
        # Each of the 5 routing tokens reads a 6-dim row (6×5=30), plus token 9 does row 5 too
        # Simpler: a single linear from the concatenation of query tokens 5-9
        self.routing_head = nn.Linear(d_model * 5, N_ROUTING)

        # Store loss weights for param_groups_for_optimizer
        self.group_loss_weights = {g[0]: g[4] for g in PARAM_GROUPS}

    def forward(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Args:
            queries: (batch, N_QUERIES, d_model)
        Returns:
            params: (batch, N_PARAMS)
        """
        q_osc    = queries[:, 0, :]   # (B, d_model)
        q_filter = queries[:, 1, :]
        q_env    = queries[:, 2, :]
        q_fx     = queries[:, 3, :]
        q_global = queries[:, 4, :]
        q_route  = queries[:, 5:, :]  # (B, 5, d_model)

        # --- continuous outputs (assembled in CONTINUOUS_KEYS order) ---
        # We need to scatter each group's continuous outputs into a (B, N_CONTINUOUS) tensor
        B = queries.shape[0]
        cont = torch.zeros(B, N_CONTINUOUS, device=queries.device, dtype=queries.dtype)

        cont[:, OSC_CONT]    = torch.sigmoid(self.osc_head(q_osc))
        cont[:, FILTER_CONT] = torch.sigmoid(self.filter_head(q_filter))
        cont[:, ENV_CONT]    = torch.sigmoid(self.env_head(q_env))
        cont[:, FX_CONT]     = torch.sigmoid(self.fx_head(q_fx))
        cont[:, GLOBAL_CONT] = torch.sigmoid(self.global_head(q_global))

        # --- categorical outputs (in CATEGORICAL_KEYS order) ---
        # CATEGORICAL_KEYS = [osc_waveform(4), osc_type(3), filter_type(3), lfo_waveform(4), lfo_target(4)]
        cat_osc_waveform = torch.softmax(self.osc_waveform_head(q_osc), dim=-1)    # (B, 4)
        cat_osc_type     = torch.softmax(self.osc_type_head(q_osc), dim=-1)        # (B, 3)
        cat_filter_type  = torch.softmax(self.filter_type_head(q_filter), dim=-1)  # (B, 3)
        cat_lfo_waveform = torch.softmax(self.lfo_waveform_head(q_global), dim=-1) # (B, 4)
        cat_lfo_target   = torch.softmax(self.lfo_target_head(q_global), dim=-1)   # (B, 4)

        cats = torch.cat(
            [cat_osc_waveform, cat_osc_type, cat_filter_type, cat_lfo_waveform, cat_lfo_target],
            dim=-1,
        )  # (B, 18)

        # --- routing logits ---
        route_in = q_route.reshape(B, -1)           # (B, 5*d_model)
        routing  = self.routing_head(route_in)       # (B, 36)

        return torch.cat([cont, cats, routing], dim=-1)  # (B, N_PARAMS)


# ── Attention Block ───────────────────────────────────────────────────


class AttentionBlock(nn.Module):
    """Standard multi-head self-attention block with pre-LayerNorm (Pre-LN)."""

    def __init__(self, d_model: int = 256, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with pre-norm
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ── Mamba Block (with S4D fallback) ──────────────────────────────────


class MambaBlock(nn.Module):
    """Mamba selective-scan block, falls back to S4DLayer when mamba-ssm is unavailable.

    Input/output: (batch, seq_len, d_model)  — batch_first convention.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        if HAS_MAMBA:
            self.ssm = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        else:
            self.ssm = S4DLayer(d_model, d_state)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self._has_mamba = HAS_MAMBA

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._has_mamba:
            # Mamba expects (B, L, D), which is already our convention
            x = x + self.ssm(self.norm1(x))
        else:
            # S4DLayer expects (B, D, L)
            h = self.norm1(x).transpose(1, 2)   # (B, D, L)
            h = self.ssm(h).transpose(1, 2)     # (B, L, D)
            x = x + h
        x = x + self.ff(self.norm2(x))
        return x


# ── Transformer Param Encoder ─────────────────────────────────────────


class TransformerParamEncoder(nn.Module):
    """Architecture A: 2D CNN stem → 6-layer Transformer encoder → DETR decoder → grouped heads.

    Args:
        n_mels:    mel bins (must match dataset, default 128)
        d_model:   embedding dimension (default 256)
        nhead:     attention heads (default 8)
        n_layers:  Transformer encoder layers (default 6)
        dropout:   dropout probability
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        n_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem     = CNN2DStem(n_mels, d_model)
        self.pos_enc  = LearnedPosEncoding2D(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.decoder  = DETRDecoder(N_QUERIES, d_model, nhead, n_layers=2, dropout=dropout)
        self.heads    = GroupedParamHeads(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, T)
        Returns:
            params: (batch, N_PARAMS)
        """
        x = self.stem(mel)                         # (B, seq_len, d_model)
        pos = self.pos_enc(x.shape[1])             # (seq_len, d_model)
        x = x + pos.unsqueeze(0)                   # broadcast over batch
        memory = self.encoder(x)                   # (B, seq_len, d_model)
        queries = self.decoder(memory)             # (B, N_QUERIES, d_model)
        return self.heads(queries)                 # (B, N_PARAMS)

    def param_groups_for_optimizer(self, lr: float) -> list[dict]:
        """Return parameter groups with per-group learning rates for Adam.

        Groups: stem (lr), encoder (lr), decoder (lr * 0.5), heads (lr).
        """
        return [
            {"params": list(self.stem.parameters()),    "lr": lr,       "name": "stem"},
            {"params": list(self.pos_enc.parameters()), "lr": lr * 0.1, "name": "pos_enc"},
            {"params": list(self.encoder.parameters()), "lr": lr,       "name": "encoder"},
            {"params": list(self.decoder.parameters()), "lr": lr * 0.5, "name": "decoder"},
            {"params": list(self.heads.parameters()),   "lr": lr,       "name": "heads"},
        ]


# ── Hybrid Param Encoder ──────────────────────────────────────────────


class HybridParamEncoder(nn.Module):
    """Architecture B: 2D CNN stem → [Mamba×5, Attention, Mamba, Attention] → DETR decoder → heads.

    This architecture follows the 7:1 Mamba-to-Attention ratio described in the research plan.
    SSMs handle long-range sequence context efficiently; attention layers provide global mixing.

    Args:
        n_mels:   mel bins (default 128)
        d_model:  embedding dimension (default 256)
        nhead:    attention heads (default 8)
        d_state:  SSM state dimension (default 64)
        dropout:  dropout probability
    """

    def __init__(
        self,
        n_mels: int = 128,
        d_model: int = 256,
        nhead: int = 8,
        d_state: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stem    = CNN2DStem(n_mels, d_model)
        self.pos_enc = LearnedPosEncoding2D(d_model)

        # Backbone: Mamba×5, Attention, Mamba, Attention  (8 blocks, 6 Mamba + 2 Attention)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout),   # 0
            MambaBlock(d_model, d_state, dropout=dropout),   # 1
            MambaBlock(d_model, d_state, dropout=dropout),   # 2
            MambaBlock(d_model, d_state, dropout=dropout),   # 3
            MambaBlock(d_model, d_state, dropout=dropout),   # 4
            AttentionBlock(d_model, nhead, dropout=dropout), # 5
            MambaBlock(d_model, d_state, dropout=dropout),   # 6
            AttentionBlock(d_model, nhead, dropout=dropout), # 7
        ])
        self.norm    = nn.LayerNorm(d_model)
        self.decoder = DETRDecoder(N_QUERIES, d_model, nhead, n_layers=2, dropout=dropout)
        self.heads   = GroupedParamHeads(d_model)

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel: (batch, n_mels, T)
        Returns:
            params: (batch, N_PARAMS)
        """
        x = self.stem(mel)                     # (B, seq_len, d_model)
        pos = self.pos_enc(x.shape[1])         # (seq_len, d_model)
        x = x + pos.unsqueeze(0)
        for block in self.blocks:
            x = block(x)
        memory = self.norm(x)                  # (B, seq_len, d_model)
        queries = self.decoder(memory)         # (B, N_QUERIES, d_model)
        return self.heads(queries)             # (B, N_PARAMS)

    def param_groups_for_optimizer(self, lr: float) -> list[dict]:
        """Return parameter groups with per-group learning rates for Adam."""
        mamba_params, attn_params = [], []
        for i, block in enumerate(self.blocks):
            if isinstance(block, MambaBlock):
                mamba_params.extend(block.parameters())
            else:
                attn_params.extend(block.parameters())
        return [
            {"params": list(self.stem.parameters()),    "lr": lr,       "name": "stem"},
            {"params": list(self.pos_enc.parameters()), "lr": lr * 0.1, "name": "pos_enc"},
            {"params": mamba_params,                    "lr": lr,       "name": "mamba_blocks"},
            {"params": attn_params,                     "lr": lr * 0.8, "name": "attn_blocks"},
            {"params": list(self.decoder.parameters()), "lr": lr * 0.5, "name": "decoder"},
            {"params": list(self.heads.parameters()),   "lr": lr,       "name": "heads"},
        ]
