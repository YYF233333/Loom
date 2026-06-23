"""Parameter tokenizer — bidirectional mapping between 97-dim param vector and 16 tokens.

Token layout (one per semantically distinct group):

    idx  name      continuous indices          categorical outputs        total
    ──────────────────────────────────────────────────────────────────────────
    0    osc       [0,1,2,3,4,5]  (6)         osc_waveform(4)+osc_type(3)  13
    1    filter    [10,11,12,13,14,15,16,17] (8)  filter_type(3)           11
    2    amp_env   [6,7,8,9]  (4)              —                           4
    3    dist      [18,19]  (2)                —                           2
    4    comp      [21,22,23,24,25,26]  (6)    —                           6
    5    chorus    [27,28,29]  (3)             —                           3
    6    delay     [30,31,32]  (3)             —                           3
    7    reverb    [33,34,35,36]  (4)          —                           4
    8    eq        [37,38,39]  (3)             —                           3
    9    global    [20,40,41,42]  (4)          lfo_waveform(4)+lfo_target(4)  12
    10-15 routing  one per 6-dim row of 6×6     —                           6×6=36

Total: 43 continuous + 18 categorical + 36 routing = 97 params → 16 tokens.

Each token projects its slice of the 97-dim vector to/from d_model.
Categoricals use soft one-hot weighting (straight-through at inference).
"""

import torch
import torch.nn as nn

from loom.training.dataset import (
    CONTINUOUS_KEYS,
    CATEGORICAL_KEYS,
    N_CONTINUOUS,
    N_CATEGORICAL,
    N_ROUTING,
    N_PARAMS,
)

# ── Token group definitions ────────────────────────────────────────────────
# (name, cont_indices, cat_specs, n_routing_logits)
# cat_specs: list of (global_category_index, key_name, n_classes)
#
# CONTINUOUS_KEYS layout (43 total):
#   0:osc_pitch  1:osc_detune  2:wt_position  3:fm_carrier_ratio
#   4:fm_mod_ratio  5:fm_mod_index
#   6:amp_attack  7:amp_decay  8:amp_sustain  9:amp_release
#   10:filter_cutoff  11:filter_q  12:filter_mix
#   13:filt_env_attack  14:filt_env_decay  15:filt_env_sustain
#   16:filt_env_release  17:filt_env_amount
#   18:dist_amount  19:dist_mix
#   20:master_gain
#   21:comp_threshold  22:comp_ratio  23:comp_attack  24:comp_release
#   25:comp_makeup  26:comp_mix
#   27:chorus_rate  28:chorus_depth  29:chorus_mix
#   30:delay_time  31:delay_feedback  32:delay_mix
#   33:reverb_room_size  34:reverb_decay  35:reverb_damping  36:reverb_mix
#   37:eq_low_gain  38:eq_mid_gain  39:eq_high_gain
#   40:lfo_rate  41:lfo_depth  42:lfo_phase
#
# CATEGORICAL_KEYS (5 groups, 18 dims total):
#   0: osc_waveform (4)    1: osc_type (3)    2: filter_type (3)
#   3: lfo_waveform (4)    4: lfo_target (4)

TOKEN_GROUPS = [
    # (name, cont_indices, cat_start_end_pairs, r_out)
    ("osc",    [0,1,2,3,4,5],                             [(0,4),(1,3)],  0),
    ("filter", [10,11,12,13,14,15,16,17],                 [(2,3)],        0),
    ("amp_env",[6,7,8,9],                                 [],             0),
    ("dist",   [18,19],                                   [],             0),
    ("comp",   [21,22,23,24,25,26],                       [],             0),
    ("chorus", [27,28,29],                                [],             0),
    ("delay",  [30,31,32],                                [],             0),
    ("reverb", [33,34,35,36],                             [],             0),
    ("eq",     [37,38,39],                                [],             0),
    ("global", [20,40,41,42],                             [(3,4),(4,4)],  0),
    ("route0", [],                                        [],             6),
    ("route1", [],                                        [],             6),
    ("route2", [],                                        [],             6),
    ("route3", [],                                        [],             6),
    ("route4", [],                                        [],             6),
    ("route5", [],                                        [],             6),
]

N_TOKENS = len(TOKEN_GROUPS)  # 16


def _build_param_to_token_map():
    """Build index maps for bidirectional param ↔ token conversion."""
    # For each token, record which parameter indices it owns
    token_cont = []   # list of (cont_start, cont_end) for each token
    token_cat = []    # list of (cat_start, cat_end) for each token
    token_route = []  # list of (route_start, route_end) for each token

    cont_offset = 0
    for name, cont_idxs, cat_specs, r_out in TOKEN_GROUPS:
        # Continuous: records absolute indices into the CONT param vector
        token_cont.append(sorted(cont_idxs))

        # Categorical: records (start, end) into the 18-dim cat block
        cat_parts = []
        for cat_idx, n_classes in cat_specs:
            # Find offset of this categorical within CATEGORICAL_KEYS
            start = 0
            for ci in range(cat_idx):
                start += CATEGORICAL_KEYS[ci][1]
            cat_parts.append((start, start + n_classes))
        token_cat.append(cat_parts)

        # Routing
        token_route.append((cont_offset, cont_offset + r_out))
        cont_offset += r_out

    return token_cont, token_cat, token_route


_TOKEN_CONT, _TOKEN_CAT, _TOKEN_ROUTE = _build_param_to_token_map()


class ParamTokenizer(nn.Module):
    """Project parameter vector ↔ token sequence.

    Forward (params → tokens):
        Continuous params → per-group Linear → token embeds
        Categorical params → soft one-hot mix → token embeds
        Routing → per-group Linear → token embeds
        → (B, N_TOKENS, d_model)

    Reverse (tokens → params):
        Per-token Linear head → per-group cont (sigmoid) + cat (softmax) + routing (raw)
        → assemble to (B, N_PARAMS)
    """

    def __init__(self, d_model: int = 256):
        super().__init__()
        self.d_model = d_model

        # ── Forward projections: param slices → token embeds ──
        self.cont_proj = nn.ModuleList()
        self.cat_proj = nn.ModuleList()
        self.route_proj = nn.ModuleList()

        for gi, (name, cont_idxs, cat_specs, r_out) in enumerate(TOKEN_GROUPS):
            # Continuous projection
            n_cont = len(cont_idxs)
            if n_cont > 0:
                self.cont_proj.append(nn.Linear(n_cont, d_model))
            else:
                self.cont_proj.append(nn.Identity())  # placeholder, not used

            # Categorical embeddings: one embedding per cat group
            cat_embs = nn.ModuleList()
            for cat_idx, n_classes in cat_specs:
                cat_embs.append(nn.Embedding(n_classes, d_model))
            self.cat_proj.append(cat_embs)

            # Routing projection
            if r_out > 0:
                self.route_proj.append(nn.Linear(r_out, d_model))
            else:
                self.route_proj.append(nn.Identity())

        # ── Reverse heads: token embeds → param slices ──
        self.cont_heads = nn.ModuleList()
        self.cat_heads = nn.ModuleList()
        self.route_heads = nn.ModuleList()

        for gi, (name, cont_idxs, cat_specs, r_out) in enumerate(TOKEN_GROUPS):
            n_cont = len(cont_idxs)
            if n_cont > 0:
                self.cont_heads.append(nn.Linear(d_model, n_cont))
            else:
                self.cont_heads.append(nn.Identity())

            cat_head_list = nn.ModuleList()
            for cat_idx, n_classes in cat_specs:
                cat_head_list.append(nn.Linear(d_model, n_classes))
            self.cat_heads.append(cat_head_list)

            if r_out > 0:
                self.route_heads.append(nn.Linear(d_model, r_out))
            else:
                self.route_heads.append(nn.Identity())

    def params_to_tokens(self, param_vec: torch.Tensor) -> torch.Tensor:
        """Convert (B, N_PARAMS) → (B, N_TOKENS, d_model).

        Continuous: extract slice → linear project
        Categorical: extract one-hot → weighted sum of learned embeddings
        Routing: extract slice → linear project
        """
        B = param_vec.shape[0]
        device = param_vec.device
        tokens = []

        # Split param vector into contiguous blocks
        cont_vec = param_vec[:, :N_CONTINUOUS]           # (B, 43)
        cat_vec  = param_vec[:, N_CONTINUOUS:N_CONTINUOUS + N_CATEGORICAL]  # (B, 18)
        route_vec = param_vec[:, N_CONTINUOUS + N_CATEGORICAL:]  # (B, 36)

        cat_offset = 0
        for gi in range(N_TOKENS):
            token_parts = []

            # Continuous contribution
            cont_idxs = _TOKEN_CONT[gi]
            if cont_idxs:
                idx_tensor = torch.tensor(cont_idxs, device=device, dtype=torch.long)
                cont_slice = cont_vec[:, idx_tensor]      # (B, n_cont)
                token_parts.append(self.cont_proj[gi](cont_slice))

            # Categorical contribution (soft embedding mix)
            cat_specs = _TOKEN_CAT[gi]
            if cat_specs:
                cat_contrib = torch.zeros(B, self.d_model, device=device)
                for spec_i, (start, end) in enumerate(cat_specs):
                    one_hot = cat_vec[:, start:end]        # (B, n_classes)
                    # Weighted sum of embeddings
                    n_cls = end - start
                    idx = torch.arange(n_cls, device=device)
                    all_embs = self.cat_proj[gi][spec_i](idx)  # (n_classes, d_model)
                    cat_contrib = cat_contrib + (one_hot @ all_embs)
                token_parts.append(cat_contrib)

            # Routing contribution
            r_start, r_end = _TOKEN_ROUTE[gi]
            if r_end > r_start:
                r_slice = route_vec[:, r_start:r_end]
                token_parts.append(self.route_proj[gi](r_slice))

            # Sum all contributions for this token
            token = sum(token_parts) if token_parts else torch.zeros(
                B, self.d_model, device=device,
            )
            tokens.append(token)

        return torch.stack(tokens, dim=1)  # (B, N_TOKENS, d_model)

    def tokens_to_velocity(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert (B, N_TOKENS, d_model) → (B, N_PARAMS) as RAW velocity.

        NO activation — velocity is unbounded real-valued (v = x_1 - x_0).
        This is the output used during flow matching training and ODE steps.
        """
        B = tokens.shape[0]
        device = tokens.device

        cont_parts = []
        cat_parts = []
        route_parts = []

        for gi in range(N_TOKENS):
            token = tokens[:, gi, :]  # (B, d_model)

            # Continuous: raw linear output (no sigmoid — velocity is unbounded)
            cont_idxs = _TOKEN_CONT[gi]
            if cont_idxs:
                cont_out = self.cont_heads[gi](token)  # NO sigmoid
                cont_parts.append((cont_idxs, cont_out))

            # Categorical: raw logits (no softmax)
            for head in self.cat_heads[gi]:
                cat_parts.append(head(token))  # NO softmax

            # Routing: raw logits
            r_start, r_end = _TOKEN_ROUTE[gi]
            if r_end > r_start:
                route_out = self.route_heads[gi](token)
                route_parts.append(route_out)

        # Assemble
        cont_out = torch.zeros(B, N_CONTINUOUS, device=device)
        for idxs, vals in cont_parts:
            idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
            cont_out[:, idx_tensor] = vals

        cat_out = torch.cat(cat_parts, dim=-1) if cat_parts else torch.zeros(
            B, 0, device=device,
        )
        route_out = torch.cat(route_parts, dim=-1) if route_parts else torch.zeros(
            B, 0, device=device,
        )

        return torch.cat([cont_out, cat_out, route_out], dim=-1)

    def tokens_to_params(self, tokens: torch.Tensor) -> torch.Tensor:
        """Convert (B, N_TOKENS, d_model) → (B, N_PARAMS) with constraints.

        Continuous: linear head → sigmoid → [0,1]
        Categorical: linear head → softmax → probability simplex
        Routing: linear head → raw logits (unconstrained)

        This is used ONLY at inference time to convert ODE output to valid params.
        During training, use tokens_to_velocity() instead.
        """
        B = tokens.shape[0]
        device = tokens.device

        cont_parts = []
        cat_parts = []
        route_parts = []

        for gi in range(N_TOKENS):
            token = tokens[:, gi, :]  # (B, d_model)

            # Continuous
            cont_idxs = _TOKEN_CONT[gi]
            if cont_idxs:
                cont_out = torch.sigmoid(self.cont_heads[gi](token)).float()
                cont_parts.append((cont_idxs, cont_out))

            # Categorical
            for head in self.cat_heads[gi]:
                cat_out = torch.softmax(head(token).float(), dim=-1)
                cat_parts.append(cat_out)

            # Routing
            r_start, r_end = _TOKEN_ROUTE[gi]
            if r_end > r_start:
                route_out = self.route_heads[gi](token).float()
                route_parts.append(route_out)

        # Assemble continuous vector in correct order
        cont_out = torch.zeros(B, N_CONTINUOUS, device=device)
        for idxs, vals in cont_parts:
            idx_tensor = torch.tensor(idxs, device=device, dtype=torch.long)
            cont_out[:, idx_tensor] = vals

        # Assemble categorical and routing
        cat_out = torch.cat(cat_parts, dim=-1) if cat_parts else torch.zeros(
            B, 0, device=device,
        )
        route_out = torch.cat(route_parts, dim=-1) if route_parts else torch.zeros(
            B, 0, device=device,
        )

        return torch.cat([cont_out, cat_out, route_out], dim=-1)

    def forward(self, param_vec: torch.Tensor) -> torch.Tensor:
        """Default forward: params → tokens (for use in flow network)."""
        return self.params_to_tokens(param_vec)
