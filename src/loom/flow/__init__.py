"""Loom Flow Matching — conditional generative model for synth parameter estimation.

Replaces end-to-end regression with a flow matching approach that learns
the conditional distribution p(params | audio) instead of a point estimate.
This naturally handles parameter symmetries (multiple different parameter
sets producing the same sound).

Modules:
    tokenizer:     Parameter ↔ per-group token bijection
    frontend:      Audio frontends (Mel, Gammatone, Multi-Resolution)
    conditioner:   Audio → condition vector (CNN2D → Hybrid → Pool)
    dit:           DiT backbone with AdaLN, cross-attention to audio
    flow_matching: Rectified flow loss, ODE sampling, gradient refinement
"""
