# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Loom is a research project on **AI audio understanding** — specifically, reverse-engineering finished audio back into engineering parameters (synth settings, effects chains, mix decisions). The goal is not audio generation or classification, but reconstructing the *production process* from the output.

This project pivoted from a REAPER MCP-based DAW assistant (experimentally disproven — Claude can execute DAW scripts but cannot judge audio quality). The current direction is building a differentiable synthesis engine and training models to do parameter estimation from audio.

## Research Context

The full research plan, literature survey, and architectural decisions are in `plan.md`. Key points:

- **Core problem**: audio → engineering parameters is an inverse problem with no existing solution at the full-project scale. SynthRL (IJCAI 2025) proved RL-based reversal works for single synthesizers.
- **Architecture decision**: Mamba-dominant + sparse Attention (7:1 ratio). SSMs handle long audio sequences efficiently; attention provides global context.
- **Training strategy**: differentiable gradients for own engine (signal-chain loss + Wasserstein loss), RL reserved for black-box VST integration later. Scaling law (SODA, ICLR 2026): bias toward data over parameters.
- **Reward design**: mel-spectrogram L1/L2 (low-level acoustic) + MERT embedding distance (high-level semantic).
- **Compute**: H100 available, models up to ~4B trainable. 1-person team.

## Planned Architecture

### Differentiable Synthesis Engine (Phase 0 — "headless DAW")

The engine will be pure Python/PyTorch, all operations differentiable for gradient-based training:

- **Synth modules**: subtractive (~35% of EDM sounds), wavetable (~20%), FM (~10%), sampler (~30%)
- **Effects chain** (7-8 types): distortion, LP/HP/BP filter, compressor/OTT, reverb, delay, chorus, EQ
- **Sequencer + multi-track mixer**
- **Sample library**: ~5K-10K curated samples with MERT/CLAP embedding index for retrieval

The model routes audio through a classifier (synthesized vs. sampled), then either estimates synth parameters or does embedding-based sample retrieval + transform estimation.

### Key Libraries

| Library | Role |
|---------|------|
| torchsynth | GPU-accelerated synth, 16200x realtime, synth1B1 dataset |
| DawDreamer | JUCE wrapper, loads VST3, Faust→JAX differentiable |
| Pedalboard (Spotify) | Effects processing, VST3 hosting, 300x faster than pySoX |
| SemantiCodec / X-Codec | Semantic audio tokenization (low-bitrate, meaning-preserving) |
| MERT / MuQ | Audio feature extraction / music understanding |

### Phased Roadmap

| Phase | What | Go/No-go |
|-------|------|----------|
| 0 | Differentiable synth engine | |
| 0.5 | Sample library + embedding index | |
| 1 | Audio tokenizer integration | |
| 2a | Self-supervised encoder pretraining (~100M) | |
| 2b | Single-synth parameter reversal | **Decision point** |
| 3 | Curriculum learning: multi-osc → FX → multi-track (~1B) | |
| 4 | Language alignment (music terminology) | |
| 5 | Real audio + RLHF aesthetics | |

## Environment

- **OS**: Windows 10, development on `F:\Code\loom`
- **DAW** (legacy, may still be used for validation): REAPER v7.74 with xDarkzx/Reaper-MCP
- **VST**: Serum 1.35b1, OTT, Sylenth1
- **Audio tools already validated**: Demucs (htdemucs 4-stem), scipy/numpy spectral analysis, ffmpeg
- **MCP**: `reaper-mcp` configured at user level (stdio server)

## Domain Knowledge

- Target genre: DnB (170-176 BPM) — Liquid, Neuro, Artcore
- Reese bass = detuned saw waves + filter (subtractive, not FM) — present in nearly every DnB track
- Neuro bass = simple waveform + heavy distortion/filter chain — effects contribute 60-70% of final timbre
- FM synthesis is the academic hard case (frequency coupling doesn't converge) but only ~10% of real EDM sounds
- Wavetable synthesis is cheap to make differentiable (`grid_sample` + interpolation, per DWTS ICASSP 2022)
