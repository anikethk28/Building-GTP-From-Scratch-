# GPT-2 Transformer — From-Scratch Implementation

A complete walkthrough of loading, analysing, hand-building, and training a GPT-2 style language model using PyTorch & EasyTransformer.



---

## Overview

This project implements GPT-2 Small (117M parameters) from the ground up every matrix multiplication, normalisation step, attention score, and residual connection is coded by hand in pure PyTorch and validated against the pretrained model. A report generator (`generate_gpt2_report.py`) produces a fully formatted `.docx` technical report documenting the entire implementation.

## Project Phases

1. **Exploration** — Inspect the pretrained GPT-2 Small model via EasyTransformer: vocabulary, tokenisation, attention patterns, and parameter layout.
2. **From-Scratch Implementation** — Re-implement every sub-module in pure PyTorch with numerical validation against the reference model.
3. **Training** — Train a compact 2-layer, 4-head variant on the Pile-10k dataset (8,506 batches) with validation loss tracking, perplexity logging, and checkpoint saving.
4. **Text Generation** — Implement and compare three decoding strategies: greedy, top-k sampling, and nucleus (top-p) sampling.

## Implemented Components

| Module | Description |
|---|---|
| `LayerNorm` | Pre-norm layer normalisation |
| `Embed` | Token embedding table |
| `PosEmbed` | Learned positional encoding |
| `Attention` | Multi-head self-attention |
| `MLP` | Feed-forward block with GELU activation |
| `TransformerBlock` | Full transformer layer (Attention + MLP + residuals) |
| `Unembed` | Linear projection to vocabulary logits |
| `DemoTransformer` | Complete GPT-2 style model assembly |

## Key Features

- **100% numerical agreement** with the pretrained GPT-2 Small reference model
- **Seven visualisations** including a cross-layer, cross-head attention heatmap grid
- **BPE tokenisation** analysis vocabulary structure, token frequency, and encoding behaviour
- **Training curves**  validation loss and perplexity tracked per epoch
- **Best-model checkpointing** during training
- **Reproducible environment** via pinned `requirements.txt`

## Requirements

```
torch
transformers
easy-transformer   # EasyTransformer / TransformerLens
python-docx
matplotlib
numpy
datasets           # for Pile-10k
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Generate the Technical Report

```bash
python generate_gpt2_report.py
```

This produces `gpt2_report.docx` — a fully formatted Word document covering all four project phases.

## Architecture Notes

- **Pre-norm vs post-norm** — GPT-2 uses pre-norm (LayerNorm before each sub-block) for more stable training.
- **GELU activation** — Chosen over ReLU for smoother gradient flow in the MLP blocks.
- **Learned positional encoding** — GPT-2 uses learned (not sinusoidal) position embeddings, capped at a context length of 1024 tokens.
- **Byte-level BPE** — 50,257 token vocabulary (50,000 merges + 256 byte tokens + `<|endoftext|>`).

## Results

| Model | Layers | Heads | Params | Final Val Loss |
|---|---|---|---|---|
| GPT-2 Small (reference) | 12 | 12 | 117M | — |
| From-scratch variant | 2 | 4 | ~few M | tracked per run |


