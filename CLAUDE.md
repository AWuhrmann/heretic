# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Heretic** is a tool for automatic censorship removal from language models via directional ablation ("abliteration"). It extracts hidden states for harmful/harmless prompt pairs, computes per-layer refusal directions, then optimizes ablation parameters (weight kernel shape/position) using TPE-based hyperparameter search (Optuna), applies orthogonal projection to suppress refusal directions, and optionally applies LoRA compensation.

Entry point: `heretic` CLI → `heretic.main:main`

## Commands

```bash
# Install for development
uv sync --all-extras --dev

# Run the tool
uv run heretic

# Lint and format checks (as run in CI)
uv run ruff format --check .
uv run ruff check --output-format=github --extend-select I .
uv run ty check --output-format=github --error-on-warning .

# Build package
uv build
```

There are no automated tests — CI only covers formatting, linting, type checking, and package build.

## Code Style Requirements

Every Python source file **must** start with:
```python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2025-2026  Philipp Emanuel Weidmann <pew@worldwidemann.com> + contributors
```

Additional conventions (enforced by `.gemini/styleguide.md`):
- No abbreviations anywhere (variable names, function names, comments)
- Full type annotations on all function signatures (enforced by `ty`)
- Use `Rich` for all terminal output (not `print` or `tqdm` directly)

## Architecture

### Core Modules (`src/heretic/`)

| Module | Responsibility |
|--------|---------------|
| `main.py` | Orchestration: GPU setup, study checkpoints, Optuna objective function, post-optimization user flow |
| `model.py` | `Model` class — loads transformers (causal LM + multimodal), quantization (bitsandbytes 4-bit), LoRA application, inference, residual extraction, logprob computation |
| `evaluator.py` | `Evaluator` class — measures refusals via configurable markers, computes KL divergence from base model against good/bad prompt datasets |
| `analyzer.py` | `Analyzer` class — residual geometry analysis, PaCMAP projections, animated visualizations (requires `[research]` extras) |
| `config.py` | `Settings` (Pydantic BaseSettings) — priority: CLI > env > TOML file; `DatasetSpecification`, `BenchmarkSpecification`, quantization/normalization enums |
| `utils.py` | Dataset loading (HuggingFace/disk), prompt batching/formatting, device monitoring, interactive prompts (select/text/path/password), Optuna trial parameter extraction |
| `progress.py` | Patches tqdm for Rich-compatible progress display |

### Configuration

- `config.default.toml` — default abliteration params (harmless/harmful datasets, refusal markers, 200 trials, 60 startup trials, TPE sampler)
- `config.noslop.toml` — alternative config targeting "slop" (purple prose) instead of harmful content
- `config.toml` (gitignored) — user-local overrides

### Key Design Patterns

**Optimization loop:** Optuna `JournalStorage` with JSONL backend enables checkpoint/resume across runs. Studies persist in `/checkpoints/`.

**Abliteration pipeline:**
1. Extract residuals (hidden states) from `Model` for harmless and harmful prompt batches
2. `Analyzer` computes per-layer refusal directions (difference-of-means)
3. Optuna TPE optimizes ablation kernel parameters over N trials
4. Apply orthogonal projection via `Model` to suppress refusal directions
5. Optional row normalization + LoRA compensation

**Hardware support:** `model.py` maps across CUDA, XPU, MLU, SDAA, MUSA, NPU, and MPS backends with automatic batch size binary-search for available VRAM.

**Post-optimization interactive flow** (in `main.py`): After the study, user is prompted to save (merge LoRA or keep adapter), upload to HuggingFace Hub, chat with the model, or evaluate alternatives.
