<div align="center">

# 🎼 Calliope

> A language model research project exploring how far a dumb student can push language model capabilities under severe compute constraints to avoid paying for subscriptions.
</div>

## Vision

Modern language models are primarily scaled through:

* More parameters
* More compute
* More data
* More GPUs

Calliope explores a different question:

**How much intelligence can be extracted per unit of compute?**

Instead of immediately scaling to billions of active parameters, the project focuses on:

* Efficient architectures
* Data quality
* Distillation
* Sparse activation
* Specialist models
* Adapter composition
* Evolutionary improvement strategies

The long-term goal is to evolve a small model into a capable sparse system that can be trained and experimented with on consumer hardware.

---

# Research Questions

1. How much capability can be obtained per FLOP?
2. How much knowledge can be transferred through distillation?
3. Can specialist adapters outperform dense scaling?
4. Can dataset evolution improve training efficiency?
5. Can sparse activation outperform dense architectures under consumer hardware constraints?
6. What is the optimal trade-off between data quality and model size?

---

# Current Hardware

Primary (and only) development machine:

* Lenovo LOQ with RTX 5050 Laptop GPU (8GB VRAM)

Design constraint:

Every architectural decision should consider consumer hardware feasibility.

---

# Long-Term Goal

Calliope is not intended to become the largest model.

The goal is to become:

**the somewhat compute-efficient model that can reasonably be developed by an independent researcher on consumer hardware.**

---

## Current Config

Calliope-10M:

```python
vocab_size = 50257
block_size = 256
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False
```

Calliope-30M:

```python
block_size = 512
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.1
rope_theta = 50000
batch_size = 2
grad_accum_steps = 16
```

## Setup

```powershell
uv sync
```

## Prepare TinyStories

Default prep uses a small subset so the first run is quick:

```powershell
uv run python scripts/prepare_tinystories.py
```

Use the full dataset by passing negative limits:

```powershell
uv run python scripts/prepare_tinystories.py --max-train-docs -1 --max-val-docs -1
```

This writes:

- `data/tinystories_gpt2/train.bin`
- `data/tinystories_gpt2/val.bin`
- `data/tinystories_gpt2/tokenizer/`
- `data/tinystories_gpt2/meta.json`

## Train

```powershell
uv run python -m train.train
```

Train Calliope-30M:

```powershell
uv run python -m train.train --config configs.calliope_30m --run-name Calliope-30M-run001
```

Experiment configs:

```powershell
uv run python -m train.train --config configs.calliope_30m_dropout01 --run-name Calliope-30M-dropout01-run001
uv run python -m train.train --config configs.calliope_30m_ctx512 --run-name Calliope-30M-ctx512-run001
uv run python -m train.train --config configs.calliope_30m_rope_theta50000 --run-name Calliope-30M-rope50000-run001
```

Smaller tokenizer experiment:

```powershell
uv run python scripts/prepare_tinystories_bpe.py --out-dir data/tinystories_bpe8192 --vocab-size 8192 --max-tokenizer-docs -1 --max-train-docs -1 --max-val-docs -1
uv run python -m train.train --config configs.calliope_30m_tok8192 --run-name Calliope-30M-tok8192-run001
```

Useful smoke run:

```powershell
uv run python -m train.train --max-iters 20 --eval-iters 5 --batch-size 2 --grad-accum-steps 1
```

Outputs:

- checkpoints in `checkpoints/calliope_10m/`
- run config and metrics in `experiments/<run_name>/`

## Sample

```powershell
uv run python scripts/sample.py --prompt "Once upon a time"
```

Generate 12 samples from the fixed prompt set:

```powershell
uv run python scripts/sample_fixed_prompts.py --samples-per-prompt 4
```

The fixed prompt set is grouped by `story_start`, `continuation`, `dialogue`, `cause_effect`, and `ending`. Each sample row includes simple quality metrics: `repetition_score`, `eos_inside_output`, `unfinished_sentence`, `average_sentence_length`, `unique_token_ratio`, and `character_name_consistency`.

Check prepared train/val shard quality:

```powershell
uv run python scripts/check_split_quality.py
```

## Count Parameters

```powershell
uv run python scripts/count_params.py
```

```powershell
uv run python scripts/count_params.py --config configs.calliope_30m
```

## Check Model Wiring

```powershell
uv run python scripts/smoke_model.py
```
