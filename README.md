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

```python
vocab_size = 50257
block_size = 256
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.0
bias = False
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

## Count Parameters

```powershell
uv run python scripts/count_params.py
```

## Check Model Wiring

```powershell
uv run python scripts/smoke_model.py
```
