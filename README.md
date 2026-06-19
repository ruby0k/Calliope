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
