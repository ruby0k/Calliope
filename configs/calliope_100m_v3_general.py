from model.config import ModelConfig, TrainConfig

# Calliope-100M v3 — GENERAL base (not a narrator).
# Mix is general-dominant: FineWeb-Edu core + Code + WikiText, with only a small
# SimpleStories slice for short-text fluency. TinyStories/Calliope dropped.
# North star is general-text loss (fineweb/wikitext/code), NOT story metrics.

model_config = ModelConfig(
    vocab_size=32000,  # custom byte-level BPE (down from GPT-2 50257) — frees ~14M params
    block_size=512,
    n_layer=8,
    n_head=12,
    n_kv_head=4,  # GQA: 4 KV heads (from 12) — smaller KV cache, ~free at this scale
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    data_dir="data/v3_general_bpe32k",
    out_dir="checkpoints/calliope_100m_v3_general",
    batch_size=8,
    grad_accum_steps=2,
    max_iters=60000,
    eval_iters=200,
    ckpt_interval=100,  # save last.pt every 100 iters so you can stop/resume freely
    optimizer="muon",  # hybrid Muon (hidden matrices) + AdamW (embeddings/norms)
    muon_lr=0.006,  # lowered from 0.02 — 0.02 diverged this model after warmup
    warmup_iters=1000,  # longer warmup for from-scratch stability (was 200)
    lr_schedule="wsd",  # warmup-stable-decay, better for long/over-trained runs
    sequential_train=False,
    early_stop_patience=0,
    sample_prompts=("", "The process of photosynthesis", "def fibonacci(n):", "The history of the Roman Empire"),
)
