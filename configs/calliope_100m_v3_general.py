from model.config import ModelConfig, TrainConfig

# Calliope-100M v3 — GENERAL base (not a narrator).
# Mix is general-dominant: FineWeb-Edu core + Code + WikiText, with only a small
# SimpleStories slice for short-text fluency. TinyStories/Calliope dropped.
# North star is general-text loss (fineweb/wikitext/code), NOT story metrics.

model_config = ModelConfig(
    block_size=512,
    n_layer=8,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    data_dir="data/v3_general_gpt2",
    out_dir="checkpoints/calliope_100m_v3_general",
    batch_size=8,
    grad_accum_steps=2,
    max_iters=60000,
    eval_iters=200,
    ckpt_interval=100,  # save last.pt every 100 iters so you can stop/resume freely
    sequential_train=False,
    early_stop_patience=0,
    sample_prompts=("", "The process of photosynthesis", "def fibonacci(n):", "The history of the Roman Empire"),
)
