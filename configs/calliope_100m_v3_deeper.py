from model.config import ModelConfig, TrainConfig

# Ablation lever: reinvest the ~14M params freed by the 32K vocab cut into depth.
# Identical to calliope_100m_v3_general EXCEPT n_layer 8 -> 10 (~75M -> ~87M params).
# Run the same fixed budget as baseline and compare via eval_behavior / compare_runs.

model_config = ModelConfig(
    vocab_size=32000,
    block_size=512,
    n_layer=10,  # +2 layers vs baseline (the one changed variable)
    n_head=12,
    n_kv_head=4,
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    data_dir="data/v3_general_bpe32k",
    out_dir="checkpoints/calliope_100m_v3_deeper",
    batch_size=8,
    grad_accum_steps=2,
    max_iters=60000,
    eval_iters=200,
    ckpt_interval=100,
    optimizer="adamw",
    warmup_iters=1000,
    lr_schedule="wsd",
    sequential_train=False,
    early_stop_patience=0,
    sample_prompts=("", "The process of photosynthesis", "def fibonacci(n):", "The history of the Roman Empire"),
)
