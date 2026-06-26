from model.config import ModelConfig, TrainConfig

# Ablation lever: curriculum learning (easy -> hard). Identical to the baseline EXCEPT
# sequential_train=True, and it points at a CURRICULUM-ORDERED dataset built with
#   prepare_hf_mix ... --curriculum-order simplestories,wikitext,synthetic,fineweb_edu,code
# so train.bin is ordered easy->hard and sequential batching walks it in that order.

model_config = ModelConfig(
    vocab_size=32000,
    block_size=512,
    n_layer=8,
    n_head=12,
    n_kv_head=4,
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    data_dir="data/v3_curriculum_bpe32k",  # build with --curriculum-order
    out_dir="checkpoints/calliope_100m_v3_curriculum",
    batch_size=8,
    grad_accum_steps=2,
    max_iters=60000,
    eval_iters=200,
    ckpt_interval=100,
    optimizer="adamw",
    warmup_iters=1000,
    lr_schedule="wsd",
    sequential_train=True,  # walk the curriculum-ordered data front-to-back
    early_stop_patience=0,
    sample_prompts=("", "The process of photosynthesis", "def fibonacci(n):", "The history of the Roman Empire"),
)
