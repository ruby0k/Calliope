from model.config import ModelConfig, TrainConfig

# Calliope-100M base model, de-biased mix v2.
# Same 95M architecture as calliope_100m_hf_mix; the experiment is the DATA + STOPPING:
#   - TinyStories share cut 50% -> 15%, SimpleStories raised to the primary story source
#     (see scripts/prepare_hf_mix.py invocation in docs/EXPERIMENT_WORKFLOW.md).
#   - early stop DISABLED so the cosine LR fully anneals over one ~474M-token epoch
#     (the v3-60k run early-stopped at 81% of the epoch with LR still high).
#   - eval_iters raised 100 -> 200 to halve val-estimate noise.

model_config = ModelConfig(
    block_size=512,
    n_layer=8,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    data_dir="data/hf_mix_v2_gpt2",
    out_dir="checkpoints/calliope_100m_base_mix_v2",
    batch_size=8,
    grad_accum_steps=2,
    max_iters=60000,
    eval_iters=200,
    sequential_train=False,
    early_stop_patience=0,
)
