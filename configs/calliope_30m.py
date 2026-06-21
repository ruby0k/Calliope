from model.config import ModelConfig, TrainConfig

model_config = ModelConfig(
    block_size=512,
    n_layer=6,
    n_head=6,
    n_embd=384,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    out_dir="checkpoints/calliope_30m",
    batch_size=2,
    grad_accum_steps=16,
)
