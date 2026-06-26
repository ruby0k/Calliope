from model.config import ModelConfig, TrainConfig

model_config = ModelConfig(
    block_size=512,
    n_layer=8,
    n_head=12,
    n_embd=768,
    dropout=0.1,
    rope_theta=50000.0,
)

train_config = TrainConfig(
    out_dir="checkpoints/calliope_100m",
    batch_size=8,
    grad_accum_steps=2,
)
