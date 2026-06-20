from model.config import ModelConfig, TrainConfig

model_config = ModelConfig(
    n_layer=6,
    n_head=6,
    n_embd=384,
)

train_config = TrainConfig(
    out_dir="checkpoints/calliope_30m",
    batch_size=4,
    grad_accum_steps=16,
)
