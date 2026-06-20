from model.config import ModelConfig, TrainConfig

model_config = ModelConfig(
    vocab_size=8192,
    n_layer=6,
    n_head=6,
    n_embd=384,
)

train_config = TrainConfig(
    data_dir="data/tinystories_bpe8192",
    out_dir="checkpoints/calliope_30m_tok8192",
    batch_size=4,
    grad_accum_steps=16,
)
