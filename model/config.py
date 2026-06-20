from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    block_size: int = 256
    n_layer: int = 4
    n_head: int = 4
    n_kv_head: int | None = None
    n_embd: int = 256
    dropout: float = 0.0
    rope_theta: float = 10000.0
    bias: bool = False


@dataclass
class TrainConfig:
    data_dir: str = "data/tinystories_gpt2"
    out_dir: str = "checkpoints/calliope_10m"
    experiment_dir: str = "experiments"
    run_name: str = ""
    seed: int = 1337
    batch_size: int = 8
    grad_accum_steps: int = 8
    max_iters: int = 10000
    eval_interval: int = 500
    eval_iters: int = 100
    ckpt_interval: int = 1000
    sample_interval: int = 1000
    sample_tokens: int = 120
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    loss_ema_beta: float = 0.98
    compile: bool = False
