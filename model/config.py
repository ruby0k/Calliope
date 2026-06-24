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
    sample_prompts: tuple[str, ...] = ("", "Tom opened the box and saw", '"Why are you crying?" asked Lily.')
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 200
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    loss_ema_beta: float = 0.98
    compile: bool = False
    sequential_train: bool = False
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0
    early_stop_min_iters: int = 0


@dataclass
class LoraConfig:
    # Frozen base + where the adapter trains.
    base_checkpoint: str = "checkpoints/calliope_100m_base_mix_v2/best.pt"
    data_dir: str = ""  # "" -> inherit the base checkpoint's data_dir
    out_dir: str = "checkpoints/lora_demo"
    run_name: str = ""
    experiment_dir: str = "experiments"
    # Adapter shape.
    rank: int = 8
    alpha: int = 16
    lora_dropout: float = 0.0
    targets: tuple[str, ...] = ("c_attn", "c_proj")
    # Training (only adapter params update).
    seed: int = 1337
    batch_size: int = 8
    grad_accum_steps: int = 2
    max_iters: int = 2000
    eval_interval: int = 200
    eval_iters: int = 100
    ckpt_interval: int = 500
    sample_interval: int = 500
    sample_tokens: int = 120
    sample_prompts: tuple[str, ...] = ("", "Tom opened the box and saw")
    learning_rate: float = 2e-4
    min_lr: float = 2e-5
    warmup_iters: int = 50
    weight_decay: float = 0.0
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
