from model.config import LoraConfig

lora_config = LoraConfig(
    base_checkpoint="checkpoints/calliope_100m_base_mix_v2/best.pt",
    data_dir="",  # inherit base's data_dir (the v2 mix) unless overridden
    out_dir="checkpoints/lora_demo",
    rank=8,
    alpha=16,
    targets=("c_attn", "c_proj"),
    max_iters=2000,
    learning_rate=2e-4,
)
