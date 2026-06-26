from model.config import LoraConfig

# Wiki/encyclopedic-prose register specialist on the frozen v2 base.
# Data: data/wikitext_only_gpt2 (prepare_hf_mix with only the wikitext source).
lora_config = LoraConfig(
    base_checkpoint="checkpoints/calliope_100m_base_mix_v2/best.pt",
    data_dir="data/wikitext_only_gpt2",
    out_dir="checkpoints/lora_wiki",
    rank=16,
    alpha=32,
    targets=("c_attn", "c_proj"),
    max_iters=2000,
    learning_rate=2e-4,
    sample_prompts=("", "The history of", "In physics,"),
)
