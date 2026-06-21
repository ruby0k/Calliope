# Calliope-100M Run 001 Report

Date: 2026-06-21

## Run

| Field | Value |
|---|---|
| Run | Calliope-100M-run001 |
| Parameters | 95,233,536 |
| Dataset | TinyStories |
| Tokenizer | GPT-2 |
| Context | 512 |
| Layers | 8 |
| Heads | 12 |
| Embedding | 768 |
| Dropout | 0.1 |
| RoPE theta | 50000 |
| Batch / grad accumulation | 1 / 16 |

## Result

| Metric | Value |
|---|---:|
| Final iter | 10000 |
| Final train loss | 1.6298 |
| Final validation loss | 1.6142 |
| Final loss EMA | 1.6785 |
| Best validation loss | 1.5787 |
| Best validation iter | 9000 |
| VRAM | 2.304 GB |

## Comparison

| Run | Params | Best Val Loss | Final Val Loss | VRAM |
|---|---:|---:|---:|---:|
| Calliope-30M-run001 | 29,920,512 | 1.6511 | 1.6788 | 1.370 GB |
| Calliope-30M-GQA-run001 | 28,740,864 | 1.6799 | 1.6799 | 1.341 GB |
| Calliope-100M-run001 | 95,233,536 | 1.5787 | 1.6142 | 2.304 GB |

## Readout

Calliope-100M is the best model so far by validation loss. It improves over the 30M baseline best validation loss by 0.0724.

The best checkpoint is more important than the final checkpoint here: validation loss bottoms at iter 9000, then rises by iter 10000.

## Recommendation

Keep `checkpoints/calliope_100m/best.pt` as the current best model.

Next run should either stop around 9000 iterations or use early stopping/checkpoint selection by best validation loss. Do not scale architecture again until fixed-prompt sample metrics confirm the quality improvement.
