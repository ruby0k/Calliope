import math
import re
from collections import Counter

KNOWN_NAMES = {"Anna", "Ben", "Dad", "Ella", "Lily", "Max", "Mia", "Nora", "Sam", "Timmy", "Tom"}


def words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text)


def sentence_lengths(text: str) -> list[int]:
    return [len(words(part)) for part in re.split(r"[.!?]+", text) if words(part)]


def repetition_score(tokens: list[str], n: int = 4) -> float:
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return round(1.0 - len(set(grams)) / len(grams), 4)


def character_name_consistency(prompt: str, text: str) -> float:
    prompt_names = set(words(prompt)) & KNOWN_NAMES
    text_names = [name for name in words(text) if name in KNOWN_NAMES]
    if not text_names:
        return 1.0
    if prompt_names:
        return round(sum(name in prompt_names for name in text_names) / len(text_names), 4)
    counts = {name: text_names.count(name) for name in set(text_names)}
    return round(max(counts.values()) / len(text_names), 4)


def quality_metrics(prompt: str, text: str, generated_ids: list[int], eos_id: int) -> dict:
    toks = words(text)
    lengths = sentence_lengths(text)
    return {
        "repetition_score": repetition_score(toks),
        "eos_inside_output": eos_id in generated_ids,
        "unfinished_sentence": not text.rstrip().endswith((".", "!", "?", "\"")),
        "average_sentence_length": round(sum(lengths) / len(lengths), 2) if lengths else 0.0,
        "unique_token_ratio": round(len(set(generated_ids)) / len(generated_ids), 4) if generated_ids else 0.0,
        "character_name_consistency": character_name_consistency(prompt, text),
    }


def normalize_opening(text: str, k_words: int = 5) -> str:
    return " ".join(words(text)[:k_words]).lower()


def _entropy(counts) -> float:
    total = sum(counts)
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log2(c / total) for c in counts if c)


def opening_diversity(texts: list[str], k_words: int = 5, top: int = 10) -> dict:
    if not texts:
        return {"n": 0, "once_upon_rate": 0.0, "unique_opening_ratio": 0.0, "opening_entropy": 0.0, "top_openings": []}
    openings = [normalize_opening(t, k_words) for t in texts]
    trigrams = [normalize_opening(t, 3) for t in texts]
    counts = Counter(trigrams)
    once = sum(1 for o in openings if o.startswith("once upon a time"))
    return {
        "n": len(texts),
        "once_upon_rate": round(once / len(texts), 4),
        "unique_opening_ratio": round(len(set(openings)) / len(texts), 4),
        "opening_entropy": round(_entropy(counts.values()), 4),
        "top_openings": [[prefix, n] for prefix, n in counts.most_common(top)],
    }
