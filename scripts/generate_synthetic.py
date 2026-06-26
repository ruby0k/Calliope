"""Generate textbook-style synthetic data from a local teacher via LM Studio's
OpenAI-compatible API (sequence-level distillation, the Phi approach).

Start LM Studio, load a teacher (e.g. Qwen2.5-7B-Instruct), enable the local server,
then:
    uv run python scripts/generate_synthetic.py --out data/synthetic/textbooks.jsonl --per-cell 8

Appends to the jsonl as it goes (resumable — re-run to add more; dedup skips repeats).
Feed the result into training via prepare_hf_mix --synthetic-path.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOPICS = [
    "basic arithmetic", "algebra", "geometry", "probability and statistics",
    "physics: forces and motion", "physics: energy", "chemistry: atoms and molecules",
    "biology: cells", "biology: evolution", "astronomy and the solar system",
    "earth science and weather", "computer science: algorithms", "programming in Python",
    "data structures", "world history", "ancient civilizations", "economics: supply and demand",
    "human anatomy", "ecology and ecosystems", "the scientific method",
    "logic and reasoning", "grammar and writing", "geography", "electricity and circuits",
]
DIFFICULTIES = ["beginner", "intermediate", "advanced"]

# Diversity axis: each generation gets a distinct angle so the teacher doesn't return
# near-identical passages for the same (topic, difficulty) — that was collapsing ~97%
# of outputs at the dedup step. Use per-cell <= len(ASPECTS) for fully distinct prompts.
ASPECTS = [
    "the core definition and key terminology",
    "a fully worked numerical example",
    "a real-world application",
    "a common misconception and its correction",
    "a step-by-step procedure or method",
    "an intuitive analogy that builds understanding",
    "a comparison with a closely related concept",
    "a short exercise followed by its complete solution",
    "the historical context and how it was discovered",
    "edge cases, exceptions, or limitations",
    "why it matters in practice",
    "a step-by-step derivation or proof sketch",
    "three practice questions with full answers",
    "how it connects to a broader field of study",
]

PROMPT = (
    "Write a clear, self-contained educational passage about {topic} for a {difficulty} reader, "
    "in the style of a high-quality textbook, focusing on {aspect}. Use natural prose, "
    "no markdown headers or bullet lists."
)


def text_key(text: str) -> str:
    return hashlib.md5(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def chat(base_url: str, model: str, prompt: str, temperature: float, max_tokens: int, api_key: str, timeout: int = 180) -> str:
    body = json.dumps(
        {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": temperature, "max_tokens": max_tokens}
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="local-model")  # whatever LM Studio reports; usually ignored
    ap.add_argument("--api-key", default="lm-studio")
    ap.add_argument("--out", default="data/synthetic/textbooks.jsonl")
    ap.add_argument("--per-cell", type=int, default=8)  # distinct aspects per (topic, difficulty); cap ~len(ASPECTS) for all-distinct
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=700)
    ap.add_argument("--min-chars", type=int, default=250)
    ap.add_argument("--teacher-tag", default="")  # label stored per example, e.g. "qwen2.5-7b" / "lfm2.5" — for per-teacher ablation
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    if out.exists():  # resume-aware dedup (full-text hash)
        for line in out.open(encoding="utf-8", errors="ignore"):
            try:
                seen.add(text_key(json.loads(line)["text"]))
            except Exception:
                pass
    print(f"teacher: {args.base_url} | resuming with {len(seen)} existing examples")

    written = 0
    attempted = 0
    with out.open("a", encoding="utf-8") as f:
        for topic in TOPICS:
            for diff in DIFFICULTIES:
                for i in range(args.per_cell):
                    aspect = ASPECTS[i % len(ASPECTS)]  # distinct angle per generation
                    attempted += 1
                    try:
                        text = chat(args.base_url, args.model, PROMPT.format(topic=topic, difficulty=diff, aspect=aspect), args.temperature, args.max_tokens, args.api_key)
                    except Exception as e:
                        print(f"  [skip] {topic}/{diff}: {repr(e)[:120]}", flush=True)
                        continue
                    if len(text) < args.min_chars:
                        continue
                    key = text_key(text)
                    if key in seen:
                        continue
                    seen.add(key)
                    f.write(json.dumps({"topic": topic, "difficulty": diff, "aspect": aspect, "teacher": args.teacher_tag, "text": text}, ensure_ascii=False) + "\n")
                    f.flush()
                    written += 1
                    if written % 10 == 0:
                        print(f"  wrote {written}/{attempted} (latest: {topic}/{diff} — {aspect})", flush=True)
    print(f"done: +{written} new examples from {attempted} attempts ({100 * written / max(1, attempted):.0f}% kept), {out} now has {len(seen)} total")


if __name__ == "__main__":
    main()
