#!/usr/bin/env python
"""Generate one AI meme idea per meme (the paper's generation prompt).

Generators: gemma = Gemma-4-31B-IT (local vLLM, chat-template default = no thinking),
gptoss = gpt-oss-120b and llama = Llama 4 Scout (Jetstream2 inference service, OpenAI-compatible API).
Sampling: temperature 0.1, top-p 1.0, max tokens 10,240; generation uses seed 0 and a single attempt, and an
unusable answer stays missing.
Output: <MEME_RUNS>/generations/<generator>.jsonl (row_idx, idea, cleaning, status, raw answer, reasoning).
"""
import argparse
import os

from common import GEN_DIR, clean_idea_how, generation_prompt, human_text, load_pool, template_text
from backends import run_api, run_vllm

API_MODELS = {"gptoss": ("gpt-oss-120b", {"reasoning_effort": "medium"}), "llama": ("llama-4-scout", {})}
GEMMA = ("google/gemma-4-31B-it", "842da3794eaa0b77d5f08bae87a17459d91ff475")   # Hugging Face id, revision
API_SAMPLING = dict(temperature=0.1, top_p=1.0, max_tokens=10240)
VLLM_SAMPLING = dict(temperature=0.1, top_p=1.0, top_k=-1, max_tokens=10240)


def build_tasks(pool, limit=0):
    tasks = []
    for idx, r in sorted(pool.items()):
        template = template_text(r)
        if not template:                    # no template description (meme without an extraction)
            continue
        tasks.append({"key": str(idx), "row_idx": idx, "seed": 0,
                      "prompt": generation_prompt(template, human_text(r))})
    return tasks[:limit] if limit else tasks


def make_handle(generator):
    def handle(task, answer, reasoning):
        idea, how = clean_idea_how(generator, answer)
        return True, {"idea": idea, "cleaning": how, "status": "ok" if idea else "unusable"}
    return handle


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--generator", required=True, choices=["gemma", "gptoss", "llama"])
    ap.add_argument("--pool", default=None, help="meme CSV (default: data/memes.csv)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N memes only (testing)")
    ap.add_argument("--model-path", default=None, help="local copy of the Gemma checkpoint (default: Hugging Face)")
    ap.add_argument("--tp", type=int, default=1)
    a = ap.parse_args()
    pool = load_pool(a.pool) if a.pool else load_pool()
    out = a.out or os.path.join(GEN_DIR, f"{a.generator}.jsonl")
    tasks = build_tasks(pool, a.limit)
    if a.generator in API_MODELS:
        model, extra = API_MODELS[a.generator]
        run_api(tasks, out, model, API_SAMPLING, make_handle(a.generator), a.concurrency, extra)
    else:
        llm_kwargs = dict(max_model_len=16384, gpu_memory_utilization=0.9, max_num_seqs=128,
                          language_model_only=True)   # text only
        if not a.model_path:
            llm_kwargs["revision"] = GEMMA[1]
        run_vllm(tasks, out, a.model_path or GEMMA[0], VLLM_SAMPLING, make_handle(a.generator), tp=a.tp,
                 llm_kwargs=llm_kwargs)
