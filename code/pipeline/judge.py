#!/usr/bin/env python
"""LLM-prompting discrimination with cross-evaluation.

Sampling: temperature 0.1, top-p 1.0, max tokens 10,240, per-request seed 0-4; unparseable answers are retried with
seed + 1000 * attempt, up to 3 attempts.
Output: <MEME_RUNS>/judgments/<judge>.jsonl (key, source, row_idx, strategy, seed, label, attempts, answer,
reasoning). Tasks that already have a finished record (also in the released <judge>.jsonl.gz) are skipped.
"""
import argparse
import json
import os

os.environ.setdefault("VLLM_USE_DEEP_GEMM", "0")

from common import (JUDGE_DIR, SEEDS, STRATEGIES, human_text, judge_prompt, judged_sources, load_ideas,
                    load_judge_records, load_pool, parse_label, template_text)
from backends import run_api, run_vllm

API_MODELS = {"gptoss": "gpt-oss-120b", "llama": "llama-4-scout"}
GEMMA = "google/gemma-4-31B-it" 
QWEN38 = "Qwen/Qwen3.8-27B-FP8"
VLLM_MODELS = {     
    "gemma": GEMMA,
    "gemma8": GEMMA,
    "qwen38": QWEN38,
}
API_SAMPLING = dict(temperature=0.1, top_p=1.0, max_tokens=10240)
VLLM_SAMPLING = dict(temperature=0.1, top_p=1.0, top_k=-1, max_tokens=10240)


def build_tasks(judge, pool, sources=None, limit=0):
    sources = sources or judged_sources(judge)
    ideas = {}
    for src in sources:
        if src == "human":
            ideas[src] = {i: human_text(r) for i, r in pool.items() if human_text(r).strip()}
        else:
            ideas[src] = {i: v for i, v in load_ideas(src).items() if v}
        if limit:
            ideas[src] = dict(sorted(ideas[src].items())[:limit])
    tasks = []
    for seed in SEEDS:                      # seed-major order: a full seed finishes early
        for strategy in STRATEGIES:
            for src in sources:
                for i, idea in sorted(ideas[src].items()):
                    tasks.append({"key": f"{src}|{i}|{strategy}|{seed}", "source": src, "row_idx": i,
                                  "strategy": strategy, "seed": seed,
                                  "prompt": judge_prompt(strategy, idea, template_text(pool[i]))})
    return tasks


def handle(task, answer, reasoning):
    label = parse_label(answer)
    return label is not None, {"label": label}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", required=True, choices=["gemma", "gemma8", "gptoss", "llama", "qwen38"])
    ap.add_argument("--sources", default=None, help="comma list, e.g. human or human,gemma (default: all)")
    ap.add_argument("--pool", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="first N rows per source (testing)")
    ap.add_argument("--model-path", default=None, help="local copy of the checkpoint (default: Hugging Face)")
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--quantization", default=None)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-seqs", type=int, default=128, help="vLLM concurrency cap (throughput only)")
    ap.add_argument("--llm-kwargs", default=None, help="extra vllm.LLM kwargs as JSON (e.g. a compilation_config)")
    a = ap.parse_args()
    if a.judge == "gemma8" and (a.quantization != "fp8" or a.max_model_len != 4096):
        raise SystemExit("gemma8 is the FP8 Gemma judge: run it with --quantization fp8 --max-model-len 4096 "
                         "--gpu-memory-utilization 0.95")
    if a.judge == "gemma" and a.quantization:
        raise SystemExit("gemma is the BF16 Gemma judge; FP8 Gemma runs as --judge gemma8")
    pool = load_pool(a.pool) if a.pool else load_pool()
    srcs = a.sources.split(",") if a.sources else None
    out = a.out or os.path.join(JUDGE_DIR, f"{a.judge}.jsonl")
    tasks = build_tasks(a.judge, pool, srcs, a.limit)
    done = {k for k, r in load_judge_records(a.judge).items() if "error" not in r}
    tasks = [t for t in tasks if t["key"] not in done]
    if a.judge in API_MODELS:
        model, extra = API_MODELS[a.judge]
        run_api(tasks, out, model, API_SAMPLING, handle, a.concurrency, extra)
    else:
        (path, revision), chat_kwargs = VLLM_MODELS[a.judge]
        llm_kwargs = dict(max_model_len=a.max_model_len, gpu_memory_utilization=a.gpu_memory_utilization,
                          max_num_seqs=a.max_num_seqs, language_model_only=True)   # text only
        if not a.model_path:
            llm_kwargs["revision"] = revision
        if a.quantization:
            llm_kwargs["quantization"] = a.quantization
        if a.llm_kwargs:
            llm_kwargs.update(json.loads(a.llm_kwargs))
        run_vllm(tasks, out, a.model_path or path, VLLM_SAMPLING, handle, tp=a.tp, chat_kwargs=chat_kwargs,
                 llm_kwargs=llm_kwargs)
