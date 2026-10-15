#!/usr/bin/env python
"""Structured extraction with Qwen3.8-27B: template description and meme text (human idea) of every meme.

Input: the meme CSV (title, image_url, score; data/memes.csv works) and images/manifest.csv from
collect/download_images.py (row index -> local image file).
Output
- --out JSONL: one line per row of the input CSV (resumable; finished rows are skipped).
- --finalize: writes the downstream CSV with the columns of data/memes.csv (row_idx, title, image_url, score,
  template_raw, human_idea_raw: the two final answers, unchanged). A meme without an extraction (no image, or a
  failed request) gets empty template_raw and human_idea_raw, which the pipeline leaves out.
"""
import argparse
import csv
import json
import os
import sys
import time

TEMPLATE_PROMPT = (
    "Describe only the visual template of this meme in one sentence. Keep any text that is a "
    "fixed, iconic part of the original meme template itself (i.e., text that appears in EVERY "
    "version of this meme format, such as a catchphrase or sign label). Omit all custom/overlay "
    "text that was added by the meme creator to express a specific opinion, joke, or message."
)
CONTENT_PROMPT = (
    "Ouput the meme strings only, as a single JSON list of strings that represent the levels. "
    "Leave an empty string if you think this level is silent."
)
SAMPLING_PRESETS = {
    # the vLLM settings of generation and discrimination
    "paper": dict(temperature=0.1, top_p=1.0, top_k=-1, max_tokens=10240,
                  frequency_penalty=0.0, presence_penalty=0.0, repetition_penalty=1.0, seed=0),
    # Qwen3.8 model card's recommendation for thinking mode, with max_tokens 10,240 (the paper's extraction run)
    "qwen-thinking": dict(temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, max_tokens=10240,
                          presence_penalty=0.0, repetition_penalty=1.0, seed=0),
}
SAMPLING = SAMPLING_PRESETS["paper"]

csv.field_size_limit(sys.maxsize)


def load_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def done_indices(out_path):
    done = set()
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("status") == "ok":
                    done.add(rec["idx"])
    return done


def conversation(image_path, prompt):
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": f"file://{os.path.abspath(image_path)}"}},
        {"type": "text", "text": prompt},
    ]}]


def downscaled_copy(image_path, cache_dir, max_side=1536):
    from PIL import Image
    os.makedirs(cache_dir, exist_ok=True)
    dst = os.path.join(cache_dir, os.path.splitext(os.path.basename(image_path))[0] + "_ds.png")
    if not os.path.exists(dst):
        im = Image.open(image_path)
        im.seek(0)
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side))
        im.save(dst)
    return dst


def run(args):
    from vllm import LLM, SamplingParams
    import vllm
    import torch
    import transformers

    rows = load_rows(args.input)
    manifest = {int(r["idx"]): r for r in load_rows(args.manifest)}
    todo = []
    done = done_indices(args.out)
    for idx, row in enumerate(rows, start=1):
        m = manifest.get(idx, {})
        if idx in done:
            continue
        if args.only and idx not in args.only:
            continue
        if m.get("status", "").startswith("ok") and m.get("file"):
            todo.append((idx, m["file"]))
    if args.limit:
        todo = todo[: args.limit]
    print(f"rows={len(rows)} already_done={len(done)} to_process={len(todo)}", flush=True)

    image_root = os.path.commonpath([os.path.dirname(os.path.abspath(p)) for _, p in todo] + [os.path.abspath(args.downscale_dir)]) if todo else "/"
    # max_num_seqs only caps concurrency (throughput), not what is generated; Qwen3.8's linear-attention
    # state cache fits fewer than vLLM's default 256 concurrent sequences on a single 46 GB GPU.
    llm = LLM(model=args.model, tensor_parallel_size=args.tp, dtype="bfloat16",
              max_model_len=args.max_model_len, gpu_memory_utilization=0.9, max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"image": 1}, allowed_local_media_path=image_root, seed=0,
              disable_log_stats=False)   # keeps per-request timestamps in RequestOutput.metrics
    sampling = SAMPLING_PRESETS[args.sampling]
    params = SamplingParams(**sampling)
    chat_kwargs = {"enable_thinking": args.thinking}
    if args.thinking:
        chat_kwargs["reasoning_effort"] = args.reasoning_effort

    meta = {
        "model_path": args.model, "model_revision": args.revision, "vllm": vllm.__version__,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "template_prompt": TEMPLATE_PROMPT, "content_prompt": CONTENT_PROMPT,
        "sampling_preset": args.sampling, "sampling": sampling,
        "chat_template_kwargs": chat_kwargs, "tensor_parallel_size": args.tp,
        "max_model_len": args.max_model_len, "gpu_memory_utilization": 0.9, "max_num_seqs": args.max_num_seqs, "chunk_memes": args.chunk,
        "env": {k: os.environ.get(k) for k in ("VLLM_USE_FLASHINFER_SAMPLER", "CUDA_VISIBLE_DEVICES", "CC")},
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.splitext(args.out)[0] + ".run_meta.json", "w") as f:   # one record per output file
        json.dump(meta, f, indent=1)

    def split_answer(text):
        """Thinking runs: (thinking, final answer); the answer is empty if thinking never closed."""
        if not args.thinking:
            return "", text
        if "</think>" in text:
            head, _, tail = text.rpartition("</think>")
            return head.replace("<think>", "").strip(), tail.strip()
        return text.strip(), ""

    def timing(o):
        m = getattr(o, "metrics", None)
        return {k: getattr(m, k, None) for k in ("scheduled_ts", "first_token_ts", "last_token_ts")} if m else {}

    def generate(items):
        convs = []
        for _, path in items:
            convs.append(conversation(path, TEMPLATE_PROMPT))
            convs.append(conversation(path, CONTENT_PROMPT))
        outs = llm.chat(convs, params, chat_template_kwargs=chat_kwargs, use_tqdm=args.tqdm)
        recs = []
        for k, (idx, path) in enumerate(items):
            to, co = outs[2 * k], outs[2 * k + 1]
            t, c = to.outputs[0], co.outputs[0]
            t_think, t_ans = split_answer(t.text)
            c_think, c_ans = split_answer(c.text)
            rec = {"idx": idx, "image_file": path, "status": "ok",
                   "template_raw": t_ans, "content_raw": c_ans,
                   "template_finish": t.finish_reason, "content_finish": c.finish_reason,
                   "template_tokens": len(t.token_ids), "content_tokens": len(c.token_ids),
                   "template_timing": timing(to), "content_timing": timing(co)}
            if args.thinking:
                rec.update(template_thinking=t_think, content_thinking=c_think)
            recs.append(rec)
        return recs

    with open(args.out, "a", encoding="utf-8") as out:
        for start in range(0, len(todo), args.chunk):
            chunk = todo[start: start + args.chunk]
            t0 = time.time()
            try:
                recs = generate(chunk)
            except Exception as e:  # fall back to one meme at a time so one bad image cannot sink the chunk
                print(f"chunk {start} failed ({type(e).__name__}: {str(e)[:200]}); retrying per item", flush=True)
                recs = []
                for item in chunk:
                    try:
                        recs += generate([item])
                    except Exception as e1:
                        try:
                            small = downscaled_copy(item[1], args.downscale_dir)
                            r = generate([(item[0], small)])[0]
                            r["note"] = f"downscaled to 1536px after: {type(e1).__name__}: {str(e1)[:160]}"
                            recs.append(r)
                        except Exception as e2:
                            recs.append({"idx": item[0], "image_file": item[1], "status": "error",
                                         "error": f"{type(e2).__name__}: {str(e2)[:300]}"})
            for r in recs:
                out.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.flush()
            with open(os.path.splitext(args.out)[0] + ".timing.jsonl", "a") as tf:   # for run-time estimates
                tf.write(json.dumps({"start": start, "memes": len(chunk), "wall_s": round(time.time() - t0, 1),
                                     "gen_tokens": sum(r.get("template_tokens", 0) + r.get("content_tokens", 0) for r in recs)}) + "\n")
            print(f"processed {min(start + args.chunk, len(todo))}/{len(todo)} "
                  f"({time.time() - t0:.0f}s for {len(chunk)} memes)", flush=True)


def finalize(args):
    rows = load_rows(args.input)
    manifest = {int(r["idx"]): r for r in load_rows(args.manifest)}
    recs = {}
    with open(args.out, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("status") == "ok" or r["idx"] not in recs:
                recs[r["idx"]] = r
    fields = ["row_idx", "title", "image_url", "score", "template_raw", "human_idea_raw"]   # = data/memes.csv
    n_ok = 0
    tmp = args.finalize + ".tmp"   # written aside and renamed, so pollers never see a half-written file
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for idx, row in enumerate(rows, start=1):
            r = recs.get(idx)
            m = manifest.get(idx, {})
            if r and r.get("status") == "ok":
                n_ok += 1
                w.writerow({"row_idx": idx, "title": row["title"], "image_url": m.get("image_url") or row["image_url"],
                            "score": row["score"], "template_raw": r["template_raw"],
                            "human_idea_raw": r["content_raw"]})
            else:                      # no extraction: empty text fields
                w.writerow({"row_idx": idx, "title": row["title"], "image_url": row["image_url"],
                            "score": row["score"], "template_raw": "", "human_idea_raw": ""})
    os.replace(tmp, args.finalize)
    print(f"wrote {args.finalize}: {n_ok} extracted of {len(rows)} rows")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="meme CSV with title, image_url, score (e.g. data/memes.csv)")
    ap.add_argument("--manifest", required=True, help="images/manifest.csv from collect/download_images.py")
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="", help="HF commit of the model snapshot, recorded in run_meta.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=1000)
    ap.add_argument("--max-model-len", type=int, default=32768)
    ap.add_argument("--max-num-seqs", type=int, default=128)
    ap.add_argument("--thinking", action="store_true", help="enable Qwen3.8 thinking mode (the paper's run)")
    ap.add_argument("--reasoning-effort", default="medium", choices=["low", "medium", "xhigh"])
    ap.add_argument("--sampling", default="paper", choices=sorted(SAMPLING_PRESETS))
    ap.add_argument("--tqdm", action="store_true", help="log vLLM progress and throughput")
    ap.add_argument("--downscale-dir", default="images_downscaled",
                    help="fallback copies at 1,536 px for images the model cannot take at full size")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", type=lambda s: {int(x) for x in s.split(",") if x}, default=None)
    ap.add_argument("--finalize", default="", help="write the downstream CSV to this path and exit")
    a = ap.parse_args()
    finalize(a) if a.finalize else run(a)
