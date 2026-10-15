"""Two ways to run a list of single-turn chat tasks and append one JSON line per finished task.

Each task is a dict with at least: key (unique), prompt (user text), seed.
`handle(task, answer, reasoning)` turns the model's final answer into (ok, fields); when it is not ok
the task is retried with a new seed (seed + 1000 * attempt) up to MAX_ATTEMPTS attempts in all.
"""
import json
import os
import random
import threading
import time
import concurrent.futures as cf

from common import MAX_ATTEMPTS, read_jsonl

API_BASE = os.environ.get("LLM_API_BASE", "https://llm.jetstream-cloud.org/api")


def pending(tasks, out_path, also=()):
    """Tasks without a finished record in out_path (or in any of the `also` files)."""
    done = {r["key"] for p in [out_path, *also] for r in read_jsonl(p) if "error" not in r}
    return [t for t in tasks if t["key"] not in done]


class _Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if os.path.exists(path) and os.path.getsize(path):
            with open(path, "rb") as f:          # a crash can leave a cut last line; start on a fresh one
                f.seek(-1, os.SEEK_END)
                cut = f.read(1) != b"\n"
            if cut:
                with open(path, "a", encoding="utf-8") as f:
                    f.write("\n")
        self.f = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()
        self.n = 0

    def write(self, rec):
        with self.lock:
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.n += 1
            if self.n % 50 == 0:
                self.f.flush()

    def close(self):
        with self.lock:
            self.f.flush()
            self.f.close()


class _Fatal(Exception):
    pass


def run_api(tasks, out_path, model, sampling, handle, concurrency=32, extra=None, log_every=500):
    """OpenAI-compatible API backend. `extra` holds model-specific kwargs (e.g. reasoning_effort)."""
    import openai
    client = openai.OpenAI(base_url=API_BASE, api_key=os.environ["LLM_API_KEY"], max_retries=0, timeout=900)
    todo = pending(tasks, out_path)
    print(f"[api {model}] {len(tasks)} tasks, {len(todo)} to do, concurrency {concurrency}", flush=True)
    w = _Writer(out_path)
    t0, done, fatal = time.time(), [0], threading.Event()

    def call(prompt, seed):
        """Returns (answer, reasoning, finish, tokens) or raises ValueError for a request the API rejects."""
        n = 0
        while not fatal.is_set():
            try:
                r = client.chat.completions.create(model=model, messages=[{"role": "user", "content": prompt}],
                                                   seed=seed, **sampling, **(extra or {}))
                m = r.choices[0].message
                reasoning = (m.model_extra or {}).get("reasoning_content") or ""
                ntok = r.usage.completion_tokens if r.usage else None
                return m.content or "", reasoning, r.choices[0].finish_reason, ntok
            except openai.APIStatusError as e:
                code = e.status_code
                if code in (401, 403):
                    fatal.set()
                    raise _Fatal(f"authentication refused ({code}): {str(e)[:200]}")
                gateway = code == 400 and any(t in str(e) for t in ("Model not found", "Server Connection Error",
                                                                     "502", "503", "504"))
                if code < 500 and code not in (408, 409, 429) and not gateway:
                    raise ValueError(f"HTTP {code}: {str(e)[:300]}")
                err = f"HTTP {code}"
            except (openai.APIConnectionError, openai.APITimeoutError) as e:
                err = type(e).__name__
            except Exception as e:  # malformed response bodies and similar: treat as transient
                err = f"{type(e).__name__}: {str(e)[:120]}"
            # transient: retry without limit (an outage must not end a multi-day run), capped backoff
            wait = min(300, 5 * 2 ** min(n, 6)) * (0.5 + random.random())
            if n % 5 == 0:
                print(f"  [api {model}] {err} -> retry in {wait:.0f}s (try {n + 1})", flush=True)
            n += 1
            time.sleep(wait)
        raise _Fatal("stopped after a fatal error")

    def work(task):
        if fatal.is_set():
            return
        rec = {k: v for k, v in task.items() if k != "prompt"}
        try:
            for attempt in range(MAX_ATTEMPTS):
                answer, reasoning, finish, ntok = call(task["prompt"], task["seed"] + 1000 * attempt)
                ok, fields = handle(task, answer, reasoning)
                rec.update(fields, attempts=attempt + 1, finish=finish, completion_tokens=ntok,
                           answer=answer, reasoning=reasoning)
                if ok:
                    break
        except ValueError as e:          # this request was rejected; record it, it is retried on the next run
            rec.update(error=str(e))
        except _Fatal:
            return
        except Exception as e:           # never let one task take the pool down
            rec.update(error=f"{type(e).__name__}: {str(e)[:300]}")
        w.write(rec)
        done[0] += 1
        if done[0] % log_every == 0:
            el = time.time() - t0
            print(f"  [api {model}] {done[0]}/{len(todo)} done, {done[0] / el:.2f} task/s, "
                  f"eta {(len(todo) - done[0]) / max(done[0] / el, 1e-9) / 3600:.1f} h", flush=True)

    try:
        with cf.ThreadPoolExecutor(concurrency) as ex:
            list(ex.map(work, todo))
    finally:
        w.close()
    if fatal.is_set():
        raise SystemExit(f"[api {model}] stopped: authentication was refused; fix the key and resubmit (resumes)")
    print(f"[api {model}] finished {done[0]} tasks in {(time.time() - t0) / 60:.1f} min", flush=True)


def _split_thinking(text, thinking):
    if "</think>" in text:
        reasoning, _, answer = text.rpartition("</think>")
    elif thinking:       # thinking never closed (hit max_tokens): no answer, so retry
        reasoning, answer = text, ""
    else:
        reasoning, answer = "", text
    return answer, reasoning


def run_vllm(tasks, out_path, model_path, sampling, handle, tp=1, chat_kwargs=None, llm_kwargs=None,
             chunk=4000, also=()):
    """Local vLLM backend: all prompts of a chunk go to the engine at once, one seed per request.
    Records are written after each chunk, so a kill loses at most one chunk."""
    from vllm import LLM, SamplingParams
    todo = pending(tasks, out_path, also)
    print(f"[vllm {os.path.basename(model_path)}] {len(tasks)} tasks, {len(todo)} to do", flush=True)
    if not todo:
        return
    thinking = bool((chat_kwargs or {}).get("enable_thinking"))
    llm = LLM(model=model_path, tensor_parallel_size=tp, seed=0, disable_log_stats=False, **(llm_kwargs or {}))
    w = _Writer(out_path)
    t0 = time.time()
    try:
        for start in range(0, len(todo), chunk):
            batch = todo[start:start + chunk]
            recs = {t["key"]: {k: v for k, v in t.items() if k != "prompt"} for t in batch}
            open_tasks = batch
            for attempt in range(MAX_ATTEMPTS):
                if not open_tasks:
                    break
                msgs = [[{"role": "user", "content": t["prompt"]}] for t in open_tasks]
                params = [SamplingParams(seed=t["seed"] + 1000 * attempt, **sampling) for t in open_tasks]
                outs = llm.chat(msgs, params, chat_template_kwargs=chat_kwargs or {}, use_tqdm=True)
                retry = []
                for t, o in zip(open_tasks, outs):
                    c = o.outputs[0]
                    answer, reasoning = _split_thinking(c.text, thinking)
                    ok, fields = handle(t, answer, reasoning)
                    recs[t["key"]].update(fields, attempts=attempt + 1, finish=c.finish_reason,
                                          completion_tokens=len(c.token_ids), answer=answer.strip(),
                                          reasoning=reasoning.replace("<think>", "").strip())
                    if not ok:
                        retry.append(t)
                open_tasks = retry
            for r in recs.values():
                w.write(r)
            w.f.flush()
            el = time.time() - t0
            print(f"  [vllm] {min(start + chunk, len(todo))}/{len(todo)} done in {el / 60:.1f} min", flush=True)
    finally:
        w.close()
