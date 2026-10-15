#!/usr/bin/env python
"""Score the discrimination runs (means over 5 seeds +/- standard error, as in the paper's tables).

For generator G, judge J, strategy S and seed s: y_true = 0 for every human idea and 1 for every idea of G;
y_pred = J's labels (1 = AI). Accuracy / precision / recall / F1 with AI as the positive class, x100. Ideas whose
label could not be parsed after 3 attempts are left out and counted, never scored as a guess. Mean and standard
error (sample sd / sqrt(n)) are over the five seeds.
Outputs (<MEME_RUNS>/results/llm_discrimination/): cells.csv (one line per G, J, S with mean/SE/n), best.csv.
"""
import csv
import math
import os
from collections import defaultdict

from common import JUDGES, RUNS, SEEDS, STRATEGIES, judged_sources, load_judge_records

EXCLUDE = set()   # memes left out of the scoring (--exclude-rows), e.g. the two dataset memes of the 4-shot examples


def metrics(h_labels, a_labels):
    tp = sum(a_labels); fn = len(a_labels) - tp
    fp = sum(h_labels); tn = len(h_labels) - fp
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n else float("nan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"acc": 100 * acc, "prec": 100 * prec, "rec": 100 * rec, "f1": 100 * f1}


def mean_se(xs):
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("nan")
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
    return m, sd / math.sqrt(len(xs))


def load_labels(judge):
    lab = defaultdict(dict)            # (source, strategy, seed) -> row -> label or None
    for r in load_judge_records(judge).values():     # all of the judge's files (cooperating workers)
        lab[(r["source"], r["strategy"], r["seed"])][r["row_idx"]] = r.get("label")
    return lab


def evaluate(min_seeds=len(SEEDS)):
    """Paired, balanced scoring: per seed, only memes whose human idea AND generator idea were both judged with a
    parsed label count, so both classes cover the same memes."""
    from common import human_text, load_ideas, load_pool
    pool = load_pool()
    # same criterion judge.py uses to create human-idea tasks
    n_human_full = sum(1 for r in pool.values() if human_text(r).strip())
    cells, incomplete = [], []
    for judge in JUDGES:
        lab = load_labels(judge)
        if not lab:
            continue
        for gen in [g for g in judged_sources(judge) if g != "human"]:
            n_ai_full = sum(1 for v in load_ideas(gen).values() if v)
            for strat in STRATEGIES:
                per_seed, n_pairs, unparsed = [], [], 0
                for seed in SEEDS:
                    h, a = lab.get(("human", strat, seed), {}), lab.get((gen, strat, seed), {})
                    if len(h) < n_human_full or len(a) < n_ai_full:      # seed not finished (or not started)
                        if h or a:
                            incomplete.append(f"{gen}/{judge}/{strat}/seed{seed}: human {len(h)}/{n_human_full}, ai {len(a)}/{n_ai_full}")
                        continue
                    rows = [i for i in a if i in h and a[i] is not None and h[i] is not None and i not in EXCLUDE]
                    unparsed += sum(1 for v in h.values() if v is None) + sum(1 for v in a.values() if v is None)
                    per_seed.append(metrics([h[i] for i in rows], [a[i] for i in rows])); n_pairs.append(len(rows))
                if len(per_seed) < min_seeds:
                    continue
                row = {"generator": gen, "judge": judge, "strategy": strat, "seeds": len(per_seed),
                       "n_human": min(n_pairs), "n_ai": min(n_pairs), "unparsed": unparsed}
                for k in ("acc", "prec", "rec", "f1"):
                    row[k], row[k + "_se"] = mean_se([m[k] for m in per_seed])
                cells.append(row)
    if incomplete:
        print(f"[evaluate] {len(incomplete)} unfinished (generator/judge/strategy/seed) runs left out, e.g.:")
        for x in incomplete[:12]:
            print("   ", x)
    return cells


def write_outputs(cells, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    keys = ["generator", "judge", "strategy", "seeds", "n_human", "n_ai", "unparsed",
            "acc", "acc_se", "prec", "prec_se", "rec", "rec_se", "f1", "f1_se"]
    with open(os.path.join(out_dir, "cells.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for c in cells:
            w.writerow({k: (round(c[k], 4) if isinstance(c[k], float) else c[k]) for k in keys})
    by_pair = defaultdict(list)
    for c in cells:
        by_pair[(c["generator"], c["judge"])].append(c)
    best = {p: max(v, key=lambda c: c["acc"]) for p, v in by_pair.items()}   # best strategy = highest mean accuracy
    with open(os.path.join(out_dir, "best.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
        for c in best.values():
            w.writerow({k: (round(c[k], 4) if isinstance(c[k], float) else c[k]) for k in keys})


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-seeds", type=int, default=len(SEEDS), help="cells need this many complete seeds")
    ap.add_argument("--out", default=os.path.join(RUNS, "results", "llm_discrimination"))
    ap.add_argument("--exclude-rows", default="", help="comma list of row_idx to leave out of the scoring, e.g. "
                    "1811,2235 (the memes of 4-shot examples 1 and 2); use with another --out")
    a = ap.parse_args()
    EXCLUDE.update(int(x) for x in a.exclude_rows.split(",") if x.strip())
    cells = evaluate(a.min_seeds)
    write_outputs(cells, a.out)
    print(f"Wrote {len(cells)} rows to cells.csv and best.csv in {a.out}")
