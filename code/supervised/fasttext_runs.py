"""fastText supervised with automatic hyperparameter tuning (paper Table 4 row, Table 11 settings).

The original run left no script; this follows Table 11: autotune for 900 s against the dev set, defaults
otherwise, input = data_to_text.py's train/dev/test.txt ("Template: ... | Idea: ..." with __label__0/1).
Five runs with seeds 0-4 give the mean +/- standard error reported in Table 4.
"""
import math

import fasttext
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def read(path):
    y, x = [], []
    for line in open(path, encoding="utf-8"):
        lab, _, text = line.rstrip("\n").partition(" ")
        y.append(int(lab.replace("__label__", ""))); x.append(text)
    return x, y


test_x, test_y = read("test.txt")
res = {k: [] for k in ("acc", "prec", "rec", "f1")}
for seed in [0, 1, 2, 3, 4]:
    # thread=12 is the fastText CLI default the original runs used (the Python API would take every core)
    model = fasttext.train_supervised(input="train.txt", autotuneValidationFile="dev.txt",
                                      autotuneDuration=900, seed=seed, thread=12, verbose=1)
    labels, _ = model.predict(test_x)    # list input: avoids fasttext's numpy-2-incompatible single-text path
    pred = [int(l[0].replace("__label__", "")) for l in labels]
    a = model.f.getArgs()
    print(f"[seed {seed}] tuned: lr={a.lr:.3g} dim={a.dim} wordNgrams={a.wordNgrams} epoch={a.epoch} "
          f"loss={a.loss} ws={a.ws} minCount={a.minCount} minn={a.minn} maxn={a.maxn}")
    for k, fn in (("acc", accuracy_score), ("prec", precision_score), ("rec", recall_score), ("f1", f1_score)):
        res[k].append(100 * fn(test_y, pred))
    print(f"[seed {seed}] Test - " + " | ".join(f"{k}: {v[-1]:.2f}" for k, v in res.items()), flush=True)

print("\nfastText test (mean +/- standard error over 5 runs):")
for k, v in res.items():
    m = sum(v) / len(v); se = math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1)) / math.sqrt(len(v))
    print(f"  {k}: {m:.2f} +/- {se:.2f}")
