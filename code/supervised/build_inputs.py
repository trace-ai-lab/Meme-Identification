"""Build the two per-generator files that finetune/ft_split.py reads, in row order (2,735 memes).

llama_2k7.csv  : human_idea, template, ai_idea_1_clean = Llama 4 Scout idea
gptoss_2k7.csv : human_idea, template, ai_idea_1_clean = gpt-oss-120b idea
Unusable or missing ideas are left empty (read back as NaN).
"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "pipeline"))
from common import human_text, load_ideas, load_pool, template_text  

pool = load_pool()
assert sorted(pool) == list(range(1, 2736)), "expected the 2,735-row pool"
for gen, name in [("llama", "llama_2k7.csv"), ("gptoss", "gptoss_2k7.csv")]:
    ideas = load_ideas(gen)
    with open(os.path.join(HERE, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["title", "image_url", "score", "human_idea", "template", "ai_idea_1_clean"])
        w.writeheader()
        for i in range(1, 2736):
            r = pool[i]
            w.writerow({"title": r["title"], "image_url": r["image_url"], "score": r["score"],
                        "human_idea": human_text(r), "template": template_text(r), "ai_idea_1_clean": ideas.get(i) or ""})
    print(f"{name}: {sum(1 for i in range(1, 2736) if ideas.get(i))} ideas of 2735")
