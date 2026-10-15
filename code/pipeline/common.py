"""Shared pieces of the pipeline (generation, discrimination, evaluation).

- Label parsing: drop everything up to the last </think>, then "label: ai" (checked first) -> 1,
  "label: human" -> 0, otherwise unparsed (retried up to 3 times with a new seed).
- Generation prompt: the paper's appendix prompt; the empty JSON schema is
  json.dumps([""] * len(json.loads(human_idea))), falling back to "[]".
- Output cleaning of generated ideas: see clean_idea_how.

Paths: the data folder is <repo>/data unless MEME_DATA is set. Model outputs are read from and written to
MEME_RUNS (default: the data folder, i.e. the released outputs): generations/<generator>.jsonl and
judgments/<judge>.jsonl(.gz). Point MEME_RUNS to an empty folder to run the experiments again from scratch.
"""
import csv
import glob
import gzip
import json
import os
import re
import sys

from prompts import TEMPLATES

csv.field_size_limit(sys.maxsize)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.environ.get("MEME_DATA", os.path.join(ROOT, "data"))
RUNS = os.environ.get("MEME_RUNS", DATA)
EXTRACTION_CSV = os.path.join(DATA, "memes.csv")
GEN_DIR = os.path.join(RUNS, "generations")
JUDGE_DIR = os.path.join(RUNS, "judgments")

GENERATORS = ["gemma", "gptoss", "llama"]
JUDGES = ["gemma", "gemma8", "gptoss", "llama", "qwen38"]
# gemma = Gemma-4-31B-IT, gptoss = gpt-oss-120b, llama = Llama 4 Scout, qwen38 = Qwen3.8-27B (judge only).
# gemma8 = the same Gemma-4-31B checkpoint with FP8 weights (vLLM online fp8, 4k context): the FP8 check in the
# appendix. It is the same model as the generator "gemma", so it does not judge Gemma's ideas either.
JUDGE_BASE = {"gemma8": "gemma"}
STRATEGIES = ["0shot", "0shotCoT", "4shot", "4shotCoT"]
SEEDS = [0, 1, 2, 3, 4]
MAX_ATTEMPTS = 3


def judged_sources(judge):
    """Cross-evaluation: a model never judges its own ideas. Qwen3.8 is not a generator, so it judges all."""
    base = JUDGE_BASE.get(judge, judge)
    return ["human"] + [g for g in GENERATORS if g != base]


# ---------------------------------------------------------------- discrimination prompts
def judge_prompt(strategy, idea, template):
    return TEMPLATES[strategy].format(idea=idea, temp=template)


def remove_thinking(text):
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text.strip()


def parse_label(answer_text):
    t = remove_thinking(answer_text or "").lower()
    if "label: ai" in t:
        return 1
    if "label: human" in t:
        return 0
    return None


# ---------------------------------------------------------------- generation
GEN_PROMPT = (
    "You are an expert at writing funny memes, memes so good that no one can tell they weren’t written by a person. "
    "You will be provided with a description of a meme template and a JSON structure of strings to be filled in. "
    "Use this input to come up with a truly awesome meme. \n\n"
    "Output your response STRICTLY as a single JSON list of strings that represent the individual text boxes. "
    "Leave an empty string if you think a specific text box should be left blank. Each text string needs to be short "
    "so that it will fit in the available space for the meme, no more than ten words per text box. "
    "DO NOT use any strings from the original meme description.\n\n"
    "**CRITICAL OUTPUT INSTRUCTIONS:**\n"
    "1. You must output ONLY the raw, parsable JSON list. \n"
    "2. Do not include any conversational filler, greetings, or explanations. \n"
    "3. Do not use Markdown code blocks (e.g., do not use ```json or ```).\n"
    "4. The very last line of your generated response MUST be exactly `]` and nothing else. "
    "Do not include any trailing whitespace, periods, or newlines after this final bracket.\n"
    "Template Description: {template_text}\n"
    "JSON Structure: {empty_json_str}"
)


def empty_schema(human_idea):
    try:
        v = json.loads(human_idea)
        if isinstance(v, list) and v:
            return json.dumps([""] * len(v))
    except Exception:
        pass
    return "[]"


def generation_prompt(template, human_idea):
    return GEN_PROMPT.format(template_text=template, empty_json_str=empty_schema(human_idea))


def _strip_fences(text):
    t = (text or "").strip()
    m = re.fullmatch(r"```[a-zA-Z]*\s*\n?(.*?)\n?```", t, flags=re.S)
    return m.group(1).strip() if m else t


def clean_idea_how(generator, answer_text):
    """Raw final answer -> (idea string fed to the judges or None, how it was obtained).

    gpt-oss ideas are the answer text itself and are dropped when it is not a JSON list; Gemma and Llama 4 Scout
    ideas are re-serialised with json.dumps(json.loads(...), ensure_ascii=False), and a Llama 4 Scout answer that is
    not valid JSON is kept as raw text. Recorded per idea in `how`: a Markdown fence (which the prompt forbids) is
    removed first ("unfenced"), and a list followed only by stray "]" characters is kept ("salvaged") - the models
    often close the list and then add the "last line must be ]" line the prompt asks for. A salvaged gpt-oss idea is
    the list text exactly as written (no re-serialisation); Gemma/Llama ones are re-serialised like the rest.
    """
    raw = remove_thinking(answer_text or "")
    t = _strip_fences(raw)
    unfenced = t != raw.strip()
    try:
        parsed = json.loads(t)
        ok = isinstance(parsed, list)
    except Exception:
        ok = False
    how = "unfenced" if unfenced else ("verbatim" if generator == "gptoss" else "reserialized")
    salvaged_text = None
    if not ok and "[" in t:
        # '[..]]': keep the list when only whitespace or brackets follow it.
        start = t.index("[")
        try:
            cand, end = json.JSONDecoder().raw_decode(t[start:])
            if isinstance(cand, list) and not t[start + end:].strip(" \n\t]"):
                parsed, ok, how, salvaged_text = cand, True, "salvaged", t[start:start + end]
        except Exception:
            pass
    if generator == "gptoss":   # gpt-oss ideas are kept verbatim (no re-serialisation)
        return ((salvaged_text or t), how) if ok else (None, "unusable")
    if ok:
        return json.dumps(parsed, ensure_ascii=False), how
    return (t, "raw_kept") if (generator == "llama" and t) else (None, "unusable")


def clean_idea(generator, answer_text):
    return clean_idea_how(generator, answer_text)[0]


def human_text(row):
    """The human idea as judges and generators see it: the extraction model's raw answer, unchanged."""
    return row.get("human_idea_raw") or row.get("human_idea") or ""


def template_text(row):
    """The template description as judges and generators see it: the extraction model's answer, with a Markdown
    fence removed if it has one (no released answer has one, so this is template_raw unchanged)."""
    return _strip_fences(row.get("template_raw") or row.get("template") or "")


# ---------------------------------------------------------------- data + jsonl helpers
def load_pool(path=EXTRACTION_CSV):
    """The 2,735 memes (data/memes.csv: row_idx, title, image_url, score, template_raw, human_idea_raw); row_idx is
    1-based. A meme without an extraction (no image) has empty text fields, so the empty-text checks of generation,
    judging and evaluation leave it out."""
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig", newline="")))
    return {int(r["row_idx"]): r for r in rows}


def read_jsonl(path):
    out = []
    if os.path.exists(path):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass  # a line cut by a crash; the task is simply redone
    return out


def judge_output_files(judge, judge_dir=None):
    """A judge's records: judgments/<judge>.jsonl.gz (released, labels only) and judgments/<judge>.jsonl (new runs)."""
    d = judge_dir or JUDGE_DIR
    return sorted(glob.glob(os.path.join(d, f"{judge}.jsonl")) + glob.glob(os.path.join(d, f"{judge}.jsonl.gz")))


def load_judge_records(judge, judge_dir=None):
    """key -> record over all of a judge's files; a finished record beats an error record, and when two files both
    have a finished record the first file in sorted order wins (deterministic)."""
    recs = {}
    for path in judge_output_files(judge, judge_dir):
        for r in read_jsonl(path):
            old = recs.get(r["key"])
            if old is None or ("error" in old and "error" not in r):
                recs[r["key"]] = r
    return recs


def load_ideas(generator):
    """row_idx -> cleaned idea for one generator (unusable answers are left out).
    Ideas are cleaned from the stored raw answer, so every consumer (judges, evaluation, supervised splits) sees the
    same strings."""
    path = os.path.join(GEN_DIR, f"{generator}.jsonl")
    out = {}
    for r in read_jsonl(path):
        if "error" in r or "answer" not in r:
            continue
        out[r["row_idx"]] = clean_idea(generator, r["answer"])
    return {k: v for k, v in out.items() if v}
