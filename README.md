# How Hard Is It to Identify AI-Generated Meme Ideas?

> Code and data for the paper (to appear)

We collected 2,735 popular memes from r/memes and used Qwen3.8-27B to extract each meme's template description and its text as the human idea. 

gpt-oss-120b, Llama 4 Scout and Gemma-4-31B-it then wrote a new idea for each template as the AI idea, and we measured how well LLMs and supervised classifiers tell the human ideas from the AI ones.

## Contents

- `code/`: the code for every step of the paper.
- `data/`: the memes (URLs, human ideas, AI ideas). 

| Step                                          | Code                        |
| --------------------------------------------- | --------------------------- |
| Collect memes from r/memes                    | `collect/`                  |
| Extract template descriptions and human ideas | `extract/extract_qwen38.py` |
| Generate AI ideas                             | `pipeline/gen.py`           |
| LLM discrimination                            | `pipeline/judge.py`         |
| Scoring                                       | `pipeline/evaluate.py`      |
| Supervised classifiers, Scattertext           | `supervised/`               |

## Running the experiments

The memes are copyrighted by their creators, so we share only their URLs; `code/collect/download_images.py` downloads them.

Install `code/requirements/vllm.txt` (Python 3.12). 

`gen.py` and `judge.py` load the Hugging Face models; for the extraction, download `Qwen/Qwen3.8-27B-FP8` and pass it with `--model`. 

gpt-oss-120b and Llama 4 Scout are called through an OpenAI-compatible API (set `LLM_API_BASE` and `LLM_API_KEY`). 

```bash
python code/pipeline/gen.py --generator gemma
python code/pipeline/judge.py --judge qwen38
python code/pipeline/evaluate.py
```

To recompute the LLM and human baseline results (Python 3, standard library only):

```bash
(cd cod/pipeline && python evaluate.py)    
```

```bash
python lr_seed.py                                    # logistic regression
python data_to_text.py && python fasttext_runs.py    # fastText (autotune, 15 min per run)
python xlmr_seeds.py && python xlmr_seeds_large.py   # XLM-R base and large (GPU)
(cd finetune && python gemma_ft.py && python gemma_eval_seeds.py)   # Gemma 4 LoRA (GPU)
```
