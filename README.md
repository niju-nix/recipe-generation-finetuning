# qwen-recipe-generation-finetuning

Fine-tuning **Qwen2.5-7B-Instruct** on the RecipeNLG dataset using LoRA, evaluated with both deterministic metrics and an LLM-as-judge pipeline across 10,000 recipes.

- Supervised fine-tuning with [Unsloth](https://github.com/unslothai/unsloth) + [vLLM](https://github.com/vllm-project/vllm) on RunPod
- LoRA rank ablation (r=8 vs r=16) against a base model baseline
- LLM-as-judge evaluation via [openRouter.ai](https://openrouter.ai) using Qwen2.5-7B as judge
- Scored across 8 domain-specific dimensions (ingredient completeness, cooking logic, safety, and more)

---

## Results

All runs evaluated on 10,000 recipes. Scores are LLM-as-judge ratings on a 1–5 scale.

| Model | Recipes | Overall ↑ | PASS % ↑ | Ingredient | Clarity | Logic | Coherence | Practicality | Originality | Safety | Alignment |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline (base model) | 10,000 | **4.07** | **86.7%** | **4.06** | **4.44** | **4.10** | **4.11** | **4.43** | **3.33** | **4.93** | **3.80** |
| LoRA r=16 | 10,000 | 3.58 | 61.8% | 3.82 | 3.68 | 3.33 | 3.60 | 3.82 | 2.87 | 4.65 | 3.22 |
| LoRA r=32 | 10,000 | 3.58 | 61.8% | 3.81 | 3.70 | 3.34 | 3.60 | 3.84 | 2.87 | 4.66 | 3.24 |

### Key findings

- **The base model outperforms both LoRA variants across all 8 dimensions** — a counterintuitive result that points to likely causes: the LoRA models were evaluated before sufficient training epochs, and the base model benefits from strong prior instruction-following on recipe-style prompts. This is an active area of investigation.
- **LoRA r=16 and r=32 are virtually identical** (max delta: 0.02 across all dimensions), suggesting diminishing returns from higher rank — a useful efficiency insight in itself.
- **Safety accuracy is the strongest dimension** across all runs (4.65–4.93), indicating the model does not hallucinate dangerous cooking instructions regardless of configuration.
- **Reference alignment is the weakest dimension** (3.22–3.80), which is expected for a generative task where paraphrasing is valid but penalised against a fixed reference.

---

## Models on Hugging Face

Fine-tuned adapters are published on Hugging Face and can be loaded directly:

| Model | Rank | Alpha | Link |
|---|---|---|---|
| Qwen2.5-7B-Instruct — RecipeNLG | r=8 | 16 | [nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank8_alpha16](https://huggingface.co/nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank8_alpha16) |
| Qwen2.5-7B-Instruct — RecipeNLG | r=16 | 32 | [nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank16_alpha32](https://huggingface.co/nijumich/Qwen2.5-7B-Instruct-recipieNLG_rank16_alpha32) |

---

## Dataset

**RecipeNLG** — 2M+ semi-structured cooking recipes.

- Homepage: [recipenlg.cs.put.poznan.pl](https://recipenlg.cs.put.poznan.pl/)
- Hugging Face mirror: [mbien/recipenlg](https://huggingface.co/datasets/mbien/recipenlg)

Download and place in `data/raw/`. Processed train/eval/test JSONL splits are exported to `data/processed/` by the cleanup notebook. Neither folder is committed to this repo.

---

## Workflow

### 1. Data exploration
`notebooks/01_data_exploration.ipynb` — inspect the raw RecipeNLG dataset, check distributions such as recipe length, ingredient count, and identify recipie quality issues.

### 2. Dataset cleanup
`notebooks/02_dataset_cleanup.ipynb` — normalise fields, filter suspicious examples, export train/eval/test JSONL files to `data/processed/`.

### 3. Fine-tuning
`notebooks/03_recipe_sft_unsloth_vllm_runpod.ipynb` — supervised fine-tuning using Unsloth + vLLM on RunPod. Fine-tuned adapters are saved to Hugging Face (see [Models](#models-on-hugging-face) above).

### 4. Inference
`notebooks/04_recipe_model_inference_vllm_runpod.ipynb` — load fine-tuned models from Hugging Face, generate predictions, save run artifacts to `runs/<RUN_ID>/`.

### 5. Deterministic evaluation
`notebooks/05_recipe_model_evaluation.ipynb` — compute row-level metrics from `runs/<RUN_ID>/predictions.parquet`.

### 6. LLM-as-judge evaluation
`notebooks/06_llm_as_judge_evaluation.ipynb` — score generated recipes using Qwen2.5-7B via the openrouter.ai API. Writes `judge_results.parquet` and `final_eval.parquet` per run.

### 7. Run comparison
`scripts/compare_df_merged_runs.py` — aggregate and compare `final_eval.parquet` outputs across multiple runs.

---

## Repository structure

```
qwen-recipe-generation-finetuning/
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/                  # not committed — download from RecipeNLG
│   └── processed/            # not committed — exported by notebook 02
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_dataset_cleanup.ipynb
│   ├── 03_recipe_sft_unsloth_vllm_runpod.ipynb
│   ├── 04_recipe_model_inference_vllm_runpod.ipynb
│   ├── 05_recipe_model_evaluation.ipynb
│   └── 06_llm_as_judge_evaluation.ipynb
│
├── scripts/
│   └── compare_df_merged_runs.py
│
├── tests/
│   └── ...                   # regression tests for shared evaluation logic
│
└── docs/
    └── ...                   # dataset and evaluation writeups
```

---

## Setup

**Requirements:** Python 3.10+

```bash
git clone https://github.com/YOUR_USERNAME/qwen-recipe-generation-finetuning
cd qwen-recipe-generation-finetuning
pip install -r requirements.txt
```

If running notebooks locally, point your kernel to the same environment where requirements were installed.

**Compute:** Fine-tuning and inference were run on RunPod GPU instances. Local execution of training notebooks is not recommended without a CUDA-capable GPU.

---

## Tech stack

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![HuggingFace](https://img.shields.io/badge/HuggingFace-FFD21E?style=flat&logo=huggingface&logoColor=black)
![Unsloth](https://img.shields.io/badge/Unsloth-fast%20finetuning-orange?style=flat)
![vLLM](https://img.shields.io/badge/vLLM-inference-blue?style=flat)

| Component | Tool |
|---|---|
| Base model | Qwen2.5-7B-Instruct |
| Fine-tuning | [Unsloth](https://github.com/unslothai/unsloth) |
| PEFT | LoRA via [HuggingFace PEFT](https://github.com/huggingface/peft) |
| Inference | [vLLM](https://github.com/vllm-project/vllm) |
| LLM-as-judge | Qwen2.5-7B via [openrouter.ai](https://openrouter.ai) |
| Compute | [RunPod](https://runpod.io) (Fine-tuning and Inference) |
| Model hosting | [Hugging Face Hub](https://huggingface.co) |

---

## Not stored in Git

| Path | Reason |
|---|---|
| `data/` | Raw and processed datasets — download from RecipeNLG |
| `runs/` | Inference and evaluation artifacts — generated locally |
| `notebooks/.venv/` | Local virtual environment |
| `notebooks/.env` | Local secrets |
| `notebooks/_todelete/` | Notebook scratch archives |
