# Recipes Fine-Tuning Workflow

This repository contains the code, notebooks, tests, and documentation used to clean a recipe dataset, fine-tune a recipe-generation model, run inference, and evaluate results.

## Repository Structure

- `scripts/`: reusable Python scripts for run-level analysis and comparisons
- `tests/`: lightweight regression tests for shared evaluation logic
- `docs/`: writeups describing the dataset and evaluation workflow
- `notebooks/`: active research notebooks for exploration, cleanup, training, inference, and evaluation

## Intentionally Not Stored In Git

The repository excludes local-only and generated content:

- raw and processed datasets under `data/`
- run artifacts under `runs/`
- local virtual environments such as `notebooks/.venv/`
- local secrets such as `notebooks/.env`
- notebook scratch archives such as `notebooks/_todelete/`

## Setup

- Python: `3.10+` recommended
- Install dependencies:

```bash
pip install -r requirements.txt
```

- If you run notebooks locally, use a kernel that points to the same environment where the requirements were installed.

## Workflow

1. Dataset exploration:
   Use `notebooks/01_data_exploration.ipynb` to inspect the raw RecipeNLG dataset.
2. Dataset cleanup and export:
   Use `notebooks/02_dataset_cleanup.ipynb` to normalize fields, filter suspicious examples, and export train/eval/test JSONL files locally under `data/`.
3. Fine-tuning:
   Use `notebooks/03_Recipie_SFT_Unsloth_vLLM_Runpod.ipynb` for supervised fine-tuning experiments.
4. Inference:
   Use `notebooks/04_Recipie_Model_Inference_vLLM_Runpod.ipynb` to generate predictions and save local run artifacts under `runs/<RUN_ID>/`.
5. Deterministic evaluation:
   Use `notebooks/05_Recipie_Model_evaluation.ipynb` to compute row-level metrics from `runs/<RUN_ID>/predictions.parquet`.
6. LLM-as-judge evaluation:
   Use `notebooks/06_LLM_as_Judge_evaluation.ipynb` to score generated recipes and write `judge_results.parquet` and `final_eval.parquet`.
7. Run comparison:
   Use `scripts/compare_df_merged_runs.py` against local `runs/<RUN_ID>/final_eval.parquet` outputs.

## Local Data Expectations

- Raw datasets are expected under `data/raw/`
- Processed datasets are expected under `data/processed/`
- Generated model outputs and evaluation artifacts are expected under `runs/`

These folders are local prerequisites for the notebooks and scripts, not committed repository contents.

## Project Status

The notebooks are working research artifacts. Shared logic that needs regression coverage is being moved into importable Python modules under `scripts/`. Tests target those shared modules directly rather than parsing notebook JSON.
