## Results

### Run overview

| Run | Recipes evaluated | Unique | PASS % | FAIL % | Mean score | Median score |
|---|---|---|---|---|---|---|
| Base run | 10,000 | 8,274 | **86.7%** | **8.3%** | **4.07** | **4.00** |
| LoRA r=16 | 10,000 | 8,274 | 61.8% | 26.4% | 3.58 | 3.75 |
| LoRA r=32 | 10,000 | 8,274 | 61.8% | 25.4% | 3.58 | 3.75 |

> Scores are LLM-as-judge ratings on a 1–5 scale. All runs evaluated 10,000 recipes with 8,274 unique keys (1,726 duplicates present in source data). Bold = best per column.

---

### Dimension breakdown (mean scores, 1–5)

| Dimension | Base run | LoRA r=16 | LoRA r=32 |
|---|---|---|---|
| Ingredient completeness | **4.06** | 3.82 | 3.81 |
| Instruction clarity | **4.43** | 3.68 | 3.70 |
| Cooking logic | **4.10** | 3.33 | 3.34 |
| Recipe coherence | **4.11** | 3.60 | 3.60 |
| Practicality | **4.43** | 3.82 | 3.84 |
| Originality | **3.33** | 2.87 | 2.87 |
| Safety accuracy | **4.93** | 4.65 | 4.66 |
| Reference alignment | **3.80** | 3.22 | 3.24 |

> Bold = best score per dimension. LoRA r=16 and r=32 are near-identical across all dimensions (max delta: 0.02).
