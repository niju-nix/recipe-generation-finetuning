# Inside the Recipe Evaluation Pipeline

This post explains how `05_Recipie_Model_evaluation.ipynb` evaluates generated recipes and, just as importantly, in what sequence the evaluation functions are called. The notebook is structured as a compact evaluation framework: it starts from a saved prediction artifact, resolves the relevant columns, normalizes the data, computes recipe-specific metrics, adds optional model-backed scores, aggregates the results, and writes everything back out as reusable artifacts.

The top-level execution flow is short and deliberate:

1. `load_input_dataframe(EVAL_CONFIG)`
2. `evaluate_dataframe(_df, EVAL_CONFIG)`
3. `save_artifacts(scored_df, summary_df, EVAL_CONFIG, runtime_info)`

Most of the evaluation logic sits inside `evaluate_dataframe`, which acts as the orchestrator for all helper and metric functions.

## Why This Notebook Matters

Recipe evaluation is harder than ordinary text evaluation. A generated recipe can be fluent and still be wrong. It can mention the right dish title while omitting crucial ingredients. It can look polished while containing weak sequencing, missing temperatures, or impractical cooking steps.

That is why this notebook does not rely on a single score. Instead, it combines:

- lexical similarity metrics,
- ingredient grounding checks,
- diversity and repetition metrics,
- structure and completeness heuristics,
- temporal and coherence checks,
- and optional model-backed semantic metrics.

The result is a more recipe-aware evaluation pipeline that measures not just similarity, but usability.

## Stage 1: Central Configuration With `EVAL_CONFIG`

Everything starts with `EVAL_CONFIG`, a central configuration object that defines:

- the input artifact path,
- expected column names,
- runtime settings,
- model choices for perplexity and BERTScore,
- pass-rate thresholds,
- and output locations.

This is important because the notebook is written to tolerate schema variation. Different runs may expose columns under slightly different names, so later functions do not hardcode a single schema. They consult the config and resolve the correct fields dynamically.

## Stage 2: Load the Prediction Artifact

The first function called in the run cell is:

```python
_df = load_input_dataframe(EVAL_CONFIG)
```

`load_input_dataframe(config)` reads the saved run artifact, which can be either CSV or Parquet. It also respects `max_rows` from the runtime config when a smaller sample is needed for quick experimentation.

This function is the entry point into the pipeline. Once it returns, the notebook has a dataframe containing the generated recipe, the reference recipe, the input prompt, and optionally structured ingredient fields.

## Stage 3: `evaluate_dataframe` Becomes the Orchestrator

The second top-level call is:

```python
scored_df, summary_df, runtime_info = evaluate_dataframe(_df, EVAL_CONFIG)
```

This function is where the full evaluation sequence begins.

### 3.1 Resolve the schema

At the start of `evaluate_dataframe`, the notebook identifies the actual columns to use by calling:

- `resolve_column(df, col_cfg["input"], required=True)`
- `resolve_column(df, col_cfg["prediction"], required=True)`
- `resolve_column(df, col_cfg["reference"], required=True)`
- `resolve_column(df, col_cfg["ingredients"], required=False)`

This makes the pipeline robust across different prediction artifacts.

### 3.2 Standardize text fields

Once the columns are known, the notebook creates working columns:

- `_input`
- `_prediction`
- `_reference`
- `_ingredients_raw`

These are populated using:

- `safe_to_text(...)`

`safe_to_text` converts null values, lists, and mixed cell types into plain strings. This prevents metric functions from having to handle data-format edge cases themselves.

## Stage 4: Build the Gold Ingredient Set

For each row in the dataframe, `evaluate_dataframe` constructs a set of source ingredients before scoring the recipe.

The main call is:

```python
ing_set = ingredient_set_with_fallback(r["_ingredients_raw"], input_text)
```

This function follows a two-step strategy.

### Primary path

If an explicit ingredient column exists and contains usable content, the notebook calls:

1. `ingredient_set(...)`
2. `parse_ingredient_lines(...)`
3. `canonical_ingredient(...)`
4. `tokenize(...)`
5. `normalize_text(...)`

This path parses ingredient strings, strips quantities and units, removes stopwords, and normalizes the ingredient names into a comparable canonical form.

### Fallback path

If the ingredient column is empty or missing, the notebook falls back to the input prompt:

1. `extract_ingredients_from_input(...)`
2. `parse_ingredient_lines(...)`
3. `canonical_ingredient(...)`

This is a practical design choice. Recipe datasets often mix clean structured metadata with loosely formatted prompt text. The fallback keeps evaluation working even when the schema is incomplete.

## Stage 5: Prepare the Prediction for Recipe-Aware Scoring

Still inside the row loop, the notebook extracts structural signals from the predicted recipe.

### Split the recipe into steps

The first call is:

```python
steps = split_steps(pred)
```

`split_steps` tries to find numbered or bullet-style steps first. If the text is not explicitly formatted as steps, it falls back to sentence segmentation.

This matters because several later metrics depend on having an approximate step structure rather than a raw text blob.

### Measure time and temperature specificity

The next supporting call is:

```python
tts = time_temp_spec_score(pred)
```

`time_temp_spec_score` checks whether the prediction includes practical cooking details such as time expressions, temperature expressions, and explicit numeric-with-unit patterns. This helps distinguish vague instructions from genuinely usable recipes.

## Stage 6: Compute Core Deterministic Metrics

Once the notebook has the prediction text, reference text, ingredient set, step list, and time/temperature signal, it computes the main metrics for that row.

### Lexical overlap metrics

These compare the generated recipe with the reference recipe:

- `sentence_bleu4(ref, pred)`
- `rouge_l_f1(ref, pred)`

`sentence_bleu4` is a smoothed BLEU-4 implementation.  
`rouge_l_f1` uses longest common subsequence overlap.

These metrics are useful for measuring surface alignment, but the notebook treats them as only one part of the evaluation picture.

### Structural and procedural metrics

These inspect whether the generation behaves like a real recipe:

- `step_complexity_score(pred)`
- `recipe_coherence_temporal_score(pred)`
- `step_entailment_score_core(ref, pred)`

`step_complexity_score` looks at step count, action verb diversity, and clause richness.

`recipe_coherence_temporal_score` penalizes sequencing problems such as baking before preheating, missing order markers, misordered numbered steps, or missing a finishing cue like "serve".

`step_entailment_score_core` is a heuristic entailment proxy based on overlap of meaningful tokens between reference and prediction.

### Diversity and repetition metrics

To detect generic or repetitive outputs, the notebook computes:

- `repetition_ratio(pred)`
- `diversity_score(pred)`

`diversity_score` internally depends on:

- `distinct_n(text, 1)`
- `distinct_n(text, 2)`
- `ngrams(tokens, n)`

This gives a signal about whether the recipe contains varied and informative language rather than recycled phrasing.

### Ingredient grounding metrics

Ingredient-aware scoring is one of the strongest parts of the notebook. The main call is:

```python
ing_stats = ingredient_metrics(pred, ing_set)
```

Inside that function, the notebook calls:

- `ingredient_mentions_in_prediction(prediction, gold_ingredients)`
- `normalize_text(...)`
- `tokenize(...)`

It then computes:

- ingredient precision,
- ingredient recall,
- ingredient F1,
- ingredient coverage ratio,
- missing ingredient count,
- extra ingredient count,
- hallucination ratio,
- and itemized missing/extra ingredient lists.

This is especially valuable for recipe generation, where a fluent recipe is still a bad recipe if it ignores the supplied ingredients or invents new ones.

### Completeness rubric

After the ingredient metrics and step structure are available, the notebook computes:

```python
recipe_completeness_0_7(
    pred,
    ingredient_recall=row_out["ingredient_recall"],
    step_count=row_out["num_steps"],
    time_temp_score=tts,
)
```

This function assigns a score from 0 to 7 using seven binary checks:

- title-like opening,
- ingredient grounding,
- minimum step count,
- sequencing markers,
- action verb coverage,
- time or temperature detail,
- and a finishing cue.

This produces a readable summary of whether the generation resembles a complete recipe rather than just recipe-like text.

## Stage 7: The Exact Per-Row Call Sequence

Inside the main row loop of `evaluate_dataframe`, the practical order is:

1. `safe_to_text` has already created `_input`, `_prediction`, `_reference`
2. `ingredient_set_with_fallback`
3. `split_steps`
4. `ingredient_metrics`
5. `time_temp_spec_score`
6. `sentence_bleu4`
7. `rouge_l_f1`
8. `step_complexity_score`
9. `diversity_score`
10. `recipe_coherence_temporal_score`
11. `step_entailment_score_core`
12. `tokenize` for `word_count`
13. `repetition_ratio`
14. `recipe_completeness_0_7`

That is the core scoring path repeated for every recipe in the dataframe.

## Stage 8: Add Model-Backed Metrics After Deterministic Scoring

After all row-level deterministic metrics are collected into `metrics_df`, the notebook computes the heavier optional metrics.

### Perplexity

The sequence is:

1. `ppl = PerplexityScorer(config)`
2. `ppl.load()`
3. `ppl.score_batch(work["_prediction"].tolist())`

`PerplexityScorer` wraps a Hugging Face causal language model. It loads the tokenizer and model, selects the device, and scores each generated recipe for fluency. If anything fails, it records a status string instead of failing silently.

The resulting values are written to:

- `perplexity`
- `metric_status_perplexity`

### BERTScore

The notebook then calls:

```python
bs_vals, bs_status = compute_bertscore_batch(
    work["_prediction"].tolist(),
    work["_reference"].tolist(),
    config,
)
```

`compute_bertscore_batch` adds a semantic similarity signal that is less sensitive to exact wording than BLEU or ROUGE.

Its outputs are written to:

- `bertscore_f1`
- `metric_status_bertscore`

This explicit status handling is good engineering practice. If the metric cannot be computed because a dependency is unavailable, the notebook makes that visible in the saved outputs.

## Stage 9: Optional Advanced Paths

If `run_advanced` is enabled, `evaluate_dataframe` also attempts two stubbed extensions:

- `advanced_step_entailment_model_score(...)`
- `llm_judge_recipe(...)`

Both are intentionally adapter-based and raise `NotImplementedError` unless a stronger entailment model or LLM judge adapter is provided. In the current notebook, they serve as extension points rather than the active default path.

## Stage 10: Aggregate the Results

After row-level scoring is complete, `evaluate_dataframe` merges the metric columns back into the original working dataframe and builds summary statistics.

It computes:

- mean,
- median,
- and standard deviation

for every numeric metric in the scored dataframe.

It then calculates pass-rate metrics using the thresholds from `EVAL_CONFIG`, including:

- ingredient recall pass rate,
- coherence pass rate,
- completeness pass rate,
- diversity pass rate.

Finally, it returns:

- `scored_df`
- `summary_df`
- `runtime_info`

`runtime_info` records the resolved columns, metric statuses, and row count so the run stays auditable.

## Stage 11: Save the Evaluation Artifacts

The third top-level function call is:

```python
artifact_paths = save_artifacts(scored_df, summary_df, EVAL_CONFIG, runtime_info)
```

`save_artifacts(...)` writes three files into the configured run directory:

- a row-level scored CSV,
- a summary CSV,
- and a JSON file containing the config and runtime metadata.

This turns the notebook into a reproducible evaluation pipeline rather than a one-off analysis script. Each run produces both detailed scores and compact summaries that can be compared across experiments.

## Full Function Sequence From Start to Finish

If we flatten the notebook into one linear view, the sequence is:

1. `load_input_dataframe`
2. `evaluate_dataframe`
3. `resolve_column`
4. `safe_to_text`
5. `ingredient_set_with_fallback`
6. `ingredient_set`
7. `parse_ingredient_lines`
8. `canonical_ingredient`
9. `extract_ingredients_from_input` when needed
10. `split_steps`
11. `ingredient_metrics`
12. `ingredient_mentions_in_prediction`
13. `time_temp_spec_score`
14. `sentence_bleu4`
15. `rouge_l_f1`
16. `step_complexity_score`
17. `diversity_score`
18. `recipe_coherence_temporal_score`
19. `step_entailment_score_core`
20. `recipe_completeness_0_7`
21. `PerplexityScorer.load`
22. `PerplexityScorer.score_batch`
23. `compute_bertscore_batch`
24. optional advanced metric calls
25. summary aggregation inside `evaluate_dataframe`
26. `save_artifacts`

## Final Thoughts

The notebook is effective because it treats recipe evaluation as a layered problem. It does not assume that overlap with the reference is enough. It checks whether the recipe is grounded in the ingredient list, whether the steps are coherent, whether the instructions feel complete, whether the text is repetitive, and whether the output remains semantically close to the target recipe.

That design makes the evaluation pipeline practical for real recipe-generation work. It provides both fine-grained debugging signals and summary-level metrics, and its function sequence is clear: load, resolve, normalize, score, enrich, summarize, and save.
