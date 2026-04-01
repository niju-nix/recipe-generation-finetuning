# Inside the Recipe Evaluation Pipeline

This post explains how the recipe evaluation workflow in the notebooks turns raw model generations into a structured assessment of quality. The pipeline is split across two notebooks:

- `notebooks/05_Recipie_Model_evaluation.ipynb`
- `notebooks/06_LLM_as_Judge_evaluation.ipynb`

Together, they implement a layered evaluation strategy. The first notebook produces deterministic, row-level metrics from model predictions. The second notebook adds an LLM-as-a-judge pass that compares each generated recipe against the ground-truth reference and produces culinary quality scores, pass/fail verdicts, and qualitative feedback. The result is a more complete picture of model behavior than any single metric could provide.

## Why This Pipeline Exists

Recipe generation is harder to evaluate than generic text generation. A recipe can be fluent but still wrong. It can mention the right ingredients but order the steps badly. It can sound convincing while omitting temperatures, missing core actions, or drifting away from the reference dish.

Because of that, the pipeline does not rely on one score. Instead, it combines:

- lexical overlap metrics,
- structure and completeness heuristics,
- ingredient grounding checks,
- optional model-based semantic metrics,
- and a final rubric-based LLM judge.

This design makes the evaluation process both practical and interpretable. Deterministic metrics provide consistency and scale. The judge model adds context-sensitive reasoning that fixed rules cannot capture on their own.

## Stage 1: Load Predictions From a Run Artifact

The process begins with a saved inference run inside the `runs/` directory. In `05_Recipie_Model_evaluation.ipynb`, the evaluation config points to a predictions artifact such as:

```text
../runs/<RUN_ID>/predictions.parquet
```

The notebook centralizes runtime settings inside `EVAL_CONFIG`, including:

- where the input file lives,
- which column names to expect,
- runtime limits such as batch size or row caps,
- model settings for optional metrics like perplexity and BERTScore,
- and threshold values used to compute pass rates.

This matters because recipe generation outputs are rarely perfectly standardized. Different training or inference runs may expose slightly different schemas. The config-driven approach makes the notebook reusable across experiments without rewriting metric logic every time.

## Stage 2: Resolve Columns and Normalize the Data

Once the run artifact is loaded, the notebook resolves the actual columns to use for input, prediction, reference, title, and ingredients. It supports multiple candidate names for each field, which is important when artifacts come from different notebook versions.

After column resolution, helper functions normalize the text into a consistent format:

- text is cleaned and lowercased,
- lists are converted into strings where needed,
- ingredients are parsed from explicit columns or extracted from the prompt as a fallback,
- ingredient names are canonicalized by stripping quantities, units, and common stopwords.

This normalization step is foundational. Without it, evaluation would be brittle. Small formatting differences would look like model failures even when the underlying recipe content was acceptable.

## Stage 3: Compute Deterministic Recipe Metrics

The first notebook then computes a set of deterministic metrics for every generated recipe. These metrics are designed to answer different questions about quality.

### Lexical Similarity

The notebook computes `bleu4` and `rouge_l_f1` between the generated recipe and the reference. These scores capture surface-level overlap and give a quick signal about how closely the generation tracks the expected wording and content sequence.

They are useful, but intentionally not treated as the whole story. In recipe generation, a semantically correct rewrite can score lower than expected on lexical overlap, while a shallow copy can score high.

### Diversity and Repetition

To detect bland or repetitive generations, the notebook computes:

- `repetition_ratio`
- `distinct-n`
- `diversity_score`

These metrics help reveal whether the model is producing varied, information-rich instructions or falling into generic repeated phrasing.

### Ingredient Grounding

A recipe is not useful if it ignores the supplied ingredients. The pipeline explicitly measures ingredient usage with:

- `ingredient_precision`
- `ingredient_recall`
- `ingredient_f1`
- `ingredient_coverage_ratio`
- missing and extra ingredient counts
- hallucination ratio

This is one of the most recipe-specific parts of the evaluation design. Instead of judging only the final prose, the notebook checks whether the generation stays grounded in the ingredient set that should constrain the recipe.

### Structural and Practicality Heuristics

The notebook also evaluates whether the recipe behaves like a usable recipe rather than arbitrary text. It computes:

- `num_steps`
- `word_count`
- `time_temp_spec_score`
- `step_complexity_score`
- `recipe_coherence_temporal_score`
- `recipe_completeness_0_7`

These heuristics look for signals such as:

- whether steps exist and are sufficiently developed,
- whether action verbs are present,
- whether time and temperature cues appear,
- whether the order of operations is plausible,
- and whether the recipe includes a coherent start-to-finish structure.

This part is especially valuable because many generation failures are procedural rather than lexical. A recipe can overlap well with the reference and still be incomplete, badly ordered, or impractical for a real cook.

## Stage 4: Add Model-Backed Semantic Metrics

After the deterministic scores are computed, the notebook optionally enriches the evaluation with model-backed metrics.

### Perplexity

A Hugging Face causal language model is used to compute perplexity over the generated recipe. This acts as a fluency-oriented signal: lower perplexity generally indicates more natural, more probable text under the scoring model.

### BERTScore

BERTScore is used as a softer semantic similarity metric between prediction and reference. Unlike BLEU or ROUGE, it is less tied to exact phrasing and can better reward semantically similar paraphrases.

The notebook is careful about failure handling here. If a model package is unavailable or a load fails, the metric status is recorded explicitly rather than silently dropping the score. That makes the evaluation run auditable and easier to debug.

## Stage 5: Save Row-Level Scores and Summary Artifacts

At the end of the first notebook, the scored dataframe is saved to the run directory as an artifact like `eval_scored_*.csv`, along with:

- a summary file containing aggregate statistics,
- and a JSON config snapshot containing resolved columns and metric runtime status.

This stage is important operationally. It freezes the evaluation state for a particular run, which means later analysis and judge passes do not need to recompute the entire metric stack.

## Stage 6: Standardize Inputs for LLM-as-a-Judge

The second notebook, `06_LLM_as_Judge_evaluation.ipynb`, starts from the scored artifact produced earlier. Its first job is to standardize the fields expected by the judge:

- `recipe_id`
- `title`
- `ingredients`
- `reference`
- `prediction`

It also converts references into a list-of-steps format and normalizes ingredients into clean bullet-like lists. This allows the judge prompt to stay consistent even when source data formats vary.

That consistency matters because judge reliability depends heavily on input formatting. If the recipe, ingredient list, and reference are presented inconsistently, the model is more likely to produce unstable evaluations.

## Stage 7: Use an LLM Judge With a Culinary Rubric

The core of the second notebook is a structured prompt that asks a judge model to act like a professional chef, food writer, and culinary evaluator. The judge compares the generated recipe against the reference recipe and scores it across eight dimensions:

1. Ingredient completeness
2. Instruction clarity
3. Cooking logic
4. Recipe coherence
5. Practicality
6. Originality
7. Safety and accuracy
8. Reference alignment

The model must return strict JSON with:

- a 1 to 5 score for each dimension,
- an `overall_score`,
- a `verdict` of `PASS` or `FAIL`,
- short feedback,
- and a list of key differences from the reference.

This is the point where the pipeline moves beyond purely formulaic scoring. The judge can detect issues such as unsafe technique, poor sequencing, weak alignment with the target dish, or missing culinary logic that would be difficult to capture with fixed heuristics alone.

## Stage 8: Run the Judge in Async Batches With Resume Support

The notebook is built to handle large evaluation runs rather than only one-off examples. It uses asynchronous API calls through `AsyncOpenAI`, batches requests conservatively, and writes intermediate parquet results after each batch.

Several operational safeguards are built in:

- `MAX_ROWS` limits evaluation for quick iteration,
- `BATCH_SIZE` controls throughput and rate-limit risk,
- `RESUME` allows the notebook to continue from prior partial results,
- incremental saves reduce the chance of losing work after a failure,
- parquet writes are retried with backoff.

This turns the notebook from a demo into a practical experiment pipeline. Long-running judge jobs are expensive enough that resumability is not a luxury. It is part of making evaluation reproducible and usable in real workflows.

## Stage 9: Normalize Judge Outputs for Analysis

Because LLM outputs can be noisy even with strict prompting, the notebook normalizes the judge results before analysis:

- score columns are coerced to numeric,
- verdict strings are standardized to uppercase,
- `key_differences` is converted into a reliable list format,
- sparse columns trigger warnings.

This cleanup step is subtle but important. The notebook assumes that downstream analysis should be deterministic, even if the upstream scorer is an LLM. That design choice keeps the analytics layer stable.

## Stage 10: Analyze Model Quality From Multiple Angles

Once judge results are normalized, the notebook computes high-level summaries such as:

- average score per dimension,
- pass/fail rate,
- score correlations between dimensions,
- score distributions,
- weakest-dimension counts,
- and the most common qualitative failures extracted from `key_differences`.

It also creates a radar chart labeled as a culinary capability profile. That visualization is a strong fit for this use case because it shows whether the model is balanced or whether performance is concentrated in a few dimensions while lagging in others.

For example, a model might be strong on clarity and originality but weaker on safety or ingredient completeness. A single overall score would hide that shape. The radar view makes it visible immediately.

## Stage 11: Merge Everything Into a Final Evaluation Table

The last step merges the judge outputs back into the original dataframe using `recipe_id` and writes a final artifact:

```text
../runs/<RUN_ID>/final_eval.parquet
```

This final table combines:

- the original run data,
- deterministic evaluation metrics,
- model-backed semantic metrics,
- and LLM judge outputs.

That merged artifact becomes the main dataset for comparing runs, diagnosing failure modes, and deciding whether a model is actually improving in the ways that matter.

## What Makes This Pipeline Strong

The most effective part of this workflow is that it treats recipe evaluation as a layered problem instead of a single-score problem.

It uses deterministic metrics where consistency matters, recipe-specific heuristics where task structure matters, and rubric-based LLM judgment where contextual culinary reasoning matters. Each layer compensates for the blind spots of the others.

In practice, that means the pipeline can answer richer questions:

- Is the model grounded in the ingredient list?
- Are the steps coherent and executable?
- Does the recipe resemble the reference in a meaningful way?
- Is the text diverse without becoming generic noise?
- Are there safety or logic issues that a simple overlap score would miss?

For recipe generation, those are the questions that actually determine whether a model output is usable.

## Final Thoughts

The notebooks implement more than an evaluation script. They define an evaluation philosophy: measure recipes the way recipes are actually used. That means checking not just similarity, but usability, grounding, sequencing, completeness, and culinary soundness.

By splitting the process into a metric-first stage and a judge-first interpretation stage, the pipeline stays both scalable and insightful. It can process large runs automatically, but it still produces the kind of nuanced feedback needed to improve a recipe generation model in a disciplined way.

If you are fine-tuning or benchmarking a recipe model, this is the kind of evaluation setup worth building: one that does not mistake fluent text for a good recipe.
