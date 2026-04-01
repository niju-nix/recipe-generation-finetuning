# Exploratory Analysis of the RecipeNLG Dataset

This report summarizes the exploratory analysis carried out across the following notebook iterations:

- `notebooks/01_data_exploration.ipynb`
- `notebooks/02_dataset_cleanup.ipynb`

The analysis evolved from basic dataset inspection into a full training-readiness workflow. Rather than stopping at summary statistics, the notebooks were used to examine data quality, normalize recipe text, detect duplicates and near-duplicates, define filtering rules, and construct leakage-aware dataset splits for fine-tuning.

## Why This Analysis Was Needed

The RecipeNLG dataset is rich and diverse, but the raw data is not immediately suitable for model training. The notebooks were used to answer a set of practical questions:

- Are recipe titles unique and reliable identifiers?
- How noisy are the ingredients and directions fields?
- What does a typical recipe look like in terms of ingredient count and step count?
- How many suspicious or malformed examples are present?
- How much duplication or near-duplication exists?
- How should the dataset be cleaned and split for downstream fine-tuning?

This made the exploratory analysis both descriptive and operational. The outcome was not only a better understanding of the dataset, but also a concrete set of preprocessing decisions grounded in evidence.

## 1. Initial Inspection of the Raw Dataset

The earliest notebook version focused on understanding the overall structure of the dataframe and inspecting the key fields such as `title`, `ingredients`, `directions`, and `NER`.

One of the first checks performed was on recipe title frequency:

```python
df['title'].value_counts()
df[df['title'] == 'Chicken Casserole'].head()
```

This immediately showed that recipe titles were not unique. Multiple rows could share the same title while containing different ingredient lists and cooking directions. That was an important early finding because it meant a title could not be treated as a unique recipe identifier. It also suggested that random train/test splitting could leak highly similar recipes across splits.

## 2. Ingredient Frequency Analysis

The notebooks then examined the ingredient space by extracting all ingredients from the `NER` column and counting their frequency across the dataset.

```python
all_ingredients = []

for ingredients_list in df['NER'].dropna():
    try:
        ingredients = eval(ingredients_list) if isinstance(ingredients_list, str) else ingredients_list
        all_ingredients.extend(ingredients)
    except:
        continue

from collections import Counter
ingredient_counts = Counter(all_ingredients)
ingredient_df = pd.DataFrame(ingredient_counts.items(), columns=['Ingredient', 'Count'])
ingredient_df = ingredient_df.sort_values('Count', ascending=False).reset_index(drop=True)
```

This step served several purposes. It showed which ingredients appeared most often, how broad the ingredient vocabulary was, and how much inconsistency existed in ingredient naming. It also highlighted the need for normalization, since similar ingredients often appeared under multiple forms due to plurals, abbreviations, punctuation, and unit variants.

In the earliest iteration, the notebooks even experimented with lightweight singularization rules to reduce variation in ingredient labels. This was a simple but useful way to test whether the ingredient vocabulary could be made more consistent before moving to more robust normalization.

## 3. Profiling Recipe Length and Structure

Another major part of the analysis focused on understanding the size and structure of each recipe. This was done in two ways:

- by counting the number of ingredients and directions per recipe,
- by estimating rough word counts for ingredient lists and cooking steps.

Representative logic looked like this:

```python
def NER_count(list_str):
    list_str = eval(list_str)
    return len(list_str)

df['ingredients_count'] = df['NER'].apply(NER_count)
df['directions_count'] = df['directions'].apply(NER_count)
```

The goal was to identify what a typical recipe looked like and, more importantly, what kinds of entries should be treated as suspicious.

This analysis showed that some recipes were clearly incomplete, containing only a handful of ingredients or very few steps. Others were excessively long, often indicating malformed rows, unusual formatting, or examples that would be poor candidates for supervised fine-tuning.

Based on that profiling, the notebooks repeatedly explored practical thresholds such as:

```python
min_ingredients = 3
max_ingredients = 50

min_steps = 3
max_steps = 30
```

These thresholds were not chosen blindly. They were informed by repeatedly inspecting outlier rows and checking whether the examples looked like legitimate recipes or dataset noise.

## 4. Manual Review of Outliers and Suspicious Examples

A useful feature of the analysis was that it did not rely only on numeric summaries. The notebooks included helper functions to print ingredients and directions in a readable format so that unusual examples could be manually reviewed.

```python
def beautify_print_ingredients(ingredients_str):
    try:
        ingredients_list = eval(ingredients_str) if isinstance(ingredients_str, str) else ingredients_str
        print("Ingredients:")
        for i, ingredient in enumerate(ingredients_list, 1):
            print(f"{i}. {ingredient}")
    except Exception as e:
        print("Error formatting ingredients:", e)
```

This made it possible to inspect recipes with unusually high ingredient counts, suspicious titles, or odd direction structures. In later versions, the notebooks also searched for rows with suspicious words in the title, such as `directions`, indicating possible contamination or malformed metadata.

This combination of quantitative filtering and manual inspection was important because it prevented the cleaning process from becoming purely mechanical. It ensured that filtering decisions remained tied to the actual content of the dataset.

## 5. Ingredient Normalization as a Core Analysis Track

As the notebooks matured, ingredient normalization became one of the most important parts of the exploratory workflow. The analysis made it clear that ingredient strings contained substantial surface-level noise:

- unit abbreviations such as `c`, `tbsp`, and `tsp`,
- unicode fractions,
- inconsistent punctuation,
- optional notes and parenthetical comments,
- line formatting differences,
- repeated whitespace and special characters.

To address this, the notebooks introduced canonical unit mappings:

```python
UNIT_VARIATIONS = {
    "cup": ["c", "c.", "cup", "cups"],
    "teaspoon": ["tsp", "tsp.", "teaspoon", "teaspoons", "t"],
    "tablespoon": ["tbsp", "tbsp.", "tablespoon", "tablespoons", "tbl"],
}
```

and regex-based normalization rules:

```python
UNIT_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(u) for u in UNIT_MAP.keys()) + r")\b",
    flags=re.IGNORECASE
)

def normalize_units_advanced(text):
    def replace_unit(match):
        raw = match.group(1).lower().replace(".", "")
        return UNIT_MAP.get(raw, raw)
    return UNIT_PATTERN.sub(replace_unit, text)
```

Later notebooks expanded this into a more complete normalization pipeline that handled unicode fractions, noise phrases, punctuation cleanup, and robust parsing using `ast.literal_eval` instead of `eval`.

This stage of the analysis showed that much of the variability in ingredient text was superficial rather than semantic. Once normalized, the ingredient field became much more useful for counting, clustering, exporting, and prompting.

## 6. Title and Direction Normalization

The notebooks also identified inconsistencies in titles and directions. Titles sometimes contained malformed parentheses, stray punctuation, casing inconsistencies, or unicode artifacts. Directions could appear as lists or strings and often needed flattening into a single output field for instruction tuning.

Representative functions included:

```python
def normalize_title(title: str) -> str:
    s = unidecode.unidecode(title).strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"^[\(\)]+(?=\w)", "", s).strip()
    s = re.sub(r"[\(\)]+$", "", s).strip()
    s = re.sub(r"[!?.,;:]+$", "", s)
    return s.title()
```

```python
def normalize_directions(directions):
    if isinstance(directions, list):
        steps = [unidecode.unidecode(str(s)).strip() for s in directions if str(s).strip()]
        return " ".join(steps)

    s = unidecode.unidecode(str(directions)).strip()
    s = re.sub(r"\s+", " ", s)
    return s
```

This part of the analysis improved overall consistency and made the dataset easier to export into prompt-completion training format.

## 7. Building Better Model Inputs from Ingredients

In `02_dataset_cleanup.ipynb`, the ingredient analysis moved one step further by transforming normalized ingredient lists into clean bullet-style inputs. This was useful not only for readability but also for prompt design.

```python
def ingredients_to_bullets(raw: str, bullet: str = "-", names_only: bool = False) -> str:
    ...
    return "\n".join(bullets)
```

This made it possible to compare multiple representations of the same recipe input:

- raw ingredients,
- normalized ingredient lists,
- bullet-formatted ingredient strings,
- names-only ingredient lists.

That comparison helped determine which input format would be most readable and useful for instruction tuning.

## 8. Formal Suspicious-Recipe Filtering

By the time the analysis reached `02_dataset_cleanup.ipynb`, the earlier observations had been turned into explicit dataset-quality filters. Rather than checking outliers manually each time, the notebooks defined suspicious-example rules directly in code:

```python
def filter_suspicious(df):
    df = df[df["ingredients_normalized"].apply(lambda x: isinstance(x, list) and len(x) > 3)]
    df = df[df["ingredients_normalized"].apply(lambda x: len(x) <= 100)]
    df = df[df["directions_normalized"].str.len() > 10]
    df = df[df["directions_normalized"].str.len() < 10000]
    df = df[(df["input_tokens"] >= 5)]
    df = df[(df["output_tokens"] >= 10)]
    df = df[(df["output_tokens"] <= 1500)]
    return df
```

This formalized the notion of a “suspicious” recipe. Examples with too few ingredients, unrealistic output length, malformed text, or implausible token counts could now be filtered systematically.

At this stage, the analysis had clearly transitioned from open-ended exploration to quality-controlled dataset curation.

## 9. Detecting Duplicates and Near-Duplicates

One of the most important later insights was that duplicate detection could not rely on titles alone. Since common titles like “Chicken Casserole” appeared many times, often with different ingredient lists, a more robust strategy was needed.

The cleanup notebook introduced blocked MinHash clustering over normalized ingredients. Titles were first normalized and grouped using title prefixes, while highly generic titles were handled differently. Ingredient overlap was then used to construct similarity signatures and assign recipes to ingredient-based clusters.

Representative logic included:

```python
def title_prefix_key(title_norm, k=5):
    toks = (title_norm or "").split()
    return " ".join(toks[:k])

def ingredient_signature(ingredients_norm, k=5):
    toks = [x for x in (ingredients_norm or []) if x not in COMMON_ING]
    toks.sort()
    return "|".join(toks[:k])
```

This clustering-based analysis made it possible to identify near-duplicate recipe groups even when the raw titles were too generic or too noisy to be reliable. It also revealed the extent of redundancy in the dataset and laid the foundation for leakage-aware train/test splitting.

## 10. Cluster Diagnostics

After clustering, the notebooks profiled the resulting cluster sizes and manually inspected large clusters. This made it possible to answer questions such as:

- Are most recipes unique after normalization?
- How many large duplicate families exist?
- Are the similarity thresholds too loose or too strict?

The notebooks also performed targeted validation on repeated titles such as “Chicken Casserole” to determine how many distinct ingredient clusters existed under the same title.

This confirmed a key conclusion of the analysis: title duplication was real, but ingredient-based clustering provided a much better signal of semantic similarity.

## 11. Train/Validation/Test Splitting Without Leakage

Earlier notebook versions used standard random splitting, which is common but risky in recipe datasets with repeated or near-duplicate examples.

Once clustering was available, the split strategy changed. `02_dataset_cleanup.ipynb` used group-based splitting with `ingredient_cluster_id` as the grouping key. This ensured that similar recipes remained in the same split and did not leak into validation or test sets.

This was a major improvement in dataset design because it made evaluation more trustworthy. The notebooks also explicitly checked that there was:

- no ID overlap between train, validation, and test,
- no cluster overlap between train, validation, and test.

In practical terms, this meant that the model would be evaluated on genuinely unseen recipe families rather than memorized variants of recipes already present in the training set.

## 12. Token Length Analysis for Training Readiness

The later notebook versions also evaluated the dataset from a model-efficiency perspective by measuring input and output token lengths with a target tokenizer.

```python
def get_length(example):
    input_length = len(tokenizer(example["input"])["input_ids"])
    output_length = len(tokenizer(example["output"])["input_ids"])
    return {
        "input_token_length": input_length,
        "output_token_length": output_length,
        "token_ratio": output_length / input_length
    }
```

From these measurements, the notebooks computed token-length percentiles, mean lengths, maxima, and output-to-input ratios. This helped estimate realistic sequence lengths for fine-tuning and identify extreme long-tail examples that would waste memory or destabilize training.

This part of the analysis connected dataset quality directly to model efficiency, which is especially important when training on constrained hardware or fixed context windows.

## 13. Creation of a High-Quality “Gold” Subset

The most refined stage of the analysis introduced a “gold” filtering track. This subset was built using token-based heuristics designed to preserve well-formed examples while removing noisy or impractical ones.

Representative criteria included:

- minimum input token length,
- minimum and maximum output token length,
- maximum output-to-input token ratio,
- maximum total token budget.

In code, this looked like:

```python
gold_pass = (
    in_len >= 20 and
    60 <= out_len <= 800 and
    ratio <= 6 and
    total <= 1024
)
```

This created a higher-quality dataset subset better aligned with instruction-tuning objectives. It also showed how exploratory analysis can evolve into a robust data selection framework rather than remaining a one-time inspection step.

## 14. Key Findings

Taken together, the notebook series produced several important findings:

- Recipe titles are not reliable unique identifiers.
- Ingredient text contains substantial formatting and normalization noise.
- The `NER` field is useful for structural analysis, especially ingredient counting.
- Recipes with very low or very high ingredient or step counts are often poor-quality examples.
- Duplicate and near-duplicate recipes are common enough to affect dataset splitting.
- Ingredient-based similarity is more useful than title matching for deduplication.
- Token length analysis is essential for selecting practical training examples.
- A higher-quality “gold” subset can be defined using structural and token-based rules.

## 15. Conclusion

The exploratory analysis across these notebook versions transformed the RecipeNLG dataset from a raw corpus into a curated training resource. What began as basic inspection of titles and ingredient counts gradually developed into a full dataset-quality pipeline covering normalization, anomaly detection, similarity clustering, leakage-aware splitting, and token-budget filtering.

This process was valuable not only for understanding the data, but also for shaping the final fine-tuning dataset. In effect, the exploratory analysis served as the foundation for every later preprocessing decision.

For recipe-generation fine-tuning, this was the most important outcome: the notebooks did not merely describe the dataset, they defined what a clean, usable, and evaluation-safe version of the dataset should look like.
