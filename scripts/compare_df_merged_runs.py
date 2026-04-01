#!/usr/bin/env python
"""Compare df_merged parquet outputs across multiple runs.

Default usage compares three run folders under ../runs and writes outputs into
../runs/_comparisons/<timestamp>.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

SCORE_COLUMNS = [
    "overall_score",
    "ingredient_completeness",
    "instruction_clarity",
    "cooking_logic",
    "recipe_coherence",
    "practicality",
    "originality",
    "safety_accuracy",
    "reference_alignment",
]

TEXT_COLUMNS = ["title", "prediction", "reference"]


@dataclass
class RunData:
    run_id: str
    path: Path
    raw_df: pd.DataFrame
    dedup_df: pd.DataFrame
    row_count: int
    distinct_keys: int
    duplicate_keys: int
    key_null_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare df_merged parquet outputs across runs.")
    parser.add_argument(
        "--runs",
        nargs="+",
        default=["20260322T165311Z_base_run", "20260221T160905Z_LoRA_32", "20260221T171043Z_LoRA_16"],
        help="Run IDs to compare. Default compares the three selected runs.",
    )
    parser.add_argument(
        "--base-run",
        default=None,
        help="Base run ID for pairwise comparisons. Defaults to first run in --runs.",
    )
    parser.add_argument(
        "--runs-root",
        default="runs",
        help="Directory containing per-run folders.",
    )
    parser.add_argument(
        "--input-file",
        default="final_eval.parquet",
        help="Input parquet filename inside each run folder.",
    )
    parser.add_argument("--key", default="recipe_id", help="Primary key column.")
    parser.add_argument("--sample-n", type=int, default=30, help="Rows per sample export.")
    parser.add_argument(
        "--null-rate-drift-threshold",
        type=float,
        default=0.05,
        help="Alert threshold for null-rate drift on shared columns.",
    )
    parser.add_argument(
        "--low-overlap-threshold",
        type=float,
        default=0.70,
        help="Warn when pairwise key overlap (inner/base) is below this ratio.",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default: runs/_comparisons/<timestamp>",
    )
    return parser.parse_args()


def normalize_verdict(series: pd.Series) -> pd.Series:
    s = series.fillna("UNKNOWN").astype(str).str.strip().str.upper()
    return s.replace({"": "UNKNOWN"})


def safe_to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def load_run(run_id: str, runs_root: Path, input_file: str, key: str) -> RunData:
    path = runs_root / run_id / input_file
    if not path.exists():
        raise FileNotFoundError(f"Run '{run_id}' missing input file: {path}")

    df = pd.read_parquet(path)
    if key not in df.columns:
        raise KeyError(f"Run '{run_id}' missing required key column '{key}' in {path}")

    row_count = len(df)
    key_series = df[key]
    key_null_rate = float(key_series.isna().mean())
    distinct_keys = int(key_series.nunique(dropna=True))
    duplicate_keys = int(row_count - distinct_keys)

    # Policy: keep last row per key for deterministic pairwise comparisons.
    dedup_df = df.drop_duplicates(subset=[key], keep="last").copy()

    return RunData(
        run_id=run_id,
        path=path,
        raw_df=df,
        dedup_df=dedup_df,
        row_count=row_count,
        distinct_keys=distinct_keys,
        duplicate_keys=duplicate_keys,
        key_null_rate=key_null_rate,
    )


def build_schema_diff(runs: Sequence[RunData], null_drift_threshold: float) -> pd.DataFrame:
    records: List[Dict[str, object]] = []
    run_ids = [r.run_id for r in runs]
    all_columns = sorted({c for r in runs for c in r.raw_df.columns})

    for col in all_columns:
        per_run = []
        null_rates = {}
        dtypes = {}
        missing_in = []
        for r in runs:
            if col in r.raw_df.columns:
                dtype = str(r.raw_df[col].dtype)
                null_rate = float(r.raw_df[col].isna().mean())
                per_run.append(r.run_id)
                null_rates[r.run_id] = null_rate
                dtypes[r.run_id] = dtype
            else:
                missing_in.append(r.run_id)

        dtype_values = sorted(set(dtypes.values()))
        dtype_mismatch = len(dtype_values) > 1
        null_rate_spread = max(null_rates.values()) - min(null_rates.values()) if len(null_rates) > 1 else 0.0
        null_rate_drift_flag = bool(len(null_rates) > 1 and null_rate_spread > null_drift_threshold)

        row: Dict[str, object] = {
            "column": col,
            "present_in_runs": ",".join(per_run),
            "missing_in_runs": ",".join(missing_in),
            "dtype_mismatch": dtype_mismatch,
            "dtype_values": ",".join(dtype_values),
            "null_rate_spread": null_rate_spread,
            "null_rate_drift_flag": null_rate_drift_flag,
        }
        for rid in run_ids:
            row[f"dtype__{rid}"] = dtypes.get(rid)
            row[f"null_rate__{rid}"] = null_rates.get(rid)
        records.append(row)

    return pd.DataFrame.from_records(records).sort_values("column").reset_index(drop=True)


def build_run_level_summary(runs: Sequence[RunData], key: str) -> pd.DataFrame:
    summary_rows: List[Dict[str, object]] = []

    for run in runs:
        df = safe_to_numeric(run.raw_df, SCORE_COLUMNS)
        row: Dict[str, object] = {
            "run_id": run.run_id,
            "input_path": str(run.path),
            "row_count": run.row_count,
            "distinct_key_count": run.distinct_keys,
            "duplicate_key_count": run.duplicate_keys,
            "key_null_rate": run.key_null_rate,
            "key_coverage_ratio": float(run.distinct_keys / run.row_count) if run.row_count else 0.0,
        }

        if "overall_score" in df.columns:
            row["overall_score_mean"] = float(df["overall_score"].mean())
            row["overall_score_median"] = float(df["overall_score"].median())
            row["overall_score_std"] = float(df["overall_score"].std())
        else:
            row["overall_score_mean"] = np.nan
            row["overall_score_median"] = np.nan
            row["overall_score_std"] = np.nan

        if "verdict" in df.columns:
            verdict = normalize_verdict(df["verdict"])
            value_counts = verdict.value_counts(normalize=True)
            row["verdict_PASS_pct"] = float(value_counts.get("PASS", 0.0) * 100)
            row["verdict_FAIL_pct"] = float(value_counts.get("FAIL", 0.0) * 100)
            row["verdict_OTHER_pct"] = float((1.0 - value_counts.get("PASS", 0.0) - value_counts.get("FAIL", 0.0)) * 100)
        else:
            row["verdict_PASS_pct"] = np.nan
            row["verdict_FAIL_pct"] = np.nan
            row["verdict_OTHER_pct"] = np.nan

        for col in SCORE_COLUMNS:
            if col == "overall_score":
                continue
            row[f"mean__{col}"] = float(df[col].mean()) if col in df.columns else np.nan

        summary_rows.append(row)

    return pd.DataFrame(summary_rows).sort_values("run_id").reset_index(drop=True)


def build_pairwise(base: RunData, other: RunData, key: str, low_overlap_threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, object]]:
    base_df = base.dedup_df.copy()
    other_df = other.dedup_df.copy()

    if "verdict" in base_df.columns:
        base_df["verdict"] = normalize_verdict(base_df["verdict"])
    if "verdict" in other_df.columns:
        other_df["verdict"] = normalize_verdict(other_df["verdict"])

    merge_cols = sorted(set(base_df.columns).intersection(set(other_df.columns)))
    aligned = base_df[merge_cols].merge(other_df[merge_cols], on=key, how="inner", suffixes=(f"__{base.run_id}", f"__{other.run_id}"))

    overlap_ratio = float(len(aligned) / len(base_df)) if len(base_df) else 0.0

    metric_rows: List[Dict[str, object]] = []
    for metric in SCORE_COLUMNS:
        left = f"{metric}__{base.run_id}"
        right = f"{metric}__{other.run_id}"
        if left not in aligned.columns or right not in aligned.columns:
            continue

        lnum = pd.to_numeric(aligned[left], errors="coerce")
        rnum = pd.to_numeric(aligned[right], errors="coerce")
        delta = rnum - lnum
        abs_delta = delta.abs()

        metric_rows.append(
            {
                "base_run": base.run_id,
                "other_run": other.run_id,
                "metric": metric,
                "aligned_rows": int(len(aligned)),
                "valid_delta_rows": int(delta.notna().sum()),
                "mean_delta": float(delta.mean()),
                "median_delta": float(delta.median()),
                "std_delta": float(delta.std()),
                "mean_abs_delta": float(abs_delta.mean()),
                "pct_changed_nonzero": float((abs_delta > 1e-12).mean() * 100),
            }
        )

    verdict_rows: List[Dict[str, object]] = []
    bcol = f"verdict__{base.run_id}"
    ocol = f"verdict__{other.run_id}"
    if bcol in aligned.columns and ocol in aligned.columns:
        vc = aligned.groupby([bcol, ocol], dropna=False).size().reset_index(name="count")
        total = vc["count"].sum()
        for _, r in vc.iterrows():
            verdict_rows.append(
                {
                    "base_run": base.run_id,
                    "other_run": other.run_id,
                    "base_verdict": r[bcol],
                    "other_verdict": r[ocol],
                    "count": int(r["count"]),
                    "pct": float((r["count"] / total) * 100) if total else 0.0,
                }
            )

    health = {
        "base_run": base.run_id,
        "other_run": other.run_id,
        "base_unique_rows": int(len(base_df)),
        "other_unique_rows": int(len(other_df)),
        "aligned_rows": int(len(aligned)),
        "overlap_ratio_vs_base": overlap_ratio,
        "low_overlap_flag": overlap_ratio < low_overlap_threshold,
    }

    return (
        pd.DataFrame(metric_rows),
        pd.DataFrame(verdict_rows),
        aligned,
        health,
    )


def select_sample_columns(aligned: pd.DataFrame, base_run: str, other_run: str, key: str) -> List[str]:
    cols: List[str] = [key]
    for tcol in TEXT_COLUMNS:
        b = f"{tcol}__{base_run}"
        o = f"{tcol}__{other_run}"
        if b in aligned.columns:
            cols.append(b)
        if o in aligned.columns:
            cols.append(o)

    for metric in SCORE_COLUMNS:
        b = f"{metric}__{base_run}"
        o = f"{metric}__{other_run}"
        if b in aligned.columns:
            cols.append(b)
        if o in aligned.columns:
            cols.append(o)

    for v in (f"verdict__{base_run}", f"verdict__{other_run}"):
        if v in aligned.columns:
            cols.append(v)

    # Preserve order and uniqueness.
    dedup_cols = []
    seen = set()
    for c in cols:
        if c not in seen:
            dedup_cols.append(c)
            seen.add(c)
    return dedup_cols


def build_sampled_changed_rows(aligned: pd.DataFrame, base_run: str, other_run: str, key: str, sample_n: int) -> pd.DataFrame:
    if aligned.empty:
        return aligned.copy()

    working = aligned.copy()
    b_score = f"overall_score__{base_run}"
    o_score = f"overall_score__{other_run}"
    if b_score in working.columns and o_score in working.columns:
        working[b_score] = pd.to_numeric(working[b_score], errors="coerce")
        working[o_score] = pd.to_numeric(working[o_score], errors="coerce")
        working["overall_score_delta"] = working[o_score] - working[b_score]
        working["overall_score_abs_delta"] = working["overall_score_delta"].abs()
    else:
        working["overall_score_delta"] = np.nan
        working["overall_score_abs_delta"] = np.nan

    b_verdict = f"verdict__{base_run}"
    o_verdict = f"verdict__{other_run}"
    if b_verdict in working.columns and o_verdict in working.columns:
        working[b_verdict] = normalize_verdict(working[b_verdict])
        working[o_verdict] = normalize_verdict(working[o_verdict])
        working["verdict_flip"] = working[b_verdict] != working[o_verdict]
    else:
        working["verdict_flip"] = False

    part_improve = working.sort_values("overall_score_delta", ascending=False).head(sample_n)
    part_regress = working.sort_values("overall_score_delta", ascending=True).head(sample_n)
    part_flip = working[working["verdict_flip"]].sort_values("overall_score_abs_delta", ascending=False).head(sample_n)

    sampled = pd.concat([part_improve, part_regress, part_flip], ignore_index=True)
    sampled = sampled.drop_duplicates(subset=[key], keep="first") if key in sampled.columns else sampled

    selected_cols = select_sample_columns(sampled, base_run, other_run, key)
    extra_cols = ["overall_score_delta", "overall_score_abs_delta", "verdict_flip"]
    selected_cols.extend([c for c in extra_cols if c in sampled.columns])

    return sampled[selected_cols].sort_values("overall_score_abs_delta", ascending=False, na_position="last").reset_index(drop=True)


def to_markdown_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "(no rows)"
    view = df.head(max_rows)
    return view.to_markdown(index=False)


def generate_report(
    outdir: Path,
    runs: Sequence[RunData],
    args: argparse.Namespace,
    schema_df: pd.DataFrame,
    run_summary_df: pd.DataFrame,
    metric_delta_df: pd.DataFrame,
    verdict_shift_df: pd.DataFrame,
    pairwise_health: Sequence[Dict[str, object]],
    sample_files: Sequence[Path],
) -> str:
    warnings: List[str] = []
    dup_warnings = [r for r in runs if r.duplicate_keys > 0]
    for r in dup_warnings:
        warnings.append(
            f"- Duplicate key rows in `{r.run_id}`: {r.duplicate_keys} (dedup policy: keep last)."
        )

    null_drift_count = int(schema_df.get("null_rate_drift_flag", pd.Series(dtype=bool)).fillna(False).sum())
    if null_drift_count > 0:
        warnings.append(f"- Null-rate drift flagged on {null_drift_count} columns.")

    for h in pairwise_health:
        if bool(h.get("low_overlap_flag", False)):
            warnings.append(
                f"- Low overlap: `{h['base_run']}` vs `{h['other_run']}` = {h['overlap_ratio_vs_base']:.1%} (< {args.low_overlap_threshold:.0%})."
            )

    warning_block = "\n".join(warnings) if warnings else "- No major data health warnings detected."

    lines = [
        "# df_merged Run Comparison Report",
        "",
        f"Generated at: `{datetime.now(timezone.utc).isoformat()}`",
        "",
        "## Configuration",
        f"- Runs: `{', '.join(args.runs)}`",
        f"- Base run: `{args.base_run}`",
        f"- Runs root: `{Path(args.runs_root).resolve()}`",
        f"- Input file: `{args.input_file}`",
        f"- Key column: `{args.key}`",
        f"- Sample size per stratum: `{args.sample_n}`",
        "",
        "## Data Health Warnings",
        warning_block,
        "",
        "## Run-Level Summary",
        to_markdown_table(run_summary_df, max_rows=20),
        "",
        "## Schema Diff (top rows)",
        to_markdown_table(schema_df, max_rows=30),
        "",
        "## Pairwise Metric Delta",
        to_markdown_table(metric_delta_df, max_rows=50),
        "",
        "## Pairwise Verdict Shift",
        to_markdown_table(verdict_shift_df, max_rows=50),
        "",
        "## Pairwise Overlap Health",
        to_markdown_table(pd.DataFrame(pairwise_health), max_rows=20),
        "",
        "## Sample Files",
    ]

    if sample_files:
        for sf in sample_files:
            lines.append(f"- `{sf.name}`")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## Output Artifacts",
            "- `schema_diff.csv`",
            "- `run_level_summary.csv`",
            "- `pairwise_metric_delta.csv`",
            "- `pairwise_verdict_shift.csv`",
            "- `sampled_changed_rows_<base>_vs_<other>.csv`",
            "- `comparison_report.md`",
        ]
    )

    return "\n".join(lines)


def ensure_outdir(args: argparse.Namespace) -> Path:
    if args.outdir:
        outdir = Path(args.outdir)
    else:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        outdir = Path(args.runs_root) / "_comparisons" / ts
    outdir.mkdir(parents=True, exist_ok=True)
    return outdir


def main() -> int:
    args = parse_args()

    if len(args.runs) < 2:
        print("ERROR: provide at least two runs via --runs", file=sys.stderr)
        return 2

    if args.base_run is None:
        args.base_run = args.runs[0]

    if args.base_run not in args.runs:
        print("ERROR: --base-run must be one of --runs", file=sys.stderr)
        return 2

    runs_root = Path(args.runs_root)

    try:
        runs = [load_run(r, runs_root, args.input_file, args.key) for r in args.runs]
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    run_map = {r.run_id: r for r in runs}
    base = run_map[args.base_run]
    others = [r for r in runs if r.run_id != args.base_run]

    schema_df = build_schema_diff(runs, args.null_rate_drift_threshold)
    run_summary_df = build_run_level_summary(runs, args.key)

    metric_frames: List[pd.DataFrame] = []
    verdict_frames: List[pd.DataFrame] = []
    pairwise_health: List[Dict[str, object]] = []
    sample_files: List[Path] = []

    outdir = ensure_outdir(args)

    for other in others:
        metric_df, verdict_df, aligned_df, health = build_pairwise(base, other, args.key, args.low_overlap_threshold)
        metric_frames.append(metric_df)
        verdict_frames.append(verdict_df)
        pairwise_health.append(health)

        sample_df = build_sampled_changed_rows(aligned_df, base.run_id, other.run_id, args.key, args.sample_n)
        sample_name = f"sampled_changed_rows_{base.run_id}_vs_{other.run_id}.csv"
        sample_path = outdir / sample_name
        sample_df.to_csv(sample_path, index=False)
        sample_files.append(sample_path)

    metric_delta_df = pd.concat(metric_frames, ignore_index=True) if metric_frames else pd.DataFrame()
    verdict_shift_df = pd.concat(verdict_frames, ignore_index=True) if verdict_frames else pd.DataFrame()

    schema_df.to_csv(outdir / "schema_diff.csv", index=False)
    run_summary_df.to_csv(outdir / "run_level_summary.csv", index=False)
    metric_delta_df.to_csv(outdir / "pairwise_metric_delta.csv", index=False)
    verdict_shift_df.to_csv(outdir / "pairwise_verdict_shift.csv", index=False)

    report_text = generate_report(
        outdir=outdir,
        runs=runs,
        args=args,
        schema_df=schema_df,
        run_summary_df=run_summary_df,
        metric_delta_df=metric_delta_df,
        verdict_shift_df=verdict_shift_df,
        pairwise_health=pairwise_health,
        sample_files=sample_files,
    )
    (outdir / "comparison_report.md").write_text(report_text, encoding="utf-8")

    print(f"Wrote comparison outputs to: {outdir.resolve()}")
    print("Files:")
    for p in sorted(outdir.iterdir()):
        print(f"- {p.name}")

    low_overlap_pairs = [h for h in pairwise_health if h.get("low_overlap_flag")]
    if low_overlap_pairs:
        print("WARNING: low-overlap pairs detected:")
        for h in low_overlap_pairs:
            print(
                f"  {h['base_run']} vs {h['other_run']}: "
                f"{h['overlap_ratio_vs_base']:.1%} overlap vs base"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
