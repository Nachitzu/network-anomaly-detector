"""Reproducible end-to-end comparison run against the real CICIDS2017 dataset.

Entry point that wires the whole pipeline together -- load the full CSVs from
`data/raw/`, run the Phase 1 feature engineering, fit both detectors, and
write the final side-by-side metrics table plus a written conclusion to
`docs/comparison_results.md` (README section 4's documented artifact).

Everything reusable lives in `src/` (README section 4): this module only
orchestrates already-tested public APIs -- `load_flows`, `prepare_dataset`,
`compare_models`, `render_comparison_table`, `explain_flagged_anomalies` --
and adds no detection or metric logic of its own.

Run it with the full dataset present under `data/raw/`:

    python -m src.evaluation.run_comparison
"""

from __future__ import annotations

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.feature_engineering import (
    Config,
    PreparedData,
    load_config,
    prepare_dataset,
    required_columns,
)
from src.data.loader import load_flows
from src.evaluation.compare_models import compare_models, render_comparison_table
from src.evaluation.explainability import explain_flagged_anomalies
from src.models.autoencoder_model import AutoencoderDetector
from src.models.base_detector import BaseDetector
from src.models.isolation_forest_model import IsolationForestDetector

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "docs" / "comparison_results.md"

# How many test rows to scan for the report's illustrative explanations.
# Explaining them is vectorized and cheap, but the returned dict holds one
# Python list per flagged row -- on a full run that would be hundreds of
# thousands of objects to print five of. This is a bound on the size of the
# result, not a workaround for slow explanation.
EXPLAIN_PREVIEW_ROWS = 50_000


def build_detectors(config: Config) -> list[BaseDetector]:
    """Instantiate both detectors from `config`, typed only as `BaseDetector`.

    Returned as `list[BaseDetector]` on purpose: everything downstream must
    work through the shared interface, never model-specific internals.
    """
    return [
        IsolationForestDetector(config.models.isolation_forest),
        AutoencoderDetector(config.models.autoencoder),
    ]


def load_prepared_dataset(config_dict: dict[str, Any], raw_dir: Path) -> PreparedData:
    """Load every CSV under `raw_dir` and run the Phase 1 pipeline over it.

    Only the columns the pipeline actually consumes are read (see
    `required_columns`): on the full CICIDS2017 that is 18 of 79, which is the
    difference between a multi-gigabyte peak and a comfortable one.
    """
    config = Config.model_validate(config_dict)
    data_section = config_dict.get("data", {})
    csv_glob = data_section.get("csv_glob", "*.csv") if isinstance(data_section, dict) else "*.csv"
    flows = load_flows(raw_dir, str(csv_glob), columns=required_columns(config))
    return prepare_dataset(flows, config_dict)


def recall_by_attack_type(
    flagged: np.ndarray, y_test: pd.Series, benign_label: str
) -> pd.Series:
    """Per-attack-type detection rate (recall) from already-computed decisions.

    Aggregate recall hides that a percentile threshold catches high-volume
    floods far more readily than low-and-slow attacks, so break it out by the
    raw ground-truth label. Labels are used here for evaluation only.

    Takes the decisions rather than a detector on purpose: this is pure
    aggregation, and asking for a detector would mean scoring the test set yet
    again, on top of the passes `compare_models` has already made.

    Args:
        flagged: Boolean decision per row of the test set.
        y_test: Raw ground-truth labels aligned with `flagged`.
        benign_label: The label value identifying normal traffic.
    """
    attack_mask = np.asarray(y_test != benign_label)
    return (
        pd.DataFrame({"label": y_test[attack_mask], "flagged": flagged[attack_mask]})
        .groupby("label")["flagged"]
        .mean()
        .sort_values(ascending=False)
    )


def _series_to_markdown(series: pd.Series, index_header: str, value_header: str) -> str:
    """Render a `pd.Series` as a two-column GitHub-flavored Markdown table."""
    lines = [f"| {index_header} | {value_header} |", "| --- | --- |"]
    lines += [f"| {index} | {value:.4f} |" for index, value in series.items()]
    return "\n".join(lines) + "\n"


def _run_context_section(
    prepared: PreparedData,
    y_test: pd.Series,
    benign_label: str,
    elapsed_seconds: float,
) -> list[str]:
    """The "Run context" block: what data produced the numbers below it."""
    n_attack = int((y_test != benign_label).sum())
    n_benign = len(y_test) - n_attack
    generated_on = datetime.now(tz=UTC).date().isoformat()

    return [
        "# Model comparison results",
        "",
        (
            "> Generated by `python -m src.evaluation.run_comparison`"
            f" on {generated_on}."
        ),
        "> Every number below comes from a single run over the full CICIDS2017 dataset,",
        "> not from the synthetic test fixture.",
        "",
        "## Run context",
        "",
        (
            f"- **Training set (benign only):** {prepared.x_train.shape[0]:,} flows"
            f" x {prepared.x_train.shape[1]} features"
        ),
        (
            f"- **Test set (mixed, labeled):** {len(y_test):,} flows"
            f" ({n_benign:,} benign / {n_attack:,} attack,"
            f" {n_attack / len(y_test):.2%} attack)"
        ),
        (
            "- **Labels:** used at evaluation time only; both models were fitted"
            " on benign traffic with no labels."
        ),
        (
            "- **Total wall clock (fit + evaluate, both models):**"
            f" {elapsed_seconds:.1f}s on CPU."
        ),
        "",
    ]


def _metrics_section(comparison: pd.DataFrame, table_markdown: str) -> list[str]:
    """The side-by-side metrics table plus a one-line reading of it."""
    best_f1 = str(comparison.loc["f1"].idxmax())
    best_auc = str(comparison.loc["roc_auc"].idxmax())
    verdict = (
        f"`{best_f1}` wins on both F1 and ROC-AUC"
        if best_f1 == best_auc
        else f"`{best_f1}` wins on F1, `{best_auc}` on ROC-AUC"
    )

    return [
        "## Side-by-side metrics",
        "",
        table_markdown.rstrip(),
        "",
        (
            f"{verdict} -- the latter being the threshold-independent measure of"
            " how well each model separates attack from benign traffic."
        ),
        "",
    ]


def _per_attack_section(
    flags_by_detector: dict[str, np.ndarray],
    y_test: pd.Series,
    benign_label: str,
) -> list[str]:
    """Per-attack-type recall for each detector, from decisions made once."""
    sections = [
        "## Detection rate by attack type",
        "",
        (
            "Aggregate recall hides a large spread: high-volume floods are caught"
            " far more readily than low-and-slow or low-volume attacks."
        ),
        "",
    ]
    for name, flagged in flags_by_detector.items():
        per_type = recall_by_attack_type(flagged, y_test, benign_label)
        sections += [
            f"### {name}",
            "",
            _series_to_markdown(per_type, "attack type", "recall").rstrip(),
            "",
        ]
    return sections


def _explanations_section(
    detectors: list[BaseDetector], x_test: np.ndarray, feature_names: list[str]
) -> list[str]:
    """A short, illustrative sample of per-anomaly explanations."""
    sample_x = x_test[:EXPLAIN_PREVIEW_ROWS]
    sections = [
        "## Example explanations",
        "",
        (
            "Top contributing features for the first flagged flows within the first"
            f" {EXPLAIN_PREVIEW_ROWS:,} test rows, via the shared"
            " `top_contributing_features` interface."
        ),
        "",
    ]
    for detector in detectors:
        explanations = explain_flagged_anomalies(detector, sample_x, feature_names, k=3)
        preview = list(explanations.items())[:5]
        sections += [f"### {detector.name}", ""]
        sections += (
            [f"- row `{row}`: {', '.join(features)}" for row, features in preview]
            if preview
            else ["- (no flows flagged within the sampled rows)"]
        )
        sections += [""]
    return sections


def _interpretation_section() -> list[str]:
    """Why recall is capped by design, and which column is threshold-free."""
    return [
        "## Reading these numbers",
        "",
        (
            "Both detectors set their threshold at the 95th percentile of *benign*"
            " scores, so roughly 5% of benign traffic is flagged by construction."
            " That fixes the false-positive budget up front and caps recall in"
            " exchange for high precision -- the intended trade-off for a triage"
            " feed, where analyst time is the scarce resource. Lowering the"
            " percentile raises recall and costs precision; the ROC-AUC column is"
            " the threshold-free comparison."
        ),
        "",
    ]


def _build_report(
    comparison: pd.DataFrame,
    table_markdown: str,
    prepared: PreparedData,
    detectors: list[BaseDetector],
    y_test: pd.Series,
    benign_label: str,
    elapsed_seconds: float,
) -> str:
    """Compose the full `comparison_results.md` document."""
    # Decide once per detector, then reuse: the per-attack breakdown and the
    # metrics table are two views of the same decisions, not a reason to score
    # a million-row test set twice more.
    flags_by_detector = {
        detector.name: detector.is_anomaly(prepared.x_test) for detector in detectors
    }

    sections = _run_context_section(prepared, y_test, benign_label, elapsed_seconds)
    sections += _metrics_section(comparison, table_markdown)
    sections += _per_attack_section(flags_by_detector, y_test, benign_label)
    sections += _explanations_section(detectors, prepared.x_test, prepared.feature_names)
    sections += _interpretation_section()
    return "\n".join(sections)


def run(config_path: Path, raw_dir: Path | None, output_path: Path) -> pd.DataFrame:
    """Execute the full comparison and write the Markdown report.

    Returns:
        The side-by-side comparison DataFrame, for programmatic use.
    """
    config_dict = load_config(config_path)
    config = Config.model_validate(config_dict)

    data_section = config_dict.get("data", {})
    configured_raw = (
        data_section.get("raw_dir", "data/raw") if isinstance(data_section, dict) else "data/raw"
    )
    resolved_raw_dir = raw_dir if raw_dir is not None else REPO_ROOT / str(configured_raw)

    print(f"Loading flows from {resolved_raw_dir} ...", flush=True)
    prepared = load_prepared_dataset(config_dict, resolved_raw_dir)
    print(f"  train {prepared.x_train.shape}  test {prepared.x_test.shape}", flush=True)

    benign_label = config.features.benign_label
    # Already normalized by `prepare_dataset`; no cleanup belongs here.
    y_test = prepared.y_test
    detectors = build_detectors(config)

    print("Fitting and evaluating both detectors ...", flush=True)
    started = time.perf_counter()
    comparison = compare_models(
        detectors, prepared.x_train, prepared.x_test, y_test, benign_label=benign_label
    )
    elapsed_seconds = time.perf_counter() - started
    print(comparison.to_string(), flush=True)

    table_path = render_comparison_table(comparison, output_path)
    table_markdown = table_path.read_text(encoding="utf-8")
    report = _build_report(
        comparison,
        table_markdown,
        prepared,
        detectors,
        y_test,
        benign_label,
        elapsed_seconds,
    )
    output_path.write_text(report, encoding="utf-8")
    print(f"\nWrote {output_path}", flush=True)
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Override the CSV directory (defaults to config.yaml's data.raw_dir).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    run(args.config, args.raw_dir, args.output)


if __name__ == "__main__":
    main()
