# Network Anomaly Detector

> ML-based network anomaly detection comparing two unsupervised approaches — **Isolation Forest** vs. **Autoencoder** — on labeled network flow data, with explainability and a rigorous, reproducible evaluation methodology.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

**Status:** In development — Phase 1 (data & feature engineering pipeline) complete.
**Author:** Ignacio Núñez — AI Security Engineer (Blue Team + ML for Security).
**Companion project:** [ai-soc-triage-agent](https://github.com/Nachitzu/ai-soc-triage-agent) — standalone for now, designed for future integration.

---

## 1. Problem statement

Rule-based and LLM-based detection (see the companion `ai-soc-triage-agent`) both depend on
either predefined signatures or an alert already having been raised. This project asks a
different question: **can we detect anomalous network behavior directly from flow statistics,
with no rules and no labels at training time?**

Two unsupervised approaches are trained and rigorously compared:

- **Isolation Forest** — tree-based, fast, an industry-standard baseline.
- **Autoencoder** — a neural network that captures non-linear feature relationships via
  reconstruction error.

The goal is not "which model wins" in the abstract — it is a **data-backed engineering
recommendation** (accuracy vs. training cost vs. explainability) suitable for a real SOC
decision. The comparison methodology and the conclusion matter as much as the models themselves.

## 2. Design principles

These constraints shape the whole codebase and are enforced in tests:

- **Unsupervised by construction.** Both models train **only on benign (normal) traffic**.
  Ground-truth labels are used **exclusively at evaluation time**, never during training —
  verified by an explicit no-label-leakage test.
- **One shared interface.** Both detectors implement the same `BaseDetector` interface, so the
  evaluation code treats them polymorphically and never special-cases a model.
- **Validated, integration-ready outputs.** Every anomaly output validates against a Pydantic
  schema whose field names mirror the `NormalizedAlert` schema in `ai-soc-triage-agent`, so
  future integration needs no rewrite.
- **Configuration over hardcoding.** Feature list, split ratios, and model hyperparameters live
  in `config.yaml`, not in Python.
- **Reproducible and tested.** Python 3.11+, full type hints, and a fast pytest suite that runs
  on a tiny synthetic fixture (never the full CICIDS2017 CSVs), targeting ≥ 80% coverage.

## 3. Architecture

```
┌───────────────────────┐
│      CICIDS2017        │   Labeled network flow dataset
│  (flow records)        │   (reused from ai-soc-triage-agent)
└───────────┬───────────┘
            │
            ▼
┌────────────────────────────────┐
│      Feature engineering        │   Scaling, feature selection,
│  (src/data/feature_engineering) │   train (benign-only) / test split
└───────────┬────────────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌───────────┐ ┌───────────┐
│Isolation  │ │Autoencoder│    Trained independently,
│Forest     │ │(PyTorch)  │    both implement BaseDetector
│(sklearn)  │ │           │
└─────┬─────┘ └─────┬─────┘
      │             │
      └──────┬──────┘
             ▼
┌─────────────────────────────────┐
│    Comparative evaluation        │   Precision, recall, F1, ROC-AUC,
│    (src/evaluation/compare)      │   training / inference latency
└───────────┬─────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│           Anomaly Report                   │   flow_id, anomaly_score,
│     (Pydantic-validated JSON)              │   model_used, top contributing
│                                            │   features
└──────────────────────────────────────────┘
```

### Data flow

1. **Dataset** — CICIDS2017 flow records (same source as `ai-soc-triage-agent`, reused for
   narrative and methodological consistency).
2. **Feature engineering** — select numeric flow features, scale them, and split into a
   **benign-only training set** and a **mixed (benign + attack) test set** with ground-truth
   labels retained for evaluation only.
3. **Two models, same interface** — Isolation Forest and Autoencoder each implement
   `BaseDetector.fit(X)` / `BaseDetector.score(X)`. Neither model sees labels during training.
4. **Evaluation** — labels are used only here, to compute precision / recall / F1 / ROC-AUC and
   to compare training and inference latency between the two models.
5. **Explainability** — for each flagged anomaly, identify the top contributing features
   (feature deviations for Isolation Forest, per-feature reconstruction error for the Autoencoder).
6. **Output** — `AnomalyReport` objects, Pydantic-validated and schema-compatible with future
   integration into the triage agent.

## 4. Repository structure

```
network-anomaly-detector/
├── README.md                          ← this file (project spec)
├── config.yaml                        ← model hyperparameters, feature list, paths
├── pyproject.toml                     ← dependencies and project metadata
├── src/
│   ├── data/
│   │   ├── loader.py                  ← loads CICIDS2017 CSVs
│   │   └── feature_engineering.py     ← scaling, feature selection, benign/attack split
│   ├── models/
│   │   ├── base_detector.py           ← abstract interface: fit(X), score(X), name
│   │   ├── isolation_forest_model.py  ← implements BaseDetector
│   │   └── autoencoder_model.py       ← implements BaseDetector (PyTorch nn.Module inside)
│   ├── evaluation/
│   │   ├── compare_models.py          ← runs both models, computes shared metrics
│   │   └── explainability.py          ← top-N contributing features per anomaly
│   └── schemas/
│       └── anomaly_report.py          ← Pydantic output schema
├── data/
│   ├── raw/                           ← full dataset (git-ignored)
│   └── samples/                       ← small labeled subset (committed, reproducible)
├── models/                            ← trained artifacts (.pkl, .pt) — git-ignored
├── notebooks/                         ← exploration & reporting (01_eda … 04_model_comparison)
├── tests/                             ← pytest suite (fixture-based, fast, no GPU/large data)
└── docs/
    └── comparison_results.md          ← final metrics table + written conclusion
```

> Notebooks are for exploration and reporting only; any reusable logic lives in `src/`.

## 5. Design details

### 5.1 `BaseDetector` interface

```python
from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    name: str

    @abstractmethod
    def fit(self, X_benign: np.ndarray) -> None:
        """Train using ONLY benign traffic. No labels involved."""

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return an anomaly score per row. Higher = more anomalous.
        Comparable across models after normalization in evaluation."""

    @abstractmethod
    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        """Explainability hook: which features drove this specific anomaly score."""
```

Both `IsolationForestDetector` and `AutoencoderDetector` implement this exact interface, and the
comparison code depends only on it — never on model-specific internals.

### 5.2 Isolation Forest

- `sklearn.ensemble.IsolationForest`, with `n_estimators`, `contamination`, and `max_samples`
  sourced from `config.yaml`.
- `score()` returns the negative of `decision_function`, so higher means more anomalous
  (consistent sign convention with the Autoencoder).
- `top_contributing_features()` approximates contribution via per-feature deviation from the
  training set's mean/std, ranked by z-score magnitude. (SHAP support is a future enhancement.)

### 5.3 Autoencoder

- A simple feedforward encoder-decoder in PyTorch (dimensions tunable via `config.yaml`; a
  reasonable starting point is `input_dim → 32 → 16 → 8 → 16 → 32 → input_dim`, ReLU activations,
  MSE reconstruction loss).
- Trained only on benign traffic; the anomaly score is the per-sample reconstruction error.
- The anomaly threshold is a configurable percentile (default: 95th) of the reconstruction error
  on a held-out benign validation split — never computed from attack data.
- `top_contributing_features()` ranks features by per-feature squared reconstruction error.

### 5.4 `AnomalyReport` schema

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Literal

class AnomalyReport(BaseModel):
    flow_id: str
    timestamp: datetime
    source_ip: str
    dest_ip: str
    model_used: Literal["isolation_forest", "autoencoder"]
    anomaly_score: float
    is_anomaly: bool                       # score above the model's configured threshold
    top_contributing_features: list[str] = Field(min_length=1, max_length=5)
    ground_truth_label: str | None = None  # populated only in evaluation, never at inference
```

Field names (`source_ip`, `dest_ip`, `timestamp`) intentionally match `NormalizedAlert` from
`ai-soc-triage-agent` for low-friction future integration.

### 5.5 Evaluation

For each model, computed against the labeled test set:

- Precision, recall, F1 (using each model's configured threshold).
- ROC-AUC (threshold-independent).
- Training time (wall clock, single run, same machine/session for a fair comparison).
- Inference latency (mean ms per sample, over a batch of at least 1000 samples).

The deliverable is a single side-by-side comparison table (a DataFrame, saved to
`docs/comparison_results.md`) — a direct comparison, not separate per-model printouts.

## 6. Dataset: CICIDS2017

- **Source:** Canadian Institute for Cybersecurity — <https://www.unb.ca/cic/datasets/ids-2017.html>.
- **Why reuse it:** consistency across the portfolio — the same dataset analyzed with two
  different techniques (LLM reasoning vs. classical/deep ML), giving a directly comparable narrative.
- **How it differs here:** this project uses the **numeric flow-level features directly** (flow
  duration, packets/sec, byte counts, TCP flag ratios, etc.) rather than converting rows into
  synthetic SIEM-style text alerts.
- **Split strategy:** training set = BENIGN-labeled rows only; test set = a mixed sample of
  BENIGN + each attack category, stratified so no single attack type dominates evaluation.
- **Handling:** full CSVs are git-ignored in `data/raw/`; a small, reproducible, class-balanced
  sample lives in `data/samples/`.

## 7. Getting started

Requires **Python 3.11+** and [uv](https://docs.astral.sh/uv/).

```bash
# install dependencies (including dev tools)
uv sync --extra dev

# run the test suite with coverage
uv run pytest -q --cov=src
```

Place the full CICIDS2017 CSVs under `data/raw/` (git-ignored) to run the full pipeline; the
committed sample under `data/samples/` is enough to exercise the code end to end.

## 8. Roadmap

**Phase 1 — Data & feature engineering** ✅
- CICIDS2017 loader, feature selection, `StandardScaler`, benign-only train / mixed labeled test
  split, non-finite value sanitization, and a typed configuration model.

**Phase 2 — Isolation Forest**
- `BaseDetector` interface and the `sklearn` implementation, with score inspection and sanity
  checks against known attack rows.

**Phase 3 — Autoencoder**
- PyTorch model implementing `BaseDetector`; training curves, reconstruction-error distribution,
  and threshold selection.

**Phase 4 — Comparison & explainability**
- Run both models over the same labeled test set, compute shared metrics, produce the top-N
  contributing features per anomaly, and the centerpiece side-by-side comparison notebook.

**Phase 5 — Documentation & polish**
- Finalize the `AnomalyReport` schema, fill the Results section with real numbers, and document
  limitations and future work.

**Future (out of scope for this repo)**
- Wire `AnomalyReport` output as a new alert source feeding `NormalizedAlert` in
  `ai-soc-triage-agent`.
- SHAP-based explainability for Isolation Forest (replacing the z-score approximation).
- Concept-drift monitoring and a periodic retraining pipeline.

## 9. Evaluation criteria (definition of done)

| Criterion | Target |
|-----------|--------|
| Both models implement the identical `BaseDetector` interface | Required — hard constraint |
| F1 score (each model, on the labeled test set) | Report actual value — exploratory |
| ROC-AUC (each model) | Report actual value |
| Comparison table with both models side by side | Required — `docs/comparison_results.md` |
| Written, reasoned conclusion (a recommendation and its tradeoffs) | Required |
| Unit-test coverage on `src/` | ≥ 80% |
| No label leakage into training | Required — hard constraint, asserted in tests |

**No label leakage is the non-negotiable check.** Any code path where the training set for either
model touches `ground_truth_label` before evaluation is a bug, not a stylistic choice — and it is
asserted explicitly in the test suite.

## 10. Tech stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.11+ | Consistency with `ai-soc-triage-agent` |
| Classical ML | scikit-learn | Isolation Forest, scaling, metrics |
| Deep learning | PyTorch | Autoencoder |
| Validation | Pydantic v2 | Strict output contracts, schema-compatible with the triage agent |
| Data | pandas | CICIDS2017 processing |
| Visualization | matplotlib, seaborn | Comparison plots (ROC curves, score distributions) |
| Config | PyYAML | Single source for hyperparameters |
| Testing | pytest | Fixture-based, fast, no GPU/large-dataset dependency |

## License

Released under the MIT License.
