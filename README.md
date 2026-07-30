# network-anomaly-detector

> **ML-based network anomaly detection: Isolation Forest vs. Autoencoder** — trains and compares two unsupervised approaches on labeled network flow data, with explainability and a rigorous evaluation methodology. Designed to plug into `ai-soc-triage-agent` in a future phase.

**Status:** 🚧 Planning / initial development
**Author:** Ignacio Núñez — AI Security Engineer | Blue Team + ML for Security
**Companion project:** [ai-soc-triage-agent](https://github.com/Nachitzu/ai-soc-triage-agent) — standalone for now, designed for future integration

---

## 📌 Instructions for Claude Code

This README is the **single source of truth** for this project. When implementing:

1. Implement in the phase order defined in the Roadmap. Do not skip ahead — Phase 2 and 3 (the two models) depend on Phase 1's feature pipeline being stable and tested first.
2. Both models MUST share the same `BaseDetector` interface (`src/models/base_detector.py`) so `compare_models.py` can treat them polymorphically. Do not special-case one model in the evaluation code.
3. Both models train **only on normal (benign) traffic** — this is an unsupervised anomaly detection setup, not a supervised classifier. Labels are used only at evaluation time, never during training. This is a hard constraint, not a suggestion.
4. All anomaly outputs MUST validate against the Pydantic schema in `src/schemas/anomaly_report.py`.
5. Design `AnomalyReport` fields to be trivially mappable to the `NormalizedAlert` schema used in `ai-soc-triage-agent` (same field names for `source_ip`, `dest_ip`, `timestamp` where applicable) — this project stays standalone for now, but the schema should not require a rewrite to integrate later.
6. Use scikit-learn for Isolation Forest and PyTorch for the Autoencoder. Keep both models' hyperparameters in a single `config.yaml`, not hardcoded in Python.
7. Write code and comments in **English**. Type hints everywhere. Python 3.11+.
8. Every module gets unit tests (pytest). Model training tests use a tiny synthetic fixture dataset, never the full CICIDS2017 CSVs (too slow/large for CI).
9. Notebooks are for exploration and reporting only — no logic that other modules depend on should live exclusively in a notebook. If a notebook does something reusable, extract it into `src/`.

---

## 1. Problem statement

Rule-based and LLM-based detection (see `ai-soc-triage-agent`) both depend on either predefined signatures or an alert already having been raised. This project asks a different question: **can we detect anomalous network behavior directly from flow statistics, with no rules and no labels at training time?**

Two unsupervised approaches are trained and rigorously compared:

- **Isolation Forest** — tree-based, fast, industry-standard baseline.
- **Autoencoder** — neural network, captures non-linear feature relationships via reconstruction error.

**Design principle:** the goal is not "which model wins" in the abstract — it's producing a **data-backed engineering recommendation** (accuracy vs. training cost vs. explainability) suitable for a real SOC decision. The comparison methodology and conclusion matter as much as the models themselves.

---

## 2. Architecture

```
┌───────────────────────┐
│      CICIDS2017       │   Labeled network flow dataset
│  (flow records)       │   (reused from ai-soc-triage-agent)
└───────────┬────────────┘
            │
            ▼
┌────────────────────────────────┐
│      Feature engineering       │   Scaling, feature selection,
│  (src/data/feature_engineering)│   train (benign-only) / test split
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
│    Evaluación comparativa       │   Precision, recall, F1, ROC-AUC,
│    (src/evaluation/compare)     │   training/inference latency
└───────────┬──────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│           Anomaly Report                  │   flow_id, anomaly_score,
│     (Pydantic-validated JSON)             │   model_used, top contributing
│                                            │   features
└───────────┬────────────────────────────────┘
            │
            ▼ (future — out of scope for this repo)
┌──────────────────────────────────────────┐
│  Input candidate for ai-soc-triage-agent  │
└──────────────────────────────────────────┘
```

### Data flow

1. **Dataset** → CICIDS2017 flow records (same source as `ai-soc-triage-agent`, reused for narrative + methodological consistency).
2. **Feature engineering** → select numeric flow features, scale them, split into a **benign-only training set** and a **mixed (benign + attack) test set** with ground-truth labels retained for evaluation.
3. **Two models, same interface** → Isolation Forest and Autoencoder each implement `BaseDetector.fit(X)` / `BaseDetector.score(X) -> AnomalyScore[]`. Neither model sees labels during training.
4. **Evaluation** → labels are used ONLY here, to compute precision/recall/F1/ROC-AUC and to compare training/inference latency between the two models.
5. **Explainability** → for each flagged anomaly, identify the top contributing features (feature importances for Isolation Forest, per-feature reconstruction error for Autoencoder).
6. **Output** → `AnomalyReport` objects, Pydantic-validated, schema-compatible with future integration into the triage agent.

---

## 3. Repository structure

```
network-anomaly-detector/
├── README.md                          ← this file (project spec)
├── config.yaml                        ← model hyperparameters, feature list, paths
├── pyproject.toml                     ← deps: scikit-learn, torch, pandas, pydantic,
│                                          matplotlib, seaborn, pytest, pyyaml
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py                  ← loads CICIDS2017 CSVs
│   │   └── feature_engineering.py     ← scaling, feature selection, benign/attack split
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base_detector.py           ← abstract interface: fit(X), score(X), name
│   │   ├── isolation_forest_model.py  ← implements BaseDetector
│   │   └── autoencoder_model.py       ← implements BaseDetector, PyTorch nn.Module inside
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── compare_models.py          ← runs both models, computes shared metrics
│   │   └── explainability.py          ← top-N contributing features per anomaly
│   └── schemas/
│       ├── __init__.py
│       └── anomaly_report.py          ← Pydantic output schema
├── data/
│   └── samples/                       ← small labeled subset (committed, reproducible)
│                                          full dataset git-ignored in data/raw/
├── models/                            ← trained artifacts (.pkl, .pt) — git-ignored
├── notebooks/
│   ├── 01_eda.ipynb                   ← exploratory analysis of flow features
│   ├── 02_isolation_forest.ipynb
│   ├── 03_autoencoder.ipynb
│   └── 04_model_comparison.ipynb      ← the centerpiece notebook for the README
├── tests/
│   ├── fixtures/                      ← tiny synthetic flow dataset for fast tests
│   ├── test_feature_engineering.py
│   ├── test_isolation_forest.py
│   ├── test_autoencoder.py
│   ├── test_compare_models.py
│   └── test_schemas.py
└── docs/
    └── comparison_results.md          ← final metrics table + written conclusion
```

---

## 4. Design details

### 4.1 `BaseDetector` interface (implement first, in `src/models/base_detector.py`)

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
        Must be comparable across models after normalization in evaluation."""

    @abstractmethod
    def top_contributing_features(self, x: np.ndarray, feature_names: list[str], k: int = 5) -> list[str]:
        """Explainability hook: which features drove this specific anomaly score."""
```

Both `IsolationForestDetector` and `AutoencoderDetector` implement this exact interface. `compare_models.py` must depend only on this interface — never import model-specific internals.

### 4.2 Isolation Forest (`src/models/isolation_forest_model.py`)

- `sklearn.ensemble.IsolationForest`, hyperparameters from `config.yaml` (`n_estimators`, `contamination`, `max_samples`).
- `score()` returns the negative of `decision_function` (so higher = more anomalous, consistent sign convention with the Autoencoder).
- `top_contributing_features()`: approximate via per-feature deviation from the training set's mean/std for the given sample, ranked by z-score magnitude. (Full SHAP support is a stretch goal, not required for v1.)

### 4.3 Autoencoder (`src/models/autoencoder_model.py`)

- Simple feedforward encoder-decoder in PyTorch. Suggested starting architecture (tune via `config.yaml`, do not hardcode): input_dim → 32 → 16 → 8 → 16 → 32 → input_dim, ReLU activations, MSE reconstruction loss.
- Trained only on benign traffic; anomaly score = per-sample reconstruction error.
- Anomaly threshold = a configurable percentile (default: 95th) of reconstruction error on a held-out benign validation split — NOT computed from the attack data.
- `top_contributing_features()`: per-feature squared error between input and reconstruction, ranked descending.

### 4.4 `AnomalyReport` schema (`src/schemas/anomaly_report.py`)

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
    ground_truth_label: str | None = None  # populated only in evaluation context, never at inference
```

Field names (`source_ip`, `dest_ip`, `timestamp`) intentionally match `NormalizedAlert` from `ai-soc-triage-agent` for low-friction future integration — do not rename them for convenience elsewhere.

### 4.5 Evaluation (`src/evaluation/compare_models.py`)

For each model, compute against the labeled test set:
- Precision, Recall, F1 (using each model's configured threshold)
- ROC-AUC (threshold-independent)
- Training time (wall clock, single run, same machine/session for a fair comparison)
- Inference latency (mean ms per sample, batch of at least 1000 samples)

Output a single comparison table (as a DataFrame, saved to `docs/comparison_results.md`) with both models side by side. Do not just print separate metrics per model — the artifact must be a direct comparison.

---

## 5. Dataset: CICIDS2017 (reused from ai-soc-triage-agent)

- **Source:** Canadian Institute for Cybersecurity — https://www.unb.ca/cic/datasets/ids-2017.html
- **Why reuse it:** consistency across the "Agentic SOC Series" portfolio — same dataset, two different techniques (LLM reasoning vs. classical/deep ML), directly comparable narrative for a README or interview.
- **Difference from the triage agent's usage:** here we use the **numeric flow-level features directly** (flow duration, packets/sec, byte counts, TCP flag ratios, etc.) rather than converting rows into synthetic SIEM-style text alerts.
- **Split strategy:** training set = BENIGN-labeled rows only. Test set = a mixed sample of BENIGN + each attack category, stratified so no single attack type dominates evaluation.
- **Handling:** full CSVs git-ignored in `data/raw/`. Commit a small reproducible sample (a few thousand rows, class-balanced for testing purposes) in `data/samples/`.

---

## 6. Roadmap (phased, no fixed deadlines)

### Phase 1 — Data & feature engineering
- [ ] `loader.py`: load and concatenate CICIDS2017 CSVs
- [ ] EDA notebook (`01_eda.ipynb`): understand feature distributions, class imbalance, correlated features
- [ ] `feature_engineering.py`: select numeric features, `StandardScaler`, benign-only train split, mixed labeled test split
- [ ] `config.yaml`: paths, selected feature list, scaler params
- [ ] Unit tests for feature engineering using the fixture dataset

### Phase 2 — Isolation Forest
- [ ] `base_detector.py`: abstract interface
- [ ] `isolation_forest_model.py`: implements `BaseDetector`
- [ ] Notebook (`02_isolation_forest.ipynb`): train, inspect scores, sanity-check against a few known attack rows
- [ ] Unit tests

### Phase 3 — Autoencoder
- [ ] `autoencoder_model.py`: PyTorch model implementing `BaseDetector`
- [ ] Notebook (`03_autoencoder.ipynb`): training curves, reconstruction error distribution, threshold selection
- [ ] Unit tests (use CPU, small fixture — no GPU dependency for tests)

### Phase 4 — Comparison & explainability
- [ ] `compare_models.py`: run both models over the same labeled test set, compute shared metrics
- [ ] `explainability.py`: top-N contributing features per flagged anomaly, for both models
- [ ] Notebook (`04_model_comparison.ipynb`) — the centerpiece: side-by-side metrics table + plots (ROC curves overlaid, score distributions)
- [ ] `docs/comparison_results.md`: final table + a written, reasoned conclusion (which model, and under what constraints, is the better operational choice)

### Phase 5 — Documentation & polish
- [ ] `AnomalyReport` schema finalized and tested
- [ ] README "Results" section filled in with real numbers (replace placeholders)
- [ ] "Limitations & Future Work": dataset staleness, concept drift in production, and the concrete plan for feeding `AnomalyReport` into `ai-soc-triage-agent`

### Future (out of scope for this repo)
- Wire `AnomalyReport` output as a new alert source feeding `NormalizedAlert` in `ai-soc-triage-agent`
- SHAP-based explainability for Isolation Forest (replacing the z-score approximation)
- Concept drift monitoring / periodic retraining pipeline

---

## 7. Evaluation criteria (definition of done)

| Metric | Target |
|--------|--------|
| Both models implement identical `BaseDetector` interface | Required — hard constraint |
| F1 score (each model, on labeled test set) | Report actual value — no fixed target, this is exploratory |
| ROC-AUC (each model) | Report actual value |
| Comparison table with both models side by side | Required deliverable in `docs/comparison_results.md` |
| Written, reasoned conclusion (not just numbers) | Required — must state a recommendation and its tradeoffs |
| Unit test coverage on `src/` | ≥ 80% |
| No label leakage into training | Required — hard constraint, verify explicitly in `test_feature_engineering.py` |

**No label leakage is the non-negotiable check.** Any code path where the training set for either model touches `ground_truth_label` before evaluation is a bug, not a stylistic choice — write a test that asserts this.

---

## 8. Tech stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.11+ | consistency with `ai-soc-triage-agent` |
| Classical ML | scikit-learn | Isolation Forest, scaling, metrics |
| Deep learning | PyTorch | Autoencoder |
| Validation | Pydantic v2 | strict output contracts, schema-compatible with the triage agent |
| Data | pandas | CICIDS2017 processing |
| Visualization | matplotlib, seaborn | comparison plots (ROC curves, score distributions) |
| Config | PyYAML | single source for hyperparameters |
| Testing | pytest | fixture-based, fast, no GPU/large-dataset dependency |

---

## License

MIT
