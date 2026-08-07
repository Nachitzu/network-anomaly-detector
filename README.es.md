# Network Anomaly Detector

> Detección de anomalías de red basada en ML, comparando dos enfoques no supervisados — **Isolation Forest** vs. **Autoencoder** — sobre datos de flujo de red etiquetados, con explicabilidad y una metodología de evaluación rigurosa y reproducible.

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-in%20development-orange)

**Estado:** En desarrollo — Fase 1 (pipeline de datos e ingeniería de características) completa.
**Autor:** Nachitzu — AI Security Engineer (Blue Team + ML para seguridad).
**Proyecto complementario:** [ai-soc-triage-agent](https://github.com/Nachitzu/ai-soc-triage-agent) — independiente por ahora, diseñado para una futura integración.

**English version:** [README.md](README.md)

---

## 1. Planteamiento del problema

Tanto la detección basada en reglas como la basada en LLM (ver el proyecto complementario `ai-soc-triage-agent`) dependen de firmas predefinidas o de que ya se haya generado una alerta. Este proyecto plantea una pregunta distinta: **¿podemos detectar comportamiento de red anómalo directamente a partir de estadísticas de flujo, sin reglas y sin etiquetas durante el entrenamiento?**

Se entrenan y comparan rigurosamente dos enfoques no supervisados:

- **Isolation Forest** — basado en árboles, rápido, un baseline estándar de la industria.
- **Autoencoder** — una red neuronal que captura relaciones no lineales entre características mediante el error de reconstrucción.

El objetivo no es "qué modelo gana" en abstracto, sino una **recomendación de ingeniería respaldada por datos** (precisión vs. costo de entrenamiento vs. explicabilidad) apta para una decisión real en un SOC. La metodología de comparación y la conclusión importan tanto como los modelos en sí.

## 2. Principios de diseño

Estas restricciones dan forma a todo el código base y están verificadas en los tests:

- **No supervisado por construcción.** Ambos modelos se entrenan **únicamente con tráfico benigno (normal)**. Las etiquetas de verdad fundamental (ground truth) se usan **exclusivamente en el momento de la evaluación**, nunca durante el entrenamiento — verificado mediante un test explícito de no fuga de etiquetas (no-label-leakage).
- **Una interfaz compartida.** Ambos detectores implementan la misma interfaz `BaseDetector`, de modo que el código de evaluación los trata de forma polimórfica y nunca hace casos especiales por modelo.
- **Salidas validadas y listas para integración.** Cada salida de anomalía se valida contra un esquema Pydantic cuyos nombres de campo reflejan el esquema `NormalizedAlert` de `ai-soc-triage-agent`, de modo que una futura integración no requiera reescritura.
- **Configuración antes que valores fijos en código.** La lista de características, las proporciones de partición y los hiperparámetros de los modelos viven en `config.yaml`, no en Python.
- **Reproducible y probado.** Python 3.11+, tipado completo (type hints), y una suite pytest rápida que corre sobre un fixture sintético pequeño (nunca sobre los CSV completos de CICIDS2017), apuntando a ≥ 80% de cobertura.

## 3. Arquitectura

```
┌───────────────────────┐
│      CICIDS2017        │   Dataset etiquetado de flujos de red
│  (flow records)        │   (reutilizado de ai-soc-triage-agent)
└───────────┬───────────┘
            │
            ▼
┌────────────────────────────────┐
│      Feature engineering        │   Escalado, selección de features,
│  (src/data/feature_engineering) │   partición train (solo benigno) / test
└───────────┬────────────────────┘
            │
      ┌─────┴─────┐
      ▼           ▼
┌───────────┐ ┌───────────┐
│Isolation  │ │Autoencoder│    Entrenados de forma independiente,
│Forest     │ │(PyTorch)  │    ambos implementan BaseDetector
│(sklearn)  │ │           │
└─────┬─────┘ └─────┬─────┘
      │             │
      └──────┬──────┘
             ▼
┌─────────────────────────────────┐
│    Comparative evaluation        │   Precision, recall, F1, ROC-AUC,
│    (src/evaluation/compare)      │   latencia de entrenamiento / inferencia
└───────────┬─────────────────────┘
            │
            ▼
┌──────────────────────────────────────────┐
│           Anomaly Report                   │   flow_id, anomaly_score,
│     (Pydantic-validated JSON)              │   model_used, top contributing
│                                            │   features
└──────────────────────────────────────────┘
```

### Flujo de datos

1. **Dataset** — registros de flujo de CICIDS2017 (misma fuente que `ai-soc-triage-agent`, reutilizada por consistencia narrativa y metodológica).
2. **Feature engineering** — se seleccionan características numéricas de flujo, se escalan, y se dividen en un **conjunto de entrenamiento solo con tráfico benigno** y un **conjunto de test mixto (benigno + ataque)** con las etiquetas de verdad fundamental retenidas únicamente para evaluación.
3. **Dos modelos, la misma interfaz** — Isolation Forest y Autoencoder implementan cada uno `BaseDetector.fit(X)` / `BaseDetector.score(X)`. Ninguno de los dos modelos ve etiquetas durante el entrenamiento.
4. **Evaluación** — las etiquetas se usan únicamente aquí, para calcular precision / recall / F1 / ROC-AUC y comparar la latencia de entrenamiento e inferencia entre ambos modelos.
5. **Explicabilidad** — para cada anomalía marcada, se identifican las características que más contribuyeron (desviaciones de características para Isolation Forest, error de reconstrucción por característica para el Autoencoder).
6. **Salida** — objetos `AnomalyReport`, validados con Pydantic y compatibles en esquema con una futura integración en el agente de triage.

## 4. Estructura del repositorio

```
network-anomaly-detector/
├── README.md                          ← especificación del proyecto (inglés)
├── README.es.md                       ← este archivo (español)
├── config.yaml                        ← hiperparámetros de modelos, lista de features, rutas
├── pyproject.toml                     ← dependencias y metadata del proyecto
├── src/
│   ├── data/
│   │   ├── loader.py                  ← carga los CSV de CICIDS2017
│   │   └── feature_engineering.py     ← escalado, selección de features, partición benigno/ataque
│   ├── models/
│   │   ├── base_detector.py           ← interfaz abstracta: fit(X), score(X), name
│   │   ├── isolation_forest_model.py  ← implementa BaseDetector
│   │   └── autoencoder_model.py       ← implementa BaseDetector (nn.Module de PyTorch por dentro)
│   ├── evaluation/
│   │   ├── compare_models.py          ← ejecuta ambos modelos, calcula métricas compartidas
│   │   └── explainability.py          ← top-N características contribuyentes por anomalía
│   └── schemas/
│       └── anomaly_report.py          ← esquema de salida en Pydantic
├── data/
│   ├── raw/                           ← dataset completo (ignorado por git)
│   └── samples/                       ← subconjunto etiquetado pequeño (versionado, reproducible)
├── models/                            ← artefactos entrenados (.pkl, .pt) — ignorados por git
├── notebooks/                         ← exploración y reportes (01_eda … 04_model_comparison)
├── tests/                             ← suite pytest (basada en fixtures, rápida, sin GPU/datos grandes)
└── docs/
    └── comparison_results.md          ← tabla final de métricas + conclusión escrita
```

> Los notebooks son solo para exploración y reportes; toda lógica reutilizable vive en `src/`.

## 5. Detalles de diseño

### 5.1 Interfaz `BaseDetector`

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

Tanto `IsolationForestDetector` como `AutoencoderDetector` implementan exactamente esta interfaz, y el código de comparación depende únicamente de ella — nunca de internals específicos de cada modelo.

### 5.2 Isolation Forest

- `sklearn.ensemble.IsolationForest`, con `n_estimators`, `contamination` y `max_samples` obtenidos de `config.yaml`.
- `score()` devuelve el negativo de `decision_function`, de modo que un valor más alto significa más anómalo (convención de signo consistente con el Autoencoder).
- `top_contributing_features()` aproxima la contribución mediante la desviación por característica respecto a la media/desviación estándar del conjunto de entrenamiento, ordenada por magnitud del z-score. (El soporte para SHAP es una mejora futura.)

### 5.3 Autoencoder

- Un encoder-decoder feedforward simple en PyTorch (dimensiones ajustables vía `config.yaml`; un punto de partida razonable es `input_dim → 32 → 16 → 8 → 16 → 32 → input_dim`, activaciones ReLU, pérdida de reconstrucción MSE).
- Entrenado únicamente con tráfico benigno; el puntaje de anomalía es el error de reconstrucción por muestra.
- El umbral de anomalía es un percentil configurable (por defecto: percentil 95) del error de reconstrucción sobre una partición de validación benigna reservada — nunca calculado a partir de datos de ataque.
- `top_contributing_features()` ordena las características por error de reconstrucción cuadrático por característica.

### 5.4 Esquema `AnomalyReport`

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

Los nombres de los campos (`source_ip`, `dest_ip`, `timestamp`) coinciden intencionalmente con `NormalizedAlert` de `ai-soc-triage-agent` para facilitar una futura integración.

### 5.5 Evaluación

Para cada modelo, calculado sobre el conjunto de test etiquetado:

- Precision, recall, F1 (usando el umbral configurado de cada modelo).
- ROC-AUC (independiente del umbral).
- Tiempo de entrenamiento (reloj de pared, una sola corrida, misma máquina/sesión para una comparación justa).
- Latencia de inferencia (ms promedio por muestra, sobre un lote de al menos 1000 muestras).

El entregable es una única tabla comparativa lado a lado (un DataFrame, guardado en `docs/comparison_results.md`) — una comparación directa, no impresiones separadas por modelo.

## 6. Dataset: CICIDS2017

- **Fuente:** Canadian Institute for Cybersecurity — <https://www.unb.ca/cic/datasets/ids-2017.html>.
- **Por qué se reutiliza:** consistencia en todo el portfolio — el mismo dataset analizado con dos técnicas distintas (razonamiento con LLM vs. ML clásico/profundo), lo que da una narrativa directamente comparable.
- **En qué se diferencia aquí:** este proyecto usa **las características numéricas de nivel de flujo directamente** (duración del flujo, paquetes/seg, conteo de bytes, ratios de flags TCP, etc.) en lugar de convertir las filas en alertas de texto sintéticas estilo SIEM.
- **Estrategia de partición:** conjunto de entrenamiento = solo filas etiquetadas como BENIGN; conjunto de test = una muestra mixta de BENIGN + cada categoría de ataque, estratificada para que ningún tipo de ataque domine la evaluación.
- **Manejo:** los CSV completos están ignorados por git en `data/raw/`; una muestra pequeña, reproducible y balanceada por clase vive en `data/samples/`.

## 7. Cómo empezar

Requiere **Python 3.11+** y [uv](https://docs.astral.sh/uv/).

```bash
# instalar dependencias (incluyendo herramientas de desarrollo)
uv sync --extra dev

# correr la suite de tests con cobertura
uv run pytest -q --cov=src
```

Colocá los CSV completos de CICIDS2017 bajo `data/raw/` (ignorado por git) para correr el pipeline completo; la muestra versionada en `data/samples/` es suficiente para ejercitar el código de punta a punta.

## 8. Roadmap

**Fase 1 — Datos e ingeniería de características** ✅
- Loader de CICIDS2017, selección de features, `StandardScaler`, partición train (solo benigno) / test etiquetado mixto, saneamiento de valores no finitos, y un modelo de configuración tipado.

**Fase 2 — Isolation Forest**
- Interfaz `BaseDetector` y la implementación con `sklearn`, con inspección de puntajes y verificaciones de sanidad contra filas de ataque conocidas.

**Fase 3 — Autoencoder**
- Modelo en PyTorch que implementa `BaseDetector`; curvas de entrenamiento, distribución del error de reconstrucción, y selección de umbral.

**Fase 4 — Comparación y explicabilidad**
- Ejecutar ambos modelos sobre el mismo conjunto de test etiquetado, calcular métricas compartidas, producir las top-N características contribuyentes por anomalía, y el notebook central de comparación lado a lado.

**Fase 5 — Documentación y pulido**
- Finalizar el esquema `AnomalyReport`, completar la sección de resultados con números reales, y documentar limitaciones y trabajo futuro.

**Futuro (fuera del alcance de este repo)**
- Conectar la salida de `AnomalyReport` como una nueva fuente de alertas que alimente `NormalizedAlert` en `ai-soc-triage-agent`.
- Explicabilidad basada en SHAP para Isolation Forest (reemplazando la aproximación por z-score).
- Monitoreo de concept drift y un pipeline de reentrenamiento periódico.

## 9. Criterios de evaluación (definición de terminado)

| Criterio | Objetivo |
|-----------|--------|
| Ambos modelos implementan la misma interfaz `BaseDetector` | Requerido — restricción dura |
| F1 score (cada modelo, sobre el conjunto de test etiquetado) | Reportar valor real — exploratorio |
| ROC-AUC (cada modelo) | Reportar valor real |
| Tabla comparativa con ambos modelos lado a lado | Requerido — `docs/comparison_results.md` |
| Conclusión escrita y razonada (una recomendación y sus tradeoffs) | Requerido |
| Cobertura de tests unitarios sobre `src/` | ≥ 80% |
| Sin fuga de etiquetas hacia el entrenamiento | Requerido — restricción dura, verificada en los tests |

**La ausencia de fuga de etiquetas es la verificación no negociable.** Cualquier ruta de código donde el conjunto de entrenamiento de cualquiera de los dos modelos toque `ground_truth_label` antes de la evaluación es un bug, no una decisión de estilo — y se verifica explícitamente en la suite de tests.

## 10. Stack tecnológico

| Componente | Elección | Razón |
|-----------|--------|--------|
| Lenguaje | Python 3.11+ | Consistencia con `ai-soc-triage-agent` |
| ML clásico | scikit-learn | Isolation Forest, escalado, métricas |
| Deep learning | PyTorch | Autoencoder |
| Validación | Pydantic v2 | Contratos de salida estrictos, compatibles en esquema con el agente de triage |
| Datos | pandas | Procesamiento de CICIDS2017 |
| Visualización | matplotlib, seaborn | Gráficos comparativos (curvas ROC, distribuciones de puntaje) |
| Configuración | PyYAML | Fuente única para hiperparámetros |
| Testing | pytest | Basado en fixtures, rápido, sin dependencia de GPU/datasets grandes |

## Licencia

Publicado bajo la Licencia MIT.
