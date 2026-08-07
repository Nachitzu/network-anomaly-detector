"""Model-agnostic comparison and explainability for `BaseDetector` implementations.

Everything under this package depends ONLY on `src.models.base_detector.BaseDetector`
(`fit`, `score`, `threshold`, `is_anomaly`, `top_contributing_features`) -- never on
`IsolationForestDetector`/`AutoencoderDetector` internals -- so both detectors are
treated polymorphically (README section 2, "One shared interface").
"""
