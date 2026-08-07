"""Autoencoder anomaly detector implementing `BaseDetector`.

A small PyTorch feedforward encoder-decoder, trained ONLY on benign traffic
(README section 5.3). Architecture width (`hidden_dims`), training
hyperparameters, and the threshold percentile all come from `config.yaml`'s
`models.autoencoder` section (validated via `AutoencoderConfig` in
`src.data.feature_engineering`) -- never hardcoded.

Anomaly score = per-sample MSE reconstruction error (higher = more anomalous,
the same sign convention as `IsolationForestDetector.score`). The anomaly
threshold is the configured percentile of the reconstruction error measured
on a held-out BENIGN validation split carved out of `X_benign` inside `fit`
-- it is never computed from, and never sees, attack data or labels.
"""
from __future__ import annotations

from itertools import pairwise

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn

from src.data.feature_engineering import AutoencoderConfig
from src.models.base_detector import BaseDetector
from src.models.feature_ranking import as_explanation_matrix, rank_top_features


class _FeedforwardAutoencoder(nn.Module):
    """Symmetric encoder-decoder MLP: `input_dim -> hidden_dims -> input_dim`.

    `hidden_dims` (e.g. `[32, 16, 8]`) gives the encoder's layer widths in
    shrinking order; the decoder mirrors them in reverse, ending in a plain
    linear layer back to `input_dim` (no final activation, since standardized
    input features may be negative). Every other layer uses ReLU, matching
    the README section 5.3 architecture sketch.
    """

    def __init__(self, input_dim: int, hidden_dims: list[int]) -> None:
        super().__init__()

        encoder_dims = [input_dim, *hidden_dims]
        encoder_layers: list[nn.Module] = []
        for in_dim, out_dim in pairwise(encoder_dims):
            encoder_layers.append(nn.Linear(in_dim, out_dim))
            encoder_layers.append(nn.ReLU())

        decoder_dims = [*hidden_dims[::-1], input_dim]
        decoder_layers: list[nn.Module] = []
        for i, (in_dim, out_dim) in enumerate(pairwise(decoder_dims)):
            decoder_layers.append(nn.Linear(in_dim, out_dim))
            if i < len(decoder_dims) - 2:
                decoder_layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded


class AutoencoderDetector(BaseDetector):
    """PyTorch feedforward-autoencoder-backed unsupervised anomaly detector."""

    name = "autoencoder"

    def __init__(self, config: AutoencoderConfig | None = None) -> None:
        """Build the detector from validated hyperparameters.

        Args:
            config: `models.autoencoder` section of `config.yaml`, already
                validated into an `AutoencoderConfig`. Defaults to
                `AutoencoderConfig()`'s own defaults if omitted.
        """
        self._config = config or AutoencoderConfig()
        self._model: _FeedforwardAutoencoder | None = None
        self._threshold: float | None = None
        # Stored for white-box testing/inspection: the reconstruction errors
        # the threshold percentile was computed from (benign validation only).
        self._validation_errors: np.ndarray | None = None

    def fit(self, X_benign: np.ndarray) -> None:
        """Train the autoencoder using ONLY benign traffic. No labels involved.

        Internally splits `X_benign` into a training subset and a held-out
        benign validation subset (`validation_fraction` of `config`). The
        validation subset never participates in gradient updates -- it is
        used exclusively to measure reconstruction error and set
        `self._threshold` at the configured percentile. Attack data is never
        passed to, nor seen by, this method: the signature (`X_benign` only,
        no labels) makes that structurally impossible.
        """
        X = np.asarray(X_benign, dtype=np.float64)
        input_dim = X.shape[1]

        torch.manual_seed(self._config.random_state)

        x_fit, x_val = train_test_split(
            X,
            test_size=self._config.validation_fraction,
            random_state=self._config.random_state,
        )

        model = _FeedforwardAutoencoder(input_dim, self._config.hidden_dims)
        optimizer = torch.optim.Adam(model.parameters(), lr=self._config.learning_rate)
        loss_fn = nn.MSELoss()

        fit_tensor = torch.tensor(x_fit, dtype=torch.float32)
        val_tensor = torch.tensor(x_val, dtype=torch.float32)

        n_fit = fit_tensor.shape[0]
        batch_size = min(self._config.batch_size, n_fit)
        rng = np.random.default_rng(self._config.random_state)

        model.train()
        for _ in range(self._config.epochs):
            permutation = rng.permutation(n_fit)
            for start in range(0, n_fit, batch_size):
                batch_indices = permutation[start : start + batch_size]
                batch = fit_tensor[batch_indices]

                optimizer.zero_grad()
                reconstruction = model(batch)
                loss = loss_fn(reconstruction, batch)
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            val_reconstruction = model(val_tensor)
            val_errors = torch.mean((val_reconstruction - val_tensor) ** 2, dim=1).numpy()

        self._model = model
        self._validation_errors = np.asarray(val_errors, dtype=np.float64)
        self._threshold = float(
            np.percentile(self._validation_errors, self._config.threshold_percentile)
        )

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return the per-row MSE reconstruction error. Higher = more anomalous.

        Consistent sign convention with `IsolationForestDetector.score`
        (README section 5.3): the autoencoder is trained to reconstruct
        benign traffic well, so unfamiliar (anomalous) rows reconstruct
        poorly and receive a higher error.

        Raises:
            RuntimeError: if called before `fit`.
        """
        if self._model is None:
            raise RuntimeError("AutoencoderDetector.fit() must be called before score()")

        X_arr = np.asarray(X, dtype=np.float64)
        x_tensor = torch.tensor(X_arr, dtype=torch.float32)

        self._model.eval()
        with torch.no_grad():
            reconstruction = self._model(x_tensor)
            errors = torch.mean((reconstruction - x_tensor) ** 2, dim=1).numpy()

        return np.asarray(errors, dtype=np.float64)

    # `is_anomaly` is intentionally NOT overridden here: `BaseDetector.is_anomaly`
    # already implements `score(X) > threshold` concretely, and reusing it
    # keeps exactly one shared decision rule across both detectors (README
    # section 5.1/5.5) instead of per-model duplication.

    @property
    def threshold(self) -> float:
        """The fitted anomaly threshold (percentile of benign validation error).

        Raises:
            RuntimeError: if accessed before `fit`.
        """
        if self._threshold is None:
            raise RuntimeError("AutoencoderDetector.fit() must be called before threshold")
        return self._threshold

    def top_contributing_features(
        self, x: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[str]:
        """Rank features by per-feature squared reconstruction error.

        Args:
            x: A single flow's feature vector (shape `(n_features,)`).
            feature_names: Names aligned with `x`'s columns.
            k: Number of top feature names to return.

        Returns:
            The `k` feature names with the largest `(x - x_reconstructed) ** 2`,
            descending.

        Raises:
            RuntimeError: if called before `fit`.
            ValueError: if `k` is not positive or `x` contains non-finite values.
        """
        return self.top_contributing_features_batch(
            np.asarray(x, dtype=float).reshape(1, -1), feature_names, k=k
        )[0]

    def top_contributing_features_batch(
        self, X: np.ndarray, feature_names: list[str], k: int = 5
    ) -> list[list[str]]:
        """Vectorized `top_contributing_features` over a whole matrix.

        One forward pass reconstructs every row at once, exactly as `score`
        already does -- the per-feature squared error is just the term `score`
        discards before averaging. Reconstructing a row at a time instead meant
        one batch-of-1 forward pass per flagged flow, which is what made
        explaining a full run impractical. It also means the errors explaining
        a flag now come from the same kernel as the score that raised it.

        The dtype chain is deliberately preserved from the single-row version:
        float64 input, float32 reconstruction, subtraction promoting back to
        float64. Doing the arithmetic in torch instead would silently change
        the numbers.

        Args:
            X: One flow per row (shape `(n, n_features)`).
            feature_names: Names aligned with `X`'s columns.
            k: Number of top feature names per row.

        Returns:
            One list of `min(k, len(feature_names))` names per row of `X`.

        Raises:
            RuntimeError: if called before `fit`.
            ValueError: if `k` is not positive or `X` contains non-finite values.
        """
        if self._model is None:
            raise RuntimeError(
                "AutoencoderDetector.fit() must be called before top_contributing_features()"
            )

        matrix = as_explanation_matrix(X, k)

        x_tensor = torch.tensor(matrix, dtype=torch.float32)
        self._model.eval()
        with torch.no_grad():
            reconstruction = self._model(x_tensor).numpy()

        squared_errors = (matrix - reconstruction) ** 2

        return rank_top_features(squared_errors, feature_names, k)
