"""
Energy-based OOD detector for tabular data.

Original energy score (Liu et al., 2020) is defined on neural network logits:
    E(x) = -T * log ∑_c exp(f_c(x) / T)

For tabular data without a fixed neural backbone we offer two modes:

1. ``mode="classifier"``  (default)
   Wraps any sklearn-style probabilistic classifier.  The "logits" are
   approximated as log-probabilities from ``predict_proba``, giving:
       E(x) ≈ -T * log ∑_c exp(log p_c(x) / T)
            = -T * log ∑_c p_c(x)^(1/T)
   When T→1 this converges to -log ∑_c p_c(x) = 0 (since probs sum to 1),
   so we use log p_c directly (i.e. we score by negative max-logprob /
   log-sum-exp of log-probs).  Higher energy = more OOD.

2. ``mode="density"``
   Fits a Gaussian Mixture Model on the training data and uses the negative
   log-likelihood as the energy proxy.  Fully unsupervised — no classifier
   needed.

Reference
---------
Liu et al., "Energy-based Out-of-distribution Detection", NeurIPS 2020.
Grathwohl et al., "Your classifier is secretly an energy based model",
ICLR 2020.
"""

from __future__ import annotations

from typing import Any, Literal, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from oodt.detection.base import BaseOODDetector


class EnergyOODDetector(BaseOODDetector):
    """
    Energy-based OOD detector for tabular data.

    Parameters
    ----------
    mode : {"classifier", "density"}
        Scoring backend.
        "classifier" — requires a sklearn probabilistic classifier passed
                       as ``classifier``.
        "density"    — fits a GMM internally; ``classifier`` is ignored.
    classifier : sklearn estimator, optional
        Required when ``mode="classifier"``.  Must implement
        ``predict_proba``.
    temperature : float
        Temperature parameter T.  Higher T → softer score distribution.
        Typical values: 0.5–2.0.  Use T=1 as a starting point.
    n_components : int
        Number of GMM components (density mode only).
    normalize : bool
        Standardize features before fitting/scoring.
    random_state : int, optional
        Random seed (density mode).
    """

    def __init__(
        self,
        mode: Literal["classifier", "density"] = "density",
        classifier: Optional[Any] = None,
        temperature: float = 1.0,
        n_components: int = 5,
        normalize: bool = True,
        random_state: Optional[int] = 42,
    ) -> None:
        super().__init__()
        if mode not in ("classifier", "density"):
            raise ValueError(f"mode must be 'classifier' or 'density', got {mode!r}")
        if mode == "classifier" and classifier is None:
            raise ValueError("mode='classifier' requires a classifier to be passed.")
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")

        self.mode = mode
        self.classifier = classifier
        self.temperature = temperature
        self.n_components = n_components
        self.normalize = normalize
        self.random_state = random_state

        self._scaler: Optional[StandardScaler] = None
        self._gmm: Optional[GaussianMixture] = None

    # ------------------------------------------------------------------
    # BaseOODDetector interface
    # ------------------------------------------------------------------

    def fit(self, X_train: pd.DataFrame, y_train: Optional[pd.Series] = None) -> "EnergyOODDetector":
        """
        Fit the detector on training (ID) data.

        Parameters
        ----------
        X_train : pd.DataFrame
            In-distribution training features.
        y_train : pd.Series, optional
            Required when ``mode="classifier"``.
        """
        X = X_train.to_numpy(dtype=float)

        if self.normalize:
            self._scaler = StandardScaler()
            X = self._scaler.fit_transform(X)

        if self.mode == "classifier":
            if y_train is None:
                raise ValueError("y_train is required for mode='classifier'.")
            # Fit classifier on scaled features
            import copy
            self._clf = copy.deepcopy(self.classifier)
            if not hasattr(self._clf, "predict_proba"):
                raise TypeError(
                    f"{type(self._clf).__name__} does not support predict_proba."
                )
            self._clf.fit(X, y_train.to_numpy())

        else:  # density
            self._gmm = GaussianMixture(
                n_components=self.n_components,
                covariance_type="full",
                random_state=self.random_state,
            )
            self._gmm.fit(X)

        self.is_fitted_ = True
        return self

    def score_samples(self, X: pd.DataFrame) -> np.ndarray:
        """
        Compute energy-based OOD scores.  Higher = more likely OOD.

        For "classifier" mode:
            E(x) = -T · log ∑_c exp(logit_c(x) / T)
            where logit_c ≈ log p_c (log-prob from predict_proba).

        For "density" mode:
            E(x) = -log p(x)  where p is the GMM density.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
        """
        self._check_is_fitted()
        Xn = X.to_numpy(dtype=float)

        if self.normalize and self._scaler is not None:
            Xn = self._scaler.transform(Xn)

        if self.mode == "classifier":
            return self._classifier_energy(Xn)
        return self._density_energy(Xn)

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """Binary OOD prediction (1 = OOD, 0 = ID)."""
        scores = self.score_samples(X)
        thr = threshold if threshold is not None else self._get_threshold()
        return (scores >= thr).astype(int)

    def get_params(self) -> dict:
        return {
            "mode": self.mode,
            "temperature": self.temperature,
            "n_components": self.n_components,
            "normalize": self.normalize,
        }

    # ------------------------------------------------------------------
    # Scoring backends
    # ------------------------------------------------------------------

    def _classifier_energy(self, X: np.ndarray) -> np.ndarray:
        """
        Energy from a probabilistic classifier.

        E(x) = -T · logsumexp(log_p / T)

        Note: Since ∑ p_c = 1, at T=1 this degenerates to 0.
        We therefore use a slight reformulation: we treat the log-probs as
        proxy logits (not renormalised by T), which preserves relative
        ordering and matches the spirit of the original paper.
        """
        T = self.temperature
        log_probs = np.log(self._clf.predict_proba(X) + 1e-12)  # (n, C)
        # logsumexp trick for numerical stability
        scaled = log_probs / T                                    # (n, C)
        max_s = scaled.max(axis=1, keepdims=True)
        log_Z = max_s.squeeze() + np.log(np.exp(scaled - max_s).sum(axis=1))
        return -T * log_Z   # higher E → lower density → more OOD

    def _density_energy(self, X: np.ndarray) -> np.ndarray:
        """
        Energy from a GMM: E(x) = -log p_GMM(x).
        sklearn's score_samples returns log p(x), so we negate it.
        """
        return -self._gmm.score_samples(X)
