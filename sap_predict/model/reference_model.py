"""Basic estimators to be used for volume prediction."""

from __future__ import annotations
from typing import Dict, Any
from sklearn.base import BaseEstimator
import pandas as pd


class BaseVolumeRegressor(BaseEstimator):
    """Base class for volume estimators."""

    name = "BaseVolumeRegressor"

    def fit(self, data, targets):
        """Train model on available data."""

    def predict(self, data: pd.DataFrame) -> int:
        """Predict next Volume value given history data."""
        return data["Volume"].iat[-1]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""

    def set_params(self, **params: Dict[str, Any]) -> BaseVolumeRegressor:
        """Set the parameters of this estimator."""
        return self

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""


class ReferenceEstimator(BaseVolumeRegressor):
    """Reference volume estimator, returning data from previous day."""

    name = "ReferenceEstimator"

    def predict(self, data: pd.DataFrame) -> int:
        """Predict next Volume value from previous day."""
        return data["Volume"].iat[-1]

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""
        return {}