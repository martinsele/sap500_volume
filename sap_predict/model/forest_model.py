from typing import Dict, Any, List

import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from sap_predict.model.reference_model import BaseVolumeRegressor


class RandomForestModel(BaseVolumeRegressor):
    """Estimator using RandomForests as a regression model."""

    name = 'RandomForestModel'

    def __init__(self, *args, **kwargs):
        self.n_estimators = kwargs.get('n_estimators', 500)
        self.max_depth = kwargs.get('max_depth', 10)
        self.min_samples_split = kwargs.get('min_samples_split', 2)
        self.fitted_model = None
        self.training_data = None

    def _get_lag_features(self, data_series: pd.Series, lags: List[int]) -> pd.DataFrame:
        """Create features using lags of current series."""
        data_frame = pd.DataFrame(index=data_series.index)
        for lag in lags:
            name = f"{data_series.name}_lag_{lag}"
            data_frame[name] = data_series.shift(lag)
        return data_frame

    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for the decision."""
        features = pd.DataFrame(index=data.index)
        features['diff_vol'] = data['Volume'] - data['Volume'].shift(1)
        features['diff_close'] = data['Close'] - data['Close'].shift(1)
        features['next_trade_day_of_week'] = data['next_trade_day_of_week']
        features['tomorrow_impact'] = data["tomorrow_impact"]
        features['today_impact'] = data["Impact"]
        volume_lags = self._get_lag_features(features["diff_vol"], [1, 2, 3, 4, 5, 8, 9, 10])
        close_lags = self._get_lag_features(features["diff_close"], [1, 2, 3, 4, 5, 8, 9, 10])
        features = features.join(volume_lags, how='left')
        features = features.join(close_lags, how='left')
        features.dropna(inplace=True)
        return features

    def _preprocess_targets(self, targets: pd.Series) -> pd.Series:
        """Differentiate targets for stationarity."""
        return (targets - targets.shift(1)).dropna()

    def fit(self, data, targets):
        """Train model on available data."""
        preproc_features = self._preprocess_data(data.shift(1))
        preproc_targets = self._preprocess_targets(targets)[preproc_features.index[0]:]

        regressor = RandomForestRegressor(self.n_estimators, max_depth=self.max_depth,
                                          min_samples_split=self.min_samples_split,
                                          n_jobs=-1)
        regressor.fit(preproc_features, preproc_targets)
        self.fitted_model = regressor

    def predict(self, data: pd.DataFrame) -> int:
        """Predict next Volume value given history data."""
        preproc_features = self._preprocess_data(data)
        prediction = self.fitted_model.predict(preproc_features.last('1D'))
        post_processed = data.last('1D')['Volume'].values[0] + prediction[0]
        return post_processed

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split
        }

    def set_params(self, **params: Dict[str, Any]) -> BaseVolumeRegressor:
        """Set the parameters of this estimator."""
        self.n_estimators = params.get('n_estimators', 500)
        self.max_depth = params.get('max_depth', 10)
        self.min_samples_split = params.get('min_samples_split', 2)
        self.fitted_model = None
        return self

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""
        return {
            'n_estimators': [100, 500, 1000, 2000],
            'max_depth': [10, 50, 100],
            'min_samples_split': [2, 3, 5]
        }
