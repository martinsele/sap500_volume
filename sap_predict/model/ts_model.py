import warnings
from typing import Dict, Any
import pandas as pd
import statsmodels.api as sm

from sap_predict.model.reference_model import BaseVolumeRegressor
from sap_predict.utils import utility_functions


class TimeSeriesModel(BaseVolumeRegressor):
    """Time Series model using SARIMAX regression model."""

    name = 'TimeSeriesModel'

    def __init__(self, *args, **kwargs):
        self.order_num = kwargs.get('order_num', 4)
        self.trend = kwargs.get('trend', None)
        self.seasonal_order_period = kwargs.get('seasonal_order_period', 5)

        self.order = (self.order_num, 0, self.order_num)
        self.seasonal_order = (self.order_num, 0, self.order_num, self.seasonal_order_period)
        self.fitted_model = None

    def _preprocess(self, data_series: pd.Series, shift_value: int = 1) -> pd.Series:
        """Differentiate input values"""
        return data_series - data_series.shift(shift_value)

    def fit(self, data, targets):
        """Train model on available data."""
        history = utility_functions.get_history(data, data.last('1D').index.values[0], 1200)
        series_preproc = self._preprocess(history['Volume']).dropna()
        close_preproc = self._preprocess(history['Close']).dropna().shift(1).fillna(0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.tsa.statespace.SARIMAX(series_preproc, exog=close_preproc,
                                              order=self.order, trend=self.trend, seasonal_order=self.seasonal_order)
        self.fitted_model = model.fit()

    def predict(self, data: pd.DataFrame) -> int:
        """Predict next Volume value given history data."""
        series_preproc = self._preprocess(data['Volume']).dropna()
        close_preproc = self._preprocess(data['Close']).dropna()
        closed_to_be_used = close_preproc.shift(1).fillna(0)
        closed_as_exog = close_preproc.last('1D').values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mod = sm.tsa.statespace.SARIMAX(series_preproc, exog=closed_to_be_used,
                                            order=self.order, trend=self.trend, seasonal_order=self.seasonal_order)
            res = mod.filter(self.fitted_model.params)
            prediction = res.forecast(1, exog=closed_as_exog)
            post_processed = data.last('1D')['Volume'].values[0] + prediction.iloc[0]
        return post_processed

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'order_num': self.order_num,
            'trend': self.trend,
            'seasonal_order_period': self.seasonal_order_period
        }

    def set_params(self, **params: Dict[str, Any]) -> BaseVolumeRegressor:
        """Set the parameters of this estimator."""
        self.order_num = params.get('order_num', 4)
        self.trend = params.get('trend', 'c')
        self.seasonal_order_period = params.get('seasonal_order_period', 20)

        self.order = (self.order_num, 0, self.order_num)
        self.seasonal_order = (self.order_num, 0, self.order_num, self.seasonal_order_period)
        self.fitted_model = None
        return self

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""
        return {
            'order_num': [4, 5, 10],
            'trend': [None, 'c'],
            'seasonal_order_period': [5, 10, 20]
        }
