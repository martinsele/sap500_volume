from typing import Dict, Any, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers

from sap_predict.model.reference_model import BaseVolumeRegressor


class DeepNeuralModel(BaseVolumeRegressor):
    """Estimator using recurrent neural networks as a regression model."""

    name = 'DeepNeuralModel'

    def __init__(self, *args, **kwargs):
        self.lstm_units = kwargs.get('lstm_units', 40)
        self.dense_units = kwargs.get('dense_units', 100)
        self.history_len = kwargs.get('history_len', 10)
        self.batch_size = kwargs.get('batch_size', 64)
        self.epochs = kwargs.get('epochs', 20)
        self.leaky_alpha = kwargs.get('leaky_alpha', 0.05)
        self.data_scaler = MinMaxScaler()
        self.volume_scaler = MinMaxScaler()
        self.use_diff = True
        self.fitted_model = None

    def _prepare_features(self, data: pd.DataFrame) ->pd.DataFrame:
        """Prepare features."""
        feature_data = data.dropna().drop(['Name', 'Currency'], axis=1)
        feature_data['diff_vol'] = data['Volume'] - data['Volume'].shift(1)
        feature_data['diff_close'] = data['Close'] - data['Close'].shift(1)
        feature_data.dropna(inplace=True)
        return feature_data

    def _preprocess_train_data(self, data: pd.DataFrame, original_targets: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training - create ordered batches."""
        feature_data = self._prepare_features(data)
        feature_data = self.data_scaler.fit_transform(feature_data)

        samples_num = len(feature_data) - self.history_len
        samples = np.empty([samples_num, self.history_len, feature_data.shape[1]])
        for idx in range(self.history_len, len(feature_data)):
            sample_id = idx - self.history_len
            samples[sample_id, :, :] = feature_data[idx-self.history_len: idx, :]

        if self.use_diff:
            original_targets = (original_targets - original_targets.shift(1)).dropna()
            targets_start_idx = self.history_len + 1
        else:
            targets_start_idx = self.history_len + 2  # one less because it is not differentiated

        targets = original_targets[targets_start_idx:]
        targets = self.volume_scaler.fit_transform(np.expand_dims(targets, axis=1))

        return samples, targets

    def _preprocess_test_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare features for prediction - take only necessary history and reshape."""
        feature_data = self._prepare_features(data)
        feature_data = self.data_scaler.transform(feature_data.iloc[-self.history_len:])
        sample = np.expand_dims(feature_data, axis=0)
        return sample

    def fit(self, data, targets):
        """Train model on available data."""
        # targets are volumes for same days as data -> shift data by 1
        preproc_features, preproc_targets = self._preprocess_train_data(data.shift(1), targets)
        model = self.prepare_model(preproc_features.shape[-1])

        model.fit(preproc_features, preproc_targets, batch_size=self.batch_size, epochs=self.epochs)
        self.fitted_model = model

    def predict(self, data: pd.DataFrame) -> int:
        """Predict next Volume value given history data."""
        preproc_features = self._preprocess_test_data(data)
        prediction = self.fitted_model.predict(preproc_features)
        rescaled = self.volume_scaler.inverse_transform(prediction)

        result = rescaled[0][0]
        if self.use_diff:
            result += data.last('1D')["Volume"]

        return result

    def prepare_model(self, n_features: int) -> keras.Model:
        """Prepare LSTM model."""
        model = keras.Sequential()
        model.add(layers.LSTM(self.lstm_units,
                              input_shape=(self.history_len, n_features)))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.Dense(self.dense_units))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        return {
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units,
            'history_len': self.history_len,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'leaky_alpha': self.leaky_alpha,
        }

    def set_params(self, **params: Dict[str, Any]) -> BaseVolumeRegressor:
        self.lstm_units = params.get('lstm_units', 40)
        self.dense_units = params.get('dense_units', 500)
        self.history_len = params.get('history_len', 20)
        self.batch_size = params.get('batch_size', 64)
        self.epochs = params.get('epochs', 30)
        self.leaky_alpha = params.get('leaky_alpha', 0.1)
        self.fitted_model = None
        return self

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""
        return {
            'lstm_units': [40, 50],  # [10, 20, 40],
            'dense_units': [100, 500],  # [50, 100, 500],
            'history_len': [10, 20],  # [20, 100, 200],
            'batch_size': [64],  # [32, 64],
            'epochs': [20, 30],
            'leaky_alpha': [0.05],  # [.1, .3],
        }


class DeepNeuralModelConvLstm(DeepNeuralModel):
    """Estimator using recurrent neural networks with convolutional layers as a regression model."""

    name = 'DeepNeuralModelConvLstm'

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)
        self.cnn_units = kwargs.get('cnn_units', 32)
        self.kernel_size = kwargs.get('kernel_size', 5)
        self.epochs = kwargs.get('epochs', 40)

    def prepare_model(self, n_features: int) -> keras.Model:
        """Prepare CNN-LSTM model."""
        model = keras.Sequential()
        model.add(layers.Conv1D(filters=self.cnn_units, kernel_size=self.kernel_size,
                                input_shape=(self.history_len, n_features)))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.Conv1D(filters=self.cnn_units/2, kernel_size=self.kernel_size,
                                input_shape=(self.history_len, n_features)))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.MaxPooling1D(pool_size=2))
        model.add(layers.LSTM(self.lstm_units))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.Dense(self.dense_units))
        model.add(layers.LeakyReLU(alpha=self.leaky_alpha))
        model.add(layers.Dense(1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this estimator."""
        params = super().get_params(deep)
        params['cnn_units'] = self.cnn_units
        params['kernel_size'] = self.kernel_size
        return params

    def set_params(self, **params: Dict[str, Any]) -> BaseVolumeRegressor:
        super().set_params(**params)
        self.cnn_units = params.get('cnn_units', 32)
        self.kernel_size = params.get('kernel_size', 5)
        return self

    def get_params_to_try(self) -> Dict[str, Any]:
        """Return dictionary of parameters to try in GridSearch hyper-parameter optimization."""
        return {
            'lstm_units': [20, 40],
            'dense_units': [100, 500],
            'history_len': [20, 100],
            'epochs': [30, 40],
            'cnn_units': [16, 32],
            'kernel_size': [5, 10],
        }
