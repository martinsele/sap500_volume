import os
import time
from typing import Tuple, Dict, Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, max_error
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm

from sap_predict.evaluate.series_grid_search import SeriesGridSearch
from sap_predict.model.dnn_model import DeepNeuralModel
from sap_predict.model.forest_model import RandomForestModel
from sap_predict.model.reference_model import BaseVolumeRegressor, ReferenceEstimator
from sap_predict.model.ts_model import TimeSeriesModel
from sap_predict.utils import utility_functions

SCRIPT_LOC = os.path.dirname(os.path.realpath(__file__))
EVENTS_LOC = os.path.join(SCRIPT_LOC, '../..', 'data/data_and_reports.csv')


def get_data(data_path: str) -> pd.DataFrame:
    """Load data from file."""
    all_data = pd.read_csv(data_path)
    all_data['Date'] = pd.to_datetime(all_data['Date'])
    all_data.fillna('', inplace=True)

    # include knowledge of history (next business day of week and next day events)
    all_data['next_trade_day_of_week'] = all_data['Date'].dt.weekday.shift(-1).dropna().astype(int)
    all_data['tomorrow_impact'] = all_data['Impact'].shift(-1).dropna().astype(int)

    # TODO: encode countries
    # country_split_pattern = re.compile("'(.*?)'")
    # all_data["tomorrow_currency"] = all_data["Currency"].apply(country_split_pattern.findall).apply(set).shift(-1)

    # set index
    all_data.dropna()
    all_data.set_index(['Date'], inplace=True)
    return all_data


def walk_forward_evaluation(all_data: pd.DataFrame, model_to_eval: BaseVolumeRegressor, retrain_on_num: int = 0) \
        -> Tuple[float, float, float]:
    """Train and validate model on test data."""
    predictions = []
    train_data, test_data = all_data[:'2016'], all_data['2017':]
    model_to_eval.fit(train_data, train_data['Volume'])

    history_samples = pd.DataFrame(train_data)
    history_targets = pd.Series(train_data['Volume'])

    # step over each time-step in the test set
    print(f'Evaluating {model_to_eval.name}')
    days = tqdm(test_data.index)
    for idx, day in enumerate(days):
        history = utility_functions.get_history(all_data, day)
        yhat = model_to_eval.predict(history)

        # add history to training data
        if retrain_on_num:
            history_samples = history_samples.append(all_data.loc[day])
            history_targets.at[day] = test_data['Volume'][day]

            if predictions and idx % retrain_on_num == 0:
                model_to_eval.fit(history_samples, history_targets)

        predictions.append(yhat)

    score = r2_score(test_data['Volume'], predictions)
    mae = mean_absolute_error(test_data['Volume'], predictions)
    maxe = max_error(test_data['Volume'], predictions)
    print(score, mae, maxe)
    plot_errors(test_data, predictions)
    return score, mae, maxe


def find_best_parameters(all_data: pd.DataFrame, model: BaseVolumeRegressor) -> Dict[str, Any]:
    """Use GridSearch for estimation of best parameters."""
    train_data, test_data = all_data[:'2016'], all_data['2017':]
    ts_cv = TimeSeriesSplit(n_splits=5)
    data_split = ts_cv.split(train_data)

    grid_search = SeriesGridSearch(estimator=model, cv=data_split, scoring='r2',
                                   param_grid=model.get_params_to_try())
    grid_search.fit(train_data, train_data["Volume"])
    return grid_search.best_params_


def plot_errors(test_data, predictions):
    """Visualize predictions."""
    fig, ax = plt.subplots()
    ax.plot(test_data['Volume'], label='True')
    ax.plot(test_data.index, predictions, label='Predict')
    ax.legend()
    plt.show()


def evaluate_models(optimize_params: bool = True):
    """Evaluate regression models."""
    data = get_data(EVENTS_LOC)
    estimators = [
        # ReferenceEstimator(),
        # TimeSeriesModel(),
        # RandomForestModel(),
        DeepNeuralModel(),
    ]
    results = {}
    for estimator in estimators:
        if optimize_params:
            print(f"Evaluating estimator {estimator.name}")
            time1 = time.time()
            best_params = find_best_parameters(data, estimator)
            print(f"GridSearch run for {time.time() - time1} sec")
            print(f"Best params found are: {best_params}")
            estimator.set_params(**best_params)

        time1 = time.time()
        score, mae, maxe = walk_forward_evaluation(data, estimator)
        results[estimator.name] = {'r2 score': score, 'mae': mae, 'max error': maxe}
        print(f"Run for {time.time() - time1} sec")

    print(results)


if __name__ == "__main__":
    evaluate_models()
