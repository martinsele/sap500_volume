import pandas as pd
from pandas.core.tools.datetimes import DatetimeScalar


def get_history(data: pd.DataFrame, time: DatetimeScalar, history_len: int = 300, include_today=False) -> pd.DataFrame:
    """Get data history for given time."""
    day_index = data.index.get_loc(time)
    day_from = day_index - history_len
    day_to = day_index + 1 if include_today else day_index
    return data[day_from:day_to]
