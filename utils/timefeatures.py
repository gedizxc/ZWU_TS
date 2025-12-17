import numpy as np
import pandas as pd


class TimeFeature:
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__ + "()"


class SecondOfMinute(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.second / 59.0 - 0.5


class MinuteOfHour(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.minute / 59.0 - 0.5


class HourOfDay(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.hour / 23.0 - 0.5


class DayOfWeek(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return index.dayofweek / 6.0 - 0.5


class DayOfMonth(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.day - 1) / 30.0 - 0.5


class DayOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.dayofyear - 1) / 365.0 - 0.5


class MonthOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        return (index.month - 1) / 11.0 - 0.5


class WeekOfYear(TimeFeature):
    def __call__(self, index: pd.DatetimeIndex) -> np.ndarray:
        # pandas >= 1.1: isocalendar().week
        return (index.isocalendar().week.astype(int) - 1) / 52.0 - 0.5


def time_features_from_frequency_str(freq: str) -> list[TimeFeature]:
    """
    Borrowed (and simplified) from Informer2020:
    returns a list of time feature extractors for a given frequency string.
    """
    freq = freq.lower()
    if freq in ("h", "hour", "hourly"):
        return [HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
    if freq in ("t", "min", "minute"):
        return [MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
    if freq in ("s", "sec", "second"):
        return [SecondOfMinute(), MinuteOfHour(), HourOfDay(), DayOfWeek(), DayOfMonth(), DayOfYear()]
    if freq in ("d", "day", "daily"):
        return [DayOfWeek(), DayOfMonth(), DayOfYear()]
    if freq in ("w", "week", "weekly"):
        return [WeekOfYear()]
    if freq in ("m", "month", "monthly"):
        return [MonthOfYear()]

    raise ValueError(f"Unsupported freq={freq!r}. Try one of: h/t/s/d/w/m.")


def time_features(dates: pd.DataFrame, timeenc: int = 1, freq: str = "h") -> np.ndarray:
    """
    dates: a DataFrame with a 'date' column already parsed to datetime.

    - timeenc=0: classic Informer "manual" features (month/day/weekday/hour/minute)
    - timeenc=1: normalized time features from frequency string (timeF)
    """
    if timeenc == 0:
        dates["date"] = pd.to_datetime(dates["date"])
        dates["month"] = dates["date"].dt.month
        dates["day"] = dates["date"].dt.day
        dates["weekday"] = dates["date"].dt.weekday
        dates["hour"] = dates["date"].dt.hour
        dates["minute"] = dates["date"].dt.minute
        dates["minute"] = dates["minute"].map(lambda x: x // 15)
        return dates[["month", "day", "weekday", "hour", "minute"]].values

    dates = pd.to_datetime(dates["date"])
    index = pd.DatetimeIndex(dates)
    features = [feat(index) for feat in time_features_from_frequency_str(freq)]
    return np.stack(features, axis=1).astype(np.float32)
