"""Preprocessing pipeline for synthetic network telemetry.

The pipeline is designed to be reusable from scripts or the dashboard layer.
It handles timestamp conversion, outlier clipping, feature selection, and
feature scaling before returning a model-ready dataframe.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.preprocessing import StandardScaler

from src.utils.analysis import load_data


TIMESTAMP_COLUMN = "timestamp"
DEFAULT_CORRELATION_THRESHOLD = 0.95
DEFAULT_OUTLIER_FACTOR = 1.5
DEFAULT_VARIANCE_THRESHOLD = 0.0


def preprocess_data(
    data: pd.DataFrame | str | Path | None = None,
    *,
    data_path: str | Path | None = None,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    outlier_factor: float = DEFAULT_OUTLIER_FACTOR,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Run the full preprocessing pipeline and return a processed dataframe.

    Args:
        data: Optional dataframe or CSV path. If omitted, the default project
            dataset is loaded.
        data_path: Optional explicit CSV path when ``data`` is not a dataframe.
        correlation_threshold: Absolute correlation limit used for pruning.
        outlier_factor: IQR multiplier used to clip numeric outliers.
        variance_threshold: Minimum variance required to keep a feature.

    Returns:
        A processed dataframe ready for downstream analysis or modeling.
    """
    frame = _load_input_frame(data, data_path)
    frame = convert_timestamp(frame)
    frame = outlier_preprocessing(frame, factor=outlier_factor)
    frame = feature_selection(
        frame,
        correlation_threshold=correlation_threshold,
        variance_threshold=variance_threshold,
    )
    frame = scaling(frame)
    return frame


def convert_timestamp(data: pd.DataFrame) -> pd.DataFrame:
    """Convert timestamp to datetime and derive model-friendly features."""
    frame = data.copy()
    if TIMESTAMP_COLUMN not in frame.columns:
        return frame

    timestamp = pd.to_datetime(frame[TIMESTAMP_COLUMN], errors="coerce")
    frame[TIMESTAMP_COLUMN] = timestamp
    frame["timestamp_year"] = timestamp.dt.year
    frame["timestamp_month"] = timestamp.dt.month
    frame["timestamp_day"] = timestamp.dt.day
    frame["timestamp_dayofweek"] = timestamp.dt.dayofweek
    frame["timestamp_hour"] = timestamp.dt.hour
    frame["timestamp_minute"] = timestamp.dt.minute
    frame = frame.drop(columns=[TIMESTAMP_COLUMN])
    return frame


def outlier_preprocessing(data: pd.DataFrame, factor: float = DEFAULT_OUTLIER_FACTOR) -> pd.DataFrame:
    """Clip numeric outliers using the IQR method."""
    frame = data.copy()
    numeric_columns = frame.select_dtypes(include="number").columns
    for column in numeric_columns:
        lower_bound, upper_bound = _iqr_bounds(frame[column], factor)
        frame[column] = frame[column].clip(lower=lower_bound, upper=upper_bound)
    return frame


def feature_selection(
    data: pd.DataFrame,
    *,
    correlation_threshold: float = DEFAULT_CORRELATION_THRESHOLD,
    variance_threshold: float = DEFAULT_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Select useful numeric features and remove highly correlated columns."""
    frame = data.copy()
    numeric_frame = frame.select_dtypes(include="number")

    if numeric_frame.empty:
        return frame

    selected_numeric = _remove_low_variance_features(numeric_frame, variance_threshold)
    selected_numeric = _remove_highly_correlated_features(selected_numeric, correlation_threshold)

    non_numeric = frame.drop(columns=selected_numeric.columns, errors="ignore").select_dtypes(exclude="number")
    return pd.concat([selected_numeric, non_numeric], axis=1)


def scaling(data: pd.DataFrame) -> pd.DataFrame:
    """Scale numeric columns with standard normalization."""
    frame = data.copy()
    numeric_columns = frame.select_dtypes(include="number").columns
    if len(numeric_columns) == 0:
        return frame

    scaler = StandardScaler()
    frame[numeric_columns] = scaler.fit_transform(frame[numeric_columns])
    return frame


def _load_input_frame(data: pd.DataFrame | str | Path | None, data_path: str | Path | None) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, (str, Path)):
        return load_data(data)
    if data_path is not None:
        return load_data(data_path)
    return load_data()


def _iqr_bounds(series: pd.Series, factor: float) -> tuple[float, float]:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - factor * iqr
    upper_bound = q3 + factor * iqr
    return lower_bound, upper_bound


def _remove_low_variance_features(data: pd.DataFrame, variance_threshold: float) -> pd.DataFrame:
    if data.empty:
        return data
    selector = VarianceThreshold(threshold=variance_threshold)
    selector.fit(data)
    selected_columns = data.columns[selector.get_support()].tolist()
    return data[selected_columns]


def _remove_highly_correlated_features(data: pd.DataFrame, correlation_threshold: float) -> pd.DataFrame:
    if data.shape[1] <= 1:
        return data

    correlation_matrix = data.corr().abs()
    upper_triangle = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    columns_to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > correlation_threshold)]
    return data.drop(columns=columns_to_drop)


if __name__ == "__main__":
    processed = preprocess_data()
    print(processed.head())
    print(processed.shape)
