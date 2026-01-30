"""Isolation Forest training for network anomaly detection.

This module trains an Isolation Forest on the processed telemetry dataset and
serializes the fitted estimator with joblib.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from src.data.preprocessing import preprocess_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "isolation_forest.pkl"


def train_isolation_forest(
    data: pd.DataFrame | str | Path | None = None,
    *,
    data_path: str | Path | None = None,
    model_path: str | Path | None = None,
    contamination: float = 0.02,
    random_state: int = 42,
) -> Tuple[IsolationForest, pd.DataFrame]:
    """Train Isolation Forest on the processed telemetry dataset.

    Args:
        data: Optional in-memory dataframe or CSV path.
        data_path: Optional explicit input CSV path when ``data`` is not a dataframe.
        model_path: Optional output path for the serialized model.
        contamination: Expected anomaly fraction in the dataset.
        random_state: Random seed used by the estimator.

    Returns:
        A tuple containing the fitted model and the processed dataframe used
        for training.
    """
    processed_data = preprocess_data(data=data, data_path=data_path)
    feature_frame = _select_numeric_features(processed_data)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(feature_frame)

    output_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)

    return model, processed_data


def load_trained_model(model_path: str | Path | None = None) -> IsolationForest:
    """Load a previously trained Isolation Forest model from disk."""
    target_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
    if not target_path.exists():
        raise FileNotFoundError(f"Model file not found: {target_path}")
    return joblib.load(target_path)


def predict_anomalies(model: IsolationForest, data: pd.DataFrame) -> pd.Series:
    """Predict anomaly labels for a processed dataframe.

    Returns 1 for normal samples and -1 for anomalies, matching scikit-learn's
    Isolation Forest convention.
    """
    feature_frame = _select_numeric_features(data)
    return pd.Series(model.predict(feature_frame), index=feature_frame.index, name="anomaly_label")


def _select_numeric_features(data: pd.DataFrame) -> pd.DataFrame:
    numeric_frame = data.select_dtypes(include="number")
    if numeric_frame.empty:
        raise ValueError("Isolation Forest requires at least one numeric feature.")
    return numeric_frame


if __name__ == "__main__":
    model, processed = train_isolation_forest()
    print(processed.shape)
    print(f"Saved model to {DEFAULT_MODEL_PATH}")
    print(model)
