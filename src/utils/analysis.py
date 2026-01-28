"""Data analysis utilities for synthetic network telemetry.

The functions in this module are intentionally notebook-free so they can be
reused from scripts, dashboards, or future pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATHS = [
    PROJECT_ROOT / "data" / "network_data.csv",
    PROJECT_ROOT / "src" / "data" / "network_data.csv",
    PROJECT_ROOT / "src" / "data" / "synthetic_network_data.csv",
]
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "data" / "analysis_plots"


def load_data(data_path: str | Path | None = None) -> pd.DataFrame:
    """Load the analysis dataset from disk.

    Args:
        data_path: Optional explicit path to a CSV file.

    Returns:
        A pandas DataFrame containing network telemetry.

    Raises:
        FileNotFoundError: If no configured dataset path exists.
    """
    candidate_paths = _resolve_data_paths(data_path)
    for path in candidate_paths:
        if path.exists():
            return pd.read_csv(path, parse_dates=["timestamp"])

    raise FileNotFoundError(
        "No dataset found. Checked: "
        + ", ".join(str(path) for path in candidate_paths)
    )


def summary_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric features."""
    numeric_data = data.select_dtypes(include="number")
    return numeric_data.describe().T


def missing_values(data: pd.DataFrame, plots_dir: str | Path | None = None) -> pd.Series:
    """Return missing-value counts and save a bar chart automatically."""
    missing_counts = data.isna().sum().sort_values(ascending=False)
    output_dir = _prepare_plots_dir(plots_dir)
    _save_missing_values_plot(missing_counts, output_dir / "missing_values.png")
    return missing_counts


def correlation_matrix(data: pd.DataFrame, plots_dir: str | Path | None = None) -> pd.DataFrame:
    """Return a numeric correlation matrix and save a heatmap automatically."""
    numeric_data = data.select_dtypes(include="number")
    correlation = numeric_data.corr()
    output_dir = _prepare_plots_dir(plots_dir)
    _save_correlation_plot(correlation, output_dir / "correlation_matrix.png")
    return correlation


def feature_distribution(
    data: pd.DataFrame,
    features: Iterable[str] | None = None,
    plots_dir: str | Path | None = None,
) -> list[Path]:
    """Save feature distribution plots for selected numeric columns.

    Args:
        data: Input telemetry data.
        features: Optional iterable of features to plot. If omitted, all numeric
            features except the timestamp column are used.
        plots_dir: Optional directory where the plots should be stored.

    Returns:
        A list of generated plot paths.
    """
    output_dir = _prepare_plots_dir(plots_dir)
    numeric_data = data.select_dtypes(include="number")

    if features is None:
        selected_features = [column for column in numeric_data.columns]
    else:
        selected_features = [column for column in features if column in numeric_data.columns]

    generated_plots: list[Path] = []
    for feature in selected_features:
        figure_path = output_dir / f"distribution_{feature}.png"
        _save_distribution_plot(numeric_data[feature], feature, figure_path)
        generated_plots.append(figure_path)

    return generated_plots


def _resolve_data_paths(data_path: str | Path | None) -> list[Path]:
    if data_path is not None:
        return [Path(data_path)]
    return list(DEFAULT_DATA_PATHS)


def _prepare_plots_dir(plots_dir: str | Path | None) -> Path:
    output_dir = Path(plots_dir) if plots_dir is not None else DEFAULT_PLOTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _save_missing_values_plot(missing_counts: pd.Series, output_path: Path) -> None:
    plt.figure(figsize=(10, 4))
    missing_counts.plot(kind="bar", color="#2E86AB")
    plt.title("Missing Values by Column")
    plt.xlabel("Column")
    plt.ylabel("Missing Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_correlation_plot(correlation: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    image = plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04)
    plt.xticks(range(len(correlation.columns)), correlation.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation.index)), correlation.index)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _save_distribution_plot(series: pd.Series, feature_name: str, output_path: Path) -> None:
    plt.figure(figsize=(8, 4))
    plt.hist(series.dropna(), bins=30, color="#1F77B4", edgecolor="white")
    plt.title(f"Distribution of {feature_name}")
    plt.xlabel(feature_name)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    print(summary_statistics(df))
    print(missing_values(df))
    print(correlation_matrix(df))
    print(feature_distribution(df))
