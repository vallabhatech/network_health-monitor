"""Synthetic network telemetry data generator.

This module creates a CSV dataset with normal traffic patterns and randomly
injected anomaly periods to support dashboarding, experimentation, and model
training.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


def generate_synthetic_network_data(
    rows: int = 100_000,
    output_path: str | Path | None = None,
    seed: int = 42,
    anomaly_count: int = 8,
) -> pd.DataFrame:
    """Generate synthetic network health telemetry data.

    Args:
        rows: Number of time points to generate.
        output_path: Optional path where the CSV file should be written.
        seed: Random seed for reproducibility.
        anomaly_count: Number of randomly injected anomaly windows.

    Returns:
        A pandas DataFrame containing synthetic network metrics.
    """
    rng = np.random.default_rng(seed)

    timestamps = pd.date_range(
        start="2026-01-01 00:00:00",
        periods=rows,
        freq="min",
    )

    base_latency = 25 + rng.normal(0, 3, size=rows)
    base_packet_loss = 0.5 + rng.normal(0, 0.2, size=rows)
    base_bandwidth = 70 + rng.normal(0, 8, size=rows)
    base_cpu = 45 + rng.normal(0, 6, size=rows)
    base_memory = 55 + rng.normal(0, 5, size=rows)
    base_jitter = 1.5 + rng.normal(0, 0.4, size=rows)
    base_connections = 300 + rng.normal(0, 30, size=rows)

    data = {
        "timestamp": timestamps,
        "latency_ms": np.clip(base_latency, 10, None),
        "packet_loss": np.clip(base_packet_loss, 0, 10),
        "bandwidth_usage": np.clip(base_bandwidth, 10, 100),
        "cpu_usage": np.clip(base_cpu, 10, 100),
        "memory_usage": np.clip(base_memory, 10, 100),
        "network_jitter": np.clip(base_jitter, 0.1, 10),
        "active_connections": np.clip(base_connections, 50, None),
    }

    df = pd.DataFrame(data)

    anomaly_windows = _build_anomaly_windows(rows, anomaly_count, rng)
    for start_idx, end_idx, anomaly_type in anomaly_windows:
        _inject_anomaly(df, start_idx, end_idx, anomaly_type, rng)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)

    return df


def _build_anomaly_windows(
    rows: int,
    anomaly_count: int,
    rng: np.random.Generator,
) -> List[Tuple[int, int, str]]:
    """Create randomly placed anomaly windows across the dataset."""
    anomaly_types = [
        "high_latency",
        "packet_loss_spike",
        "bandwidth_saturation",
        "cpu_overload",
        "memory_leak",
        "connection_flood",
    ]
    windows: List[Tuple[int, int, str]] = []

    for _ in range(max(1, min(anomaly_count, 12))):
        start_idx = int(rng.integers(0, max(1, rows - 60)))
        length = int(rng.integers(60, min(300, max(61, rows - start_idx))))
        end_idx = min(rows, start_idx + length)
        anomaly_type = anomaly_types[int(rng.integers(0, len(anomaly_types)))]
        windows.append((start_idx, end_idx, anomaly_type))

    return windows


def _inject_anomaly(
    df: pd.DataFrame,
    start_idx: int,
    end_idx: int,
    anomaly_type: str,
    rng: np.random.Generator,
) -> None:
    """Inject a single anomaly pattern into the dataset."""
    slice_df = df.iloc[start_idx:end_idx]
    if anomaly_type == "high_latency":
        df.loc[slice_df.index, "latency_ms"] += 80 + rng.normal(0, 15, size=len(slice_df))
        df.loc[slice_df.index, "network_jitter"] += 2 + rng.normal(0, 0.8, size=len(slice_df))
    elif anomaly_type == "packet_loss_spike":
        df.loc[slice_df.index, "packet_loss"] += 5 + rng.normal(0, 1.5, size=len(slice_df))
        df.loc[slice_df.index, "bandwidth_usage"] -= 20 + rng.normal(0, 5, size=len(slice_df))
    elif anomaly_type == "bandwidth_saturation":
        df.loc[slice_df.index, "bandwidth_usage"] += 30 + rng.normal(0, 10, size=len(slice_df))
        df.loc[slice_df.index, "latency_ms"] += 20 + rng.normal(0, 5, size=len(slice_df))
    elif anomaly_type == "cpu_overload":
        df.loc[slice_df.index, "cpu_usage"] += 25 + rng.normal(0, 8, size=len(slice_df))
        df.loc[slice_df.index, "latency_ms"] += 15 + rng.normal(0, 4, size=len(slice_df))
    elif anomaly_type == "memory_leak":
        df.loc[slice_df.index, "memory_usage"] += 20 + rng.normal(0, 6, size=len(slice_df))
        df.loc[slice_df.index, "cpu_usage"] += 10 + rng.normal(0, 3, size=len(slice_df))
    elif anomaly_type == "connection_flood":
        df.loc[slice_df.index, "active_connections"] += 100 + rng.normal(0, 25, size=len(slice_df))
        df.loc[slice_df.index, "packet_loss"] += 2 + rng.normal(0, 0.8, size=len(slice_df))

    numeric_columns = [
        "latency_ms",
        "packet_loss",
        "bandwidth_usage",
        "cpu_usage",
        "memory_usage",
        "network_jitter",
        "active_connections",
    ]
    df.loc[slice_df.index, numeric_columns] = df.loc[slice_df.index, numeric_columns].clip(lower=0)


if __name__ == "__main__":
    output_file = Path(__file__).resolve().parents[1] / "data" / "network_data.csv"
    generate_synthetic_network_data(rows=100_000, output_path=output_file, seed=42)
    print(f"Generated synthetic network data at {output_file}")
