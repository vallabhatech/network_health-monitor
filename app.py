"""Streamlit dashboard for network health monitoring."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data.preprocessing import preprocess_data
from src.ml.isolation_forest import load_trained_model, predict_anomalies
from src.utils.analysis import load_data

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_ROOT / "src" / "data" / "network_data.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "model" / "isolation_forest.pkl"


@st.cache_data(show_spinner=False)
def get_data(path: str) -> pd.DataFrame:
    return load_data(path)


@st.cache_resource(show_spinner=False)
def get_model(path: str):
    return load_trained_model(path)


def build_dashboard() -> None:
    st.set_page_config(page_title="Network Health Monitor", page_icon="📡", layout="wide")
    st.title("📡 Network Health Monitor")
    st.caption("ML-assisted anomaly detection for network telemetry.")

    with st.sidebar:
        st.header("Data source")
        data_path = st.text_input("CSV path", str(DEFAULT_DATA_PATH))
        model_path = st.text_input("Model path", str(DEFAULT_MODEL_PATH))
        rows = st.slider("Rows to analyze", 100, 10_000, 2_000, step=100)

    try:
        data = get_data(data_path)
        model = get_model(model_path)
    except (FileNotFoundError, OSError, ValueError) as exc:
        st.error(f"Unable to load the monitoring assets: {exc}")
        st.stop()

    if data.empty:
        st.warning("The selected dataset is empty.")
        st.stop()

    sample = data.tail(min(rows, len(data))).copy()
    processed = preprocess_data(data=sample)
    labels = predict_anomalies(model, processed)
    sample["anomaly"] = labels.to_numpy()
    sample["status"] = sample["anomaly"].map({1: "Healthy", -1: "Anomaly"})

    anomalies = int((sample["anomaly"] == -1).sum())
    healthy = int((sample["anomaly"] == 1).sum())
    anomaly_rate = anomalies / len(sample) * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Samples analyzed", f"{len(sample):,}")
    col2.metric("Healthy", f"{healthy:,}")
    col3.metric("Anomalies", f"{anomalies:,}")
    col4.metric("Anomaly rate", f"{anomaly_rate:.2f}%")

    st.subheader("Network telemetry")
    chart_columns = [
        column for column in [
            "latency_ms", "packet_loss", "bandwidth_usage", "cpu_usage", "memory_usage"
        ] if column in sample.columns
    ]
    if chart_columns:
        st.line_chart(sample.set_index("timestamp")[chart_columns])

    st.subheader("Detected events")
    anomaly_rows = sample.loc[sample["anomaly"] == -1].copy()
    if anomaly_rows.empty:
        st.success("No anomalies were detected in the selected window.")
    else:
        st.dataframe(
            anomaly_rows.sort_values("timestamp", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Model and dataset details"):
        st.write(f"Dataset: {data_path}")
        st.write(f"Model: {model_path}")
        st.write(f"Original dataset size: {len(data):,} rows")
        st.write("Inference uses the repository preprocessing pipeline before prediction.")


if __name__ == "__main__":
    build_dashboard()
