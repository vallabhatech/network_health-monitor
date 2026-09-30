import pandas as pd

from src.data.preprocessing import convert_timestamp, preprocess_data


def test_convert_timestamp_derives_time_features():
    data = pd.DataFrame({"timestamp": ["2026-01-01 12:34:00"], "latency_ms": [20.0]})

    result = convert_timestamp(data)

    assert "timestamp" not in result.columns
    assert result.loc[0, "timestamp_hour"] == 12
    assert result.loc[0, "timestamp_minute"] == 34


def test_preprocess_data_returns_numeric_model_features():
    data = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=20, freq="min"),
            "latency_ms": range(20),
            "packet_loss": [0.1] * 20,
            "cpu_usage": range(20),
            "memory_usage": range(20),
        }
    )

    result = preprocess_data(data=data, variance_threshold=0.0)

    assert not result.empty
    assert all(pd.api.types.is_numeric_dtype(dtype) for dtype in result.dtypes)
