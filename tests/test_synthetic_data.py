from src.data.synthetic_data import generate_synthetic_network_data


def test_generation_is_reproducible():
    first = generate_synthetic_network_data(rows=500, seed=42, anomaly_count=4)
    second = generate_synthetic_network_data(rows=500, seed=42, anomaly_count=4)

    assert first.equals(second)
    assert len(first) == 500
    assert {"timestamp", "latency_ms", "packet_loss", "cpu_usage"}.issubset(first.columns)


def test_generation_supports_small_datasets():
    data = generate_synthetic_network_data(rows=20, seed=7, anomaly_count=2)

    assert len(data) == 20
    assert data["latency_ms"].ge(0).all()
    assert data["packet_loss"].ge(0).all()
