# Network Health Monitor

An ML-assisted network telemetry project for detecting unusual behavior in network health metrics. It combines reproducible synthetic data generation, preprocessing, Isolation Forest anomaly detection, analysis utilities, and a Streamlit dashboard.

> Scope: the included dataset is synthetic telemetry. The repository does not currently claim to monitor a production network or ingest live traffic.

## Features

- Deterministic synthetic network telemetry with configurable anomaly windows.
- Reusable timestamp conversion, outlier handling, variance filtering, correlation filtering, and scaling.
- Isolation Forest training and inference with scikit-learn.
- Streamlit dashboard for telemetry exploration and detected events.
- Automated pytest tests.
- Ruff linting, MyPy type checking, Python compile checks, and CodeQL scanning through GitHub Actions.

## Architecture

~~~text
CSV telemetry -> preprocessing -> Isolation Forest -> Streamlit dashboard
                         |
                         +-> analysis utilities and plots
                         +-> reproducible synthetic data generator
~~~

## Repository layout

~~~text
.github/workflows/   CI and CodeQL
src/data/            data generation and preprocessing
src/ml/              model training and inference
src/utils/           analysis helpers
tests/               automated tests
data/                analysis plots
model/               serialized model artifact
app.py               Streamlit entry point
config.py            environment-aware configuration
pyproject.toml       test, lint, and type-check configuration
requirements.txt     dependencies
CHANGELOG.md         project history
~~~

## Quick start

Requires Python 3.12.

~~~bash
git clone https://github.com/vallabhatech/network_health-monitor.git
cd network_health-monitor
python -m venv .venv
.venv\\Scripts\\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
~~~

The dashboard defaults to `src/data/network_data.csv` and `model/isolation_forest.pkl`. Both can be overridden in the sidebar.

## Generate data or retrain

~~~bash
python -m src.data.synthetic_data
python -m src.ml.isolation_forest
~~~

## Quality checks

~~~bash
ruff check .
mypy src app.py config.py
pytest -q
python -m compileall -q src app.py config.py
~~~

CI runs these checks on pushes and pull requests. CodeQL scans the Python codebase.

## Configuration

| Variable | Purpose | Default |
|---|---|---|
| `NETWORK_DATA_PATH` | Dataset path | `src/data/network_data.csv` |
| `NETWORK_MODEL_PATH` | Model path | `model/isolation_forest.pkl` |
| `NETWORK_APP_TITLE` | Application title setting | `Network Health Monitor` |

## Model notes

The current detector is an Isolation Forest with 200 estimators and contamination configured to 0.02. Its output follows scikit-learn's convention: `1` is a normal sample and `-1` is an anomaly. This is an anomaly-detection experiment, not a calibrated probability of failure.

The existing serialized model is kept in the repository so the dashboard can run without retraining first.

## Limitations and next steps

Synthetic anomalies are useful for repeatable development and testing but are not a substitute for production network traces. A production-ready system would still need real telemetry ingestion, versioned preprocessing/model artifacts, alert delivery, labeled evaluation data, authentication and audit controls, and model/data-drift monitoring.

## License

No license is currently declared in the repository. Do not assume the code is freely reusable without an explicit license.

## Contributing

Open an issue for a bug or feature proposal, then submit a focused pull request. Keep changes tested and documented.
