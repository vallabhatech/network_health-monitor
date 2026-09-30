"""Central configuration for the network health monitor."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = Path(os.getenv("NETWORK_DATA_PATH", str(PROJECT_ROOT / "src" / "data" / "network_data.csv")))
MODEL_PATH = Path(os.getenv("NETWORK_MODEL_PATH", str(PROJECT_ROOT / "model" / "isolation_forest.pkl")))
APP_TITLE = os.getenv("NETWORK_APP_TITLE", "Network Health Monitor")
