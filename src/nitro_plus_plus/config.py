"""Project-wide constants and paths used across modules."""

from pathlib import Path

# Resolve project root: <repo>/src/nitro_plus_plus/... -> root is 3 up
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

# External tool sources
NIPA_REPO = "https://github.com/Wilhansen/nipa"
