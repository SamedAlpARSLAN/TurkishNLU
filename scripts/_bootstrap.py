"""Add the project root to sys.path so ``from src...`` works for scripts run as
``python scripts/foo.py`` (not just ``python -m``)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
