"""Add the project root to sys.path so ``from src...`` works for scripts run as
``python scripts/foo.py`` (not just ``python -m``), and force UTF-8 stdout so
Turkish text / unicode prints on Windows consoles (cp1254/cp1252)."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")  # py3.7+
    except (AttributeError, ValueError):
        pass
