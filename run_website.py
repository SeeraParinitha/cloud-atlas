"""Run this file in VS Code using this project's .venv Python interpreter.

Models load automatically from local_models; no ZIP uploads are required.
Complete local_settings.json once before starting.
"""
from scripts.run_local_web import main


if __name__ == '__main__':
    raise SystemExit(main())
