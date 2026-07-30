"""Root conftest.py.

Its sole purpose is to guarantee the repository root is on `sys.path` during
test collection (regardless of how pytest is invoked — `pytest`, `python -m
pytest`, or `uv run pytest`), so `from src.data... import ...` works
consistently across environments.
"""
