"""Data backends: deterministic synthetic market (default), CSV, optional yfinance."""

from finsight.data.factory import load_panel
from finsight.data.synthetic import generate_synthetic_panel

__all__ = ["generate_synthetic_panel", "load_panel"]
