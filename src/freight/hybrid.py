"""Hybrid model entry point: Ridge (distance, equipment, market_index) + LightGBM on its residuals, with a no-lane-stat fallback.

The implementation lives in models.py; it is re-exported here so the module layout matches the documentation.
"""
from .models import HybridModel, RoutedModel
from .predict import make_final_model

__all__ = ["HybridModel", "RoutedModel", "make_final_model"]
