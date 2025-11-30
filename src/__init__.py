"""
EHR Bias Analysis - Source Code Package

This package contains reusable modules for analyzing bias in
electronic health records.
"""

__version__ = "0.1.0"
__author__ = "Javokhir Arifov"

# Import main functions for easy access
from .data_loader import (
    load_mimic_data,
    load_for_analysis,
    standardize_demographics,
    simplify_race
)

__all__ = [
    'load_mimic_data',
    'load_for_analysis',
    'standardize_demographics',
    'simplify_race',
]
