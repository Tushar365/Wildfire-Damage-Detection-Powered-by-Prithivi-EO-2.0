"""
Model package for Prithivi 2.0 wildfire damage detection.
"""

from .prithivi_model import PrithiviChangeDetector
from .processor import GeoTIFFProcessor

__all__ = ['PrithiviChangeDetector', 'GeoTIFFProcessor']
