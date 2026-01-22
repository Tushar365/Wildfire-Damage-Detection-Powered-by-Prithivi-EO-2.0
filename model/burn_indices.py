"""
Burn-specific spectral indices for wildfire damage detection.

This module implements standard spectral indices used in burn severity mapping,
following USGS FIREMON guidelines.
"""

import numpy as np
from typing import Tuple, Dict


# HLS Band indices (0-indexed)
HLS_BANDS = {
    'BLUE': 0,
    'GREEN': 1,
    'RED': 2,
    'NIR': 3,      # Near Infrared (Band 4)
    'SWIR1': 4,    # Shortwave Infrared 1 (Band 5)
    'SWIR2': 5,    # Shortwave Infrared 2 (Band 6)
}


def calculate_nbr(nir: np.ndarray, swir: np.ndarray) -> np.ndarray:
    """
    Calculate Normalized Burn Ratio (NBR).
    
    NBR = (NIR - SWIR) / (NIR + SWIR)
    
    Healthy vegetation has high NBR (high NIR, low SWIR).
    Burned areas have low NBR (low NIR, high SWIR from charred material).
    
    Args:
        nir: Near-infrared band (H, W)
        swir: Shortwave infrared band (H, W)
    
    Returns:
        NBR values ranging from -1 to 1
    """
    # Add small epsilon to avoid division by zero
    epsilon = 1e-10
    numerator = nir - swir
    denominator = nir + swir + epsilon
    
    return numerator / denominator


def calculate_dnbr(pre_nbr: np.ndarray, post_nbr: np.ndarray) -> np.ndarray:
    """
    Calculate delta NBR (dNBR) - the change in NBR.
    
    dNBR = pre_NBR - post_NBR
    
    Positive dNBR indicates burn (healthy vegetation became burned).
    Higher values indicate more severe burns.
    
    Args:
        pre_nbr: Pre-fire NBR
        post_nbr: Post-fire NBR
    
    Returns:
        dNBR values (positive = burn, negative = regrowth)
    """
    return pre_nbr - post_nbr


def calculate_bai(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    """
    Calculate Burned Area Index (BAI).
    
    BAI = 1 / ((0.1 - RED)^2 + (0.06 - NIR)^2)
    
    Higher values indicate burned areas (exposed soil).
    
    Args:
        red: Red band (H, W)
        nir: Near-infrared band (H, W)
    
    Returns:
        BAI values (higher = more likely burned)
    """
    epsilon = 1e-10
    red_component = (0.1 - red) ** 2
    nir_component = (0.06 - nir) ** 2
    
    bai = 1.0 / (red_component + nir_component + epsilon)
    
    return bai


def classify_burn_severity(dnbr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify burn severity based on dNBR thresholds (USGS standards).
    
    Severity Classes:
    - 0: Enhanced regrowth/unburned (dNBR < 0.1)
    - 1: Low severity (0.1 - 0.27)
    - 2: Moderate-low (0.27 - 0.44)
    - 3: Moderate-high (0.44 - 0.66)
    - 4: High severity (> 0.66)
    
    Args:
        dnbr: Delta NBR array (H, W)
    
    Returns:
        Tuple of:
            - Severity class array (0-4)
            - Statistics dictionary with percentages
    """
    severity = np.zeros_like(dnbr, dtype=np.uint8)
    
    # Apply USGS thresholds
    severity[dnbr >= 0.1] = 1   # Low
    severity[dnbr >= 0.27] = 2  # Moderate-low
    severity[dnbr >= 0.44] = 3  # Moderate-high
    severity[dnbr >= 0.66] = 4  # High severity
    
    # Calculate statistics
    total_pixels = severity.size
    stats = {
        'unburned_pct': (severity == 0).sum() / total_pixels * 100,
        'low_severity_pct': (severity == 1).sum() / total_pixels * 100,
        'moderate_low_pct': (severity == 2).sum() / total_pixels * 100,
        'moderate_high_pct': (severity == 3).sum() / total_pixels * 100,
        'high_severity_pct': (severity == 4).sum() / total_pixels * 100,
    }
    
    return severity, stats


def calculate_nbr_from_array(data: np.ndarray, use_swir2: bool = False) -> np.ndarray:
    """
    Calculate NBR from a multi-band array.
    
    Args:
        data: Multi-band array (bands, H, W)
        use_swir2: If True, use SWIR2 instead of SWIR1 (more sensitive to char)
    
    Returns:
        NBR array (H, W)
    """
    nir = data[HLS_BANDS['NIR'], :, :]
    swir = data[HLS_BANDS['SWIR2'] if use_swir2 else HLS_BANDS['SWIR1'], :, :]
    
    return calculate_nbr(nir, swir)


def detect_burn_areas_spectral(
    pre_array: np.ndarray,
    post_array: np.ndarray,
    threshold: float = 0.1,
    use_bai: bool = False
) -> Dict[str, np.ndarray]:
    """
    Detect burn areas using pure spectral analysis.
    
    Args:
        pre_array: Pre-fire image (6, H, W)
        post_array: Post-fire image (6, H, W)
        threshold: dNBR threshold for burn detection (default 0.1 = low severity)
        use_bai: Whether to also use BAI for additional validation
    
    Returns:
        Dictionary containing:
            - dnbr: Delta NBR map
            - severity: Burn severity classification (0-4)
            - mask: Binary burn mask (1 = burned)
            - stats: Severity statistics
    """
    # Calculate NBR for pre and post
    pre_nbr = calculate_nbr_from_array(pre_array)
    post_nbr = calculate_nbr_from_array(post_array)
    
    # Calculate change
    dnbr = calculate_dnbr(pre_nbr, post_nbr)
    
    # Classify severity
    severity, stats = classify_burn_severity(dnbr)
    
    # Create binary mask
    mask = (dnbr >= threshold).astype(np.uint8)
    
    # Optional: filter with BAI to reduce false positives
    if use_bai:
        post_red = post_array[HLS_BANDS['RED'], :, :]
        post_nir = post_array[HLS_BANDS['NIR'], :, :]
        bai = calculate_bai(post_red, post_nir)
        
        # Require both high dNBR AND high BAI
        bai_threshold = np.percentile(bai, 75)  # Top 25% of BAI values
        mask = mask & (bai > bai_threshold)
    
    return {
        'dnbr': dnbr,
        'severity': severity,
        'mask': mask,
        'stats': stats,
    }
