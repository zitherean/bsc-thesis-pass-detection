"""Utility helpers for data processing, model persistence, and plotting.

This module provides small helper functions used across the project, including
safe array reduction functions, dataset validation, team mapping, model save/load
support, and visualization helpers.
"""

from typing import Any

import warnings
import numpy as np
import pandas as pd

###################################### UTILS #####################################


def get_player_id_cols(df: pd.DataFrame) -> list[str]:
    """Return player ID columns from a DataFrame.

    Args:
        df: DataFrame containing player tracking columns.

    Returns:
        List of column names that correspond to player IDs.
    """
    return [c for c in df.columns if c == "playerId" or c.startswith("playerId.")]


def safe_nanmin(arr: Any, axis: int = 0) -> Any:
    """Compute the minimum over an array while preserving all-NaN axes.

    Args:
        arr: Input array-like object.
        axis: Axis along which to compute the minimum.

    Returns:
        Minimum values along the specified axis, with NaN propagated for
        slices that are entirely NaN.
    """
    arr = np.asarray(arr)

    # Track positions where the slice contains only NaNs.
    all_nan_mask = np.all(np.isnan(arr), axis=axis)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        result = np.nanmin(arr, axis=axis)

    if np.isscalar(result):
        return np.nan if all_nan_mask else result

    # Preserve NaN where every value along the axis was missing.
    result[all_nan_mask] = np.nan
    return result


def safe_nanargmin(arr: Any, axis: int = 0) -> Any:
    """Compute the argmin over an array while handling all-NaN slices.

    Args:
        arr: Input array-like object.
        axis: Axis along which to compute the argmin.

    Returns:
        Indices of the minimum values, with NaN for slices that are entirely
        NaN.
    """
    arr = np.asarray(arr)

    # Track positions where the slice contains only NaNs.
    all_nan_mask = np.all(np.isnan(arr), axis=axis)

    filled = np.where(np.isnan(arr), np.inf, arr)
    result = np.argmin(filled, axis=axis).astype(float)

    if np.isscalar(result):
        return np.nan if all_nan_mask else result

    # Preserve NaN where every value along the axis was missing.
    result[all_nan_mask] = np.nan
    return result
