"""Create windowed feature datasets for model training.

This module converts a sequential tracking DataFrame into window-level feature
vectors, labels, and metadata suitable for supervised learning.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.features.features import extract_window_features


def build_window_meta_dataset(
    df: pd.DataFrame,
    window_size: int,
    stride: int,
    label_mode: str = "any",
    label_col: str = "is_pass",
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Build window-level dataset, labels, and metadata from tracking data.

    Args:
        df: Tracking DataFrame containing timestamp, position, and label columns.
        window_size: Number of frames in each sliding window.
        stride: Step size used to slide the window over the DataFrame.
        label_mode: How to label the window: ``any`` if any frame contains a pass,
            or ``center`` to use the center frame label.
        label_col: Column name containing the binary pass label.

    Returns:
        A tuple of ``(X, y, meta)`` where ``X`` is a DataFrame of window-level
        features, ``y`` is a NumPy array of window labels, and ``meta`` is a
        DataFrame containing window metadata.
    """
    X = []
    y = []
    meta = []

    center_offset = window_size // 2

    for start_idx in range(0, len(df) - window_size + 1, stride):
        end_idx = start_idx + window_size
        center_idx = start_idx + center_offset

        window = df.iloc[start_idx:end_idx]

        # Extract window-level features from the current frame sequence.
        features = extract_window_features(window)

        # Window label based on the selected labeling strategy.
        if label_mode == "any":
            label = int(window[label_col].max())
        elif label_mode == "center":
            label = int(df[label_col].iloc[center_idx])
        else:
            label = None
            print("Unkown label_mode. Choices are 'any' or 'center'")

        # Identify all real pass frames inside this window.
        pass_rows = window[window[label_col] == 1]
        pass_count = len(pass_rows)

        pass_frame_idxs = pass_rows.index.to_list()
        pass_timestamps = (
            pass_rows["timestamp"].to_list() if "timestamp" in pass_rows.columns else []
        )

        # Store
        X.append(features)
        y.append(label)

        meta.append(
            {
                # Basic window identity
                "window_start_idx": start_idx,
                "window_center_idx": center_idx,
                "window_end_idx": end_idx - 1,
                # Window timestamps
                "start_timestamp": df["timestamp"].iloc[start_idx],
                "center_timestamp": df["timestamp"].iloc[center_idx],
                "end_timestamp": df["timestamp"].iloc[end_idx - 1],
                # Ball position at window center
                "x_center": df["posX"].iloc[center_idx],
                "y_center": df["posY"].iloc[center_idx],
                # Label info
                "window_label": label,
                "has_pass": int(pass_count > 0),
                "pass_count": pass_count,
                # Pass locations inside window
                "pass_frame_idxs": pass_frame_idxs,
                "pass_timestamps": pass_timestamps,
            }
        )

    X = pd.DataFrame(X)
    y = np.array(y)
    meta = pd.DataFrame(meta)

    return X, y, meta


def apply_nan_mask(
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Filter rows with too many missing values.

    Args:
        X: Feature matrix.
        y: Target vector aligned with ``X``.
        threshold: Maximum allowed proportion of NaNs per row.

    Returns:
        Filtered ``X`` and ``y`` with only rows below the missing-value
        threshold.
    """
    nan_ratio = np.isnan(X).mean(axis=1)
    mask = nan_ratio < threshold
    return X[mask], y[mask]


def apply_nan_mask_with_meta(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    max_nan_share: float = 0.5,
) -> tuple[pd.DataFrame | np.ndarray, np.ndarray, pd.DataFrame]:
    """Filter windows by NaN share and align ``X``, ``y``, and ``meta``.

    Args:
        X: Feature matrix or DataFrame.
        y: Target vector aligned with ``X``.
        meta: Metadata DataFrame aligned with ``X`` and ``y``.
        max_nan_share: Maximum allowed proportion of missing values per window.

    Returns:
        Cleaned ``X``, ``y``, and ``meta`` after filtering.
    """

    if isinstance(X, pd.DataFrame):
        nan_share = X.isna().mean(axis=1)
        mask = nan_share <= max_nan_share
        X_clean = X.loc[mask].reset_index(drop=True)
    else:
        nan_share = np.isnan(X).mean(axis=1)
        mask = nan_share <= max_nan_share
        X_clean = X[mask]

    y_clean = np.asarray(y)[mask]
    meta_clean = meta.loc[mask].reset_index(drop=True)

    return X_clean, y_clean, meta_clean


def check_split_pass_counts_and_actual_passes(
    y: np.ndarray,
    meta: pd.DataFrame,
    split_name: str,
) -> None:
    """Print statistics for a split and compare pass windows to actual passes.

    Args:
        y: Target vector for the split.
        meta: Metadata table containing ``pass_count`` for each window.
        split_name: Name of the split for display.
    """
    y = np.asarray(y)

    n_total = len(y)
    n_pass_windows = np.sum(y == 1)
    n_no_pass_windows = np.sum(y == 0)

    n_actual_passes = meta["pass_count"].sum()
    n_extra_passes = n_actual_passes - n_pass_windows

    print(f"\n{split_name}")
    print("-" * len(split_name))
    print(f"Total windows:              {n_total}")
    print(f"Pass windows:               {n_pass_windows}")
    print(f"No-pass windows:            {n_no_pass_windows}")
    print(f"Pass-window percentage:     {n_pass_windows / n_total * 100:.2f}%")
    print(f"No-pass-window percentage:  {n_no_pass_windows / n_total * 100:.2f}%")
    print(f"Actual passes:              {n_actual_passes}")
    print(f"Extra passes compressed:    {n_extra_passes}")

    if n_actual_passes > 0:
        print(
            f"Compression percentage:     {n_extra_passes / n_actual_passes * 100:.2f}%"
        )


def check_stats(matches: list[dict[str, Any]], name: str = "DATA") -> tuple[int, int]:
    """Print statistics for a set of match datasets.

    Args:
        matches: List of match dictionaries produced by the pipeline.
        name: Label to use when printing statistics.

    Returns:
        Tuple with the count of no-pass and pass windows.
    """
    total_rows = 0
    total_nans = 0
    all_y = []

    for m in matches:
        X = m["X"]
        y = m["y"]

        total_rows += len(X)

        if isinstance(X, pd.DataFrame):
            total_nans += X.isna().sum().sum()
        else:
            total_nans += np.isnan(X).sum()

        all_y.append(y)

    # Consolidate labels across all matches and compute class counts.
    all_y = np.concatenate(all_y)
    counts = pd.Series(all_y).value_counts().sort_index()

    n_no_pass = counts.get(0, 0)
    n_pass = counts.get(1, 0)
    n_total = n_no_pass + n_pass

    print(f"\n{name} STATS")
    print(f"Rows: {total_rows}")
    print(f"Total NaNs: {total_nans}")
    print(f"NaNs per row (avg): {total_nans / total_rows:.4f}")

    print("\nClass distribution:")
    print(counts)

    print(f"\nNo-pass windows:   {n_no_pass}")
    print(f"Pass windows:      {n_pass}")
    print(f"Pass percentage:   {n_pass / n_total * 100:.2f}%")
    print(f"No-pass percentage:{n_no_pass / n_total * 100:.2f}%")

    return n_no_pass, n_pass
