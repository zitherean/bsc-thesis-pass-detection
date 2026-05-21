"""Visualization helpers for pipeline outputs.

This module contains helpers for collecting pass-location data across
train/validation/test splits and plotting model feature importance.
"""

from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

###################################### VISUALIZATION #####################################


def collect_pass_locations_from_split(
    matches: list[dict[str, Any]],
    split_name: str,
) -> pd.DataFrame:
    """Aggregate pass locations across a data split for plotting.

    Args:
        matches: List of match dictionaries containing ``real_passes``.
        split_name: Name of the split to tag each pass location.

    Returns:
        Concatenated DataFrame of pass locations across all matches.
    """
    all_locations = []

    for i, match in enumerate(matches):
        real_passes = match["real_passes"].copy()

        match_name = match.get(
            "match_name", match.get("name", f"{split_name}_match_{i}")
        )

        # Normalize pass coordinate columns for consistent plotting.
        if "posX" in real_passes.columns and "posY" in real_passes.columns:
            real_passes = real_passes.rename(
                columns={"posX": "x_center", "posY": "y_center"}
            )

        real_passes["split"] = split_name
        real_passes["match"] = match_name

        all_locations.append(real_passes)

    return pd.concat(all_locations, ignore_index=True)


def collect_filtered_pass_locations_from_split(
    matches: list[dict[str, Any]],
    split_name: str,
) -> pd.DataFrame:
    """Collect pass locations from windows retained after filtering.

    Args:
        matches: List of match dictionaries containing ``real_passes``, ``meta``, and ``y``.
        split_name: Name of the split for display.

    Returns:
        Concatenated DataFrame of filtered pass locations across matches.
    """
    all_locations = []

    for i, match in enumerate(matches):
        real_passes = match["real_passes"].copy()
        meta = match["meta"].copy()
        y = np.asarray(match["y"])

        match_name = match.get(
            "match_name", match.get("name", f"{split_name}_match_{i}")
        )

        # Normalize pass coordinate columns for consistent plotting.
        if "posX" in real_passes.columns and "posY" in real_passes.columns:
            real_passes = real_passes.rename(
                columns={"posX": "x_center", "posY": "y_center"}
            )

        # Only retain pass frame indices for windows labeled as positive.
        positive_meta = meta[y == 1].copy()

        kept_pass_frame_idxs = (
            positive_meta["pass_frame_idxs"].explode().dropna().astype(int).unique()
        )

        # Keep only real passes that were still present after filtering.
        filtered_passes = real_passes[
            real_passes["frame_idx"].astype(int).isin(kept_pass_frame_idxs)
        ].copy()

        filtered_passes["split"] = split_name
        filtered_passes["match"] = match_name

        all_locations.append(filtered_passes)

    return pd.concat(all_locations, ignore_index=True)


def plot_top_feature_importance(
    importance_series: pd.Series,
    model_name: str,
    top_features: Optional[list[str]] = None,
    top_n: int = 10,
    figsize: tuple = (8, 5),
) -> None:
    """Plot a horizontal bar chart of the top feature importances.

    Args:
        importance_series: Feature importance values with feature names as the index.
        model_name: Name of the model, used in the plot title.
        top_features: Specific features to plot. If None, the top_n features are selected.
        top_n: Number of top features to plot if top_features is None.
        figsize: Figure size.
    """

    if not isinstance(importance_series, pd.Series):
        raise TypeError(
            "importance_series must be a pandas Series with feature names as the index."
        )

    if top_features is None:
        plot_data = importance_series.sort_values(ascending=False).head(top_n)
    else:
        available_features = [f for f in top_features if f in importance_series.index]
        missing_features = [f for f in top_features if f not in importance_series.index]

        if missing_features:
            print("Warning: these features were not found and will be skipped:")
            print(missing_features)

        plot_data = importance_series.loc[available_features]

    # Plot features from smallest to largest for a horizontal bar chart.
    plot_data = plot_data.sort_values(ascending=True)

    plt.figure(figsize=figsize)
    plt.barh(plot_data.index, plot_data.values)
    plt.xlabel("Permutation importance")
    plt.ylabel("Feature")
    plt.title(f"Top Feature Importances for {model_name}")
    plt.tight_layout()
    plt.show()
