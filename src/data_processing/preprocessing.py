"""Preprocessing helpers for tracking and positional data.

This module includes functions to normalize data types, remove bad rows, and
interpolate missing coordinates in tracking datasets.
"""

import pandas as pd
import numpy as np
from pandas import Series

from src.constants import (
    NON_NUMERIC_COLUMN,
    TIMESTAMP_COLUMN,
    PITCH_LENGTH_MAX,
    PITCH_WIDTH_MAX,
    PITCH_LENGTH_MIN,
    PITCH_WIDTH_MIN,
    TOLERANCE,
)


def convert_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all numeric columns in the DataFrame to numeric dtype.

    Non-numeric placeholders such as '-' and ' - ' are replaced with NaN
    before converting. The column listed in ``NON_NUMERIC_COLUMN`` is left
    unchanged.

    Args:
        df: Input DataFrame to normalize.

    Returns:
        A copy of the DataFrame with numeric columns converted to numeric dtypes.
    """
    df = df.replace(["-", " - "], np.nan)
    # drop the column used for non-numeric identifiers before conversion.
    cols = df.columns.drop(NON_NUMERIC_COLUMN)
    df[cols] = df[cols].apply(pd.to_numeric)
    return df


def remove_duplicate_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows based on the timestamp column.

    Duplicate timestamps should not exist in tracking frames, so this is a
    defensive cleanup step.

    Args:
        df: Input DataFrame to deduplicate.

    Returns:
        DataFrame with duplicate timestamps removed.
    """
    df = df.drop_duplicates(TIMESTAMP_COLUMN)
    return df


def sort_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Sort the DataFrame by timestamp and reset the index.

    Args:
        df: Input DataFrame with a timestamp column.

    Returns:
        DataFrame sorted by ``TIMESTAMP_COLUMN`` with a fresh integer index.
    """
    df = df.sort_values(TIMESTAMP_COLUMN).reset_index(drop=True)
    return df


def clean_out_of_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """Clean out-of-bounds coordinates with tolerance and clipping.

    Coordinates that lie far outside the pitch plus tolerance are set to NaN,
    while small overshoots are clipped to the nearest allowed boundary.

    Args:
        df: Input DataFrame containing position columns.

    Returns:
        DataFrame with cleaned and clipped coordinate values.
    """
    x_cols = [c for c in df.columns if c.startswith("posX")]
    y_cols = [c for c in df.columns if c.startswith("posY")]

    for x in x_cols:
        df.loc[
            (df[x] < PITCH_LENGTH_MIN - TOLERANCE)
            | (df[x] > PITCH_LENGTH_MAX + TOLERANCE),
            x,
        ] = np.nan
        df[x] = df[x].clip(PITCH_LENGTH_MIN, PITCH_LENGTH_MAX)

    for y in y_cols:
        df.loc[
            (df[y] < PITCH_WIDTH_MIN - TOLERANCE)
            | (df[y] > PITCH_WIDTH_MAX + TOLERANCE),
            y,
        ] = np.nan
        df[y] = df[y].clip(PITCH_WIDTH_MIN, PITCH_WIDTH_MAX)

    return df


def interpolate_xy_coordinates(
    x: Series, y: Series, max_gap: int
) -> tuple[Series, Series]:
    """Interpolate missing XY coordinate pairs for short gaps.

    Only gaps where both X and Y are missing are interpolated, and only if the
    gap length does not exceed ``max_gap``. Leading and trailing missing values
    are preserved.

    Args:
        x: X-coordinate series.
        y: Y-coordinate series.
        max_gap: Maximum allowed gap length to interpolate.

    Returns:
        A tuple containing the interpolated X and Y series.
    """
    # frames where position is missing
    missing = x.isna() | y.isna()

    # consecutive run id over the missing mask
    run_id = missing.ne(missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform("size")

    large_gap = missing & (run_len > max_gap)

    # interpolate
    x_i = x.interpolate(method="linear", limit_area="inside")
    y_i = y.interpolate(method="linear", limit_area="inside")

    # revert large gaps back to NaN in BOTH series
    x_i[large_gap] = np.nan
    y_i[large_gap] = np.nan

    # if after interpolation a coordinate is still NaN, force both coordinates to NaN
    inconsistent = x_i.isna() ^ y_i.isna()
    x_i[inconsistent] = np.nan
    y_i[inconsistent] = np.nan

    return x_i, y_i


def interpolate_all_coordinates(df: pd.DataFrame, max_gap: int) -> pd.DataFrame:
    """Interpolate all paired coordinate columns in the DataFrame.

    Each ``posX`` column is paired with a corresponding ``posY`` column and
    interpolated together to preserve coordinate consistency.

    Args:
        df: DataFrame containing position columns.
        max_gap: Maximum allowed gap length for interpolation.

    Returns:
        DataFrame with interpolated coordinate columns.
    """
    df = df.copy()

    x_cols = [col for col in df.columns if col.startswith("posX")]

    for x_col in x_cols:
        y_col = x_col.replace("posX", "posY")
        if y_col in df.columns:
            df[x_col], df[y_col] = interpolate_xy_coordinates(
                df[x_col], df[y_col], max_gap
            )

    return df
