# Add other events in this file

"""Label pass frames in tracking data using pass event timestamps."""

import pandas as pd


def label_passes(
    tracking_df: pd.DataFrame, pass_events_df: pd.DataFrame
) -> pd.DataFrame:
    """Mark tracking frames that correspond to pass events.

    A new ``is_pass`` column is added to a copy of the tracking DataFrame.
    Frames with timestamps matching pass events are labeled with 1; all others
    are labeled with 0.

    Args:
        tracking_df: Tracking DataFrame containing a ``timestamp`` column.
        pass_events_df: Event DataFrame containing ``Type`` and ``timestamp``
            columns.

    Returns:
        A copy of ``tracking_df`` with an ``is_pass`` indicator column.
    """
    tracking_df = tracking_df.copy()
    tracking_df["is_pass"] = 0
    pass_timestamps = pass_events_df.loc[pass_events_df["Type"] == "Pass", "timestamp"]
    tracking_df["is_pass"] = tracking_df["timestamp"].isin(pass_timestamps).astype(int)

    return tracking_df
