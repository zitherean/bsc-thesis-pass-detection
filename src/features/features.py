"""Feature engineering helpers for ball and player tracking data.

This module computes motion, trajectory, interaction, and window-level
statistics for tracking datasets.
"""

from collections.abc import Mapping

import pandas as pd
import numpy as np


from src.constants import (
    SMOOTHING,
    FRAMES_PER_SECOND,
    CENTER_FRAME,
    START_FRAME,
    END_FRAME,
    PITCH_LENGTH_MIN,
    PITCH_WIDTH_MIN,
)
from src.utils import safe_nanmin, safe_nanargmin, get_player_id_cols

################################### BALL MOTION FEATURES ########################################


def add_ball_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ball displacement between consecutive frames.

    Args:
        df: Tracking DataFrame containing ``posX`` and ``posY`` columns.

    Returns:
        DataFrame with ``ball_dx``, ``ball_dy``, and ``ball_displacement``.
    """
    df = df.copy()
    df["ball_dx"] = df["posX"].diff()
    df["ball_dy"] = df["posY"].diff()
    df["ball_displacement"] = np.sqrt(df["ball_dx"] ** 2 + df["ball_dy"] ** 2)
    return df


def add_ball_velocity(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ball velocity components and speed.

    Args:
        df: Tracking DataFrame containing ``posX`` and ``posY`` columns.

    Returns:
        DataFrame with ``ball_vx``, ``ball_vy``, and ``ball_speed``.
    """
    df = df.copy()
    dt = 1 / FRAMES_PER_SECOND
    df["ball_vx"] = df["posX"].diff() / dt
    df["ball_vy"] = df["posY"].diff() / dt
    df["ball_speed"] = np.sqrt(df["ball_vx"] ** 2 + df["ball_vy"] ** 2)
    return df


def add_ball_acceleration(df: pd.DataFrame) -> pd.DataFrame:
    """Compute ball acceleration from speed changes.

    Args:
        df: Tracking DataFrame containing ``ball_speed``.

    Returns:
        DataFrame with ``ball_acc``.
    """
    df = df.copy()
    dt = 1 / FRAMES_PER_SECOND
    df["ball_acc"] = df["ball_speed"].diff() / dt
    return df


def add_ball_speed_prev(df: pd.DataFrame) -> pd.DataFrame:
    """Add a lagged ball speed value from three frames earlier.

    Args:
        df: Tracking DataFrame containing ``ball_speed``.

    Returns:
        DataFrame with ``ball_speed_prev``.
    """
    df = df.copy()
    df["ball_speed_prev"] = df["ball_speed"].shift(3)
    return df


################################## BALL TRAJECTORY FEATURES ########################################


def add_ball_direction_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute direction change of the ball between consecutive frames.

    Args:
        df: Tracking DataFrame containing ``ball_vx`` and ``ball_vy``.

    Returns:
        DataFrame with ``ball_direction_change``.
    """
    df = df.copy()

    vx = df["ball_vx"]
    vy = df["ball_vy"]

    vx_prev = vx.shift(1)
    vy_prev = vy.shift(1)

    dot = vx * vx_prev + vy * vy_prev
    mag = np.sqrt(vx**2 + vy**2) * np.sqrt(vx_prev**2 + vy_prev**2)

    cos_angle = dot / (mag + SMOOTHING)
    cos_angle = np.clip(cos_angle, -1, 1)

    df["ball_direction_change"] = 1 - cos_angle

    return df


############################# BALL-PLAYER INTERACTION FEATURES #################################


def add_ball_player_proximity_flags(
    df: pd.DataFrame, close_radius: float = 1.5
) -> pd.DataFrame:
    """Label frames where the ball is close to a player.

    Args:
        df: Tracking DataFrame containing ``dist_ball_nearest_player``.
        close_radius: Distance threshold to consider a player close to the ball.

    Returns:
        DataFrame with ``ball_player_close`` as an indicator column.
    """
    df = df.copy()
    df["ball_player_close"] = (df["dist_ball_nearest_player"] <= close_radius).astype(
        int
    )
    return df


def add_distance_to_nearest_player(
    df: pd.DataFrame, player_team_map: Mapping[object, object]
) -> pd.DataFrame:
    """Compute distance and identity of the nearest player to the ball.

    Args:
        df: Tracking DataFrame with player position columns like ``posX.<id>``.
        player_team_map: Mapping from player IDs to team labels.

    Returns:
        DataFrame with ``dist_ball_nearest_player``, ``closest_player_id``, and
        ``closest_player_team``.
    """
    df = df.copy()

    player_x_cols = [c for c in df.columns if c.startswith("posX.")]
    player_y_cols = [c for c in df.columns if c.startswith("posY.")]
    player_id_cols = get_player_id_cols(df)

    ball_x = df["posX"]
    ball_y = df["posY"]

    dists = []

    for x_col, y_col in zip(player_x_cols, player_y_cols):
        dist = np.sqrt((df[x_col] - ball_x) ** 2 + (df[y_col] - ball_y) ** 2)
        dists.append(dist)

    dist_array = np.vstack(dists).T

    # Distance to nearest player
    df["dist_ball_nearest_player"] = safe_nanmin(dist_array, axis=1)

    # Slot of nearest player
    closest_player_slot = safe_nanargmin(dist_array, axis=1)

    closest_ids = []

    for row_idx, slot in enumerate(closest_player_slot):
        if np.isnan(df["dist_ball_nearest_player"].iloc[row_idx]):
            closest_ids.append(np.nan)
        else:
            player_id_col = player_id_cols[int(slot)]
            closest_ids.append(df[player_id_col].iloc[row_idx])

    df["closest_player_id"] = closest_ids
    df["closest_player_team"] = df["closest_player_id"].map(player_team_map)

    return df


def add_own_and_opposition_players_near_ball(
    df: pd.DataFrame,
    player_team_map: Mapping[object, object],
    radius: float = 6,
) -> pd.DataFrame:
    """Count own and opposing players close to the ball.

    Args:
        df: Tracking DataFrame with player position columns.
        player_team_map: Mapping from player IDs to team labels.
        radius: Proximity threshold in the same units as positions.

    Returns:
        DataFrame with ``own_players_near_ball`` and
        ``opposition_players_near_ball``.
    """
    df = df.copy()

    player_x_cols = [c for c in df.columns if c.startswith("posX.")]
    player_y_cols = [c for c in df.columns if c.startswith("posY.")]
    player_id_cols = get_player_id_cols(df)

    ball_x = df["posX"]
    ball_y = df["posY"]

    own_counts = np.zeros(len(df))
    opposition_counts = np.zeros(len(df))

    closest_team = df["closest_player_team"]

    for x_col, y_col, id_col in zip(player_x_cols, player_y_cols, player_id_cols):
        player_team = df[id_col].map(player_team_map)

        dist = np.sqrt((df[x_col] - ball_x) ** 2 + (df[y_col] - ball_y) ** 2)
        near_ball = dist < radius

        same_team = player_team == closest_team
        opposition_team = player_team != closest_team

        valid = closest_team.notna() & player_team.notna()

        own_counts += near_ball & same_team & valid
        opposition_counts += near_ball & opposition_team & valid

    df["own_players_near_ball"] = own_counts
    df["opposition_players_near_ball"] = opposition_counts

    return df


def add_distance_to_player_change(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate frame-to-frame change in nearest-player distance.

    Args:
        df: Tracking DataFrame containing ``dist_ball_nearest_player``.

    Returns:
        DataFrame with ``dist_ball_nearest_player_change``.
    """
    df = df.copy()
    df["dist_ball_nearest_player_change"] = df["dist_ball_nearest_player"].diff()
    return df


def add_closest_player_streak(
    df: pd.DataFrame, max_ball_speed: float = 5.0
) -> pd.DataFrame:
    """Compute streak lengths for the same closest player.

    Streaks are reset to zero when the closest player changes or when the ball
    is moving too fast.

    Args:
        df: Tracking DataFrame containing ``closest_player_id`` and ``ball_speed``.
        max_ball_speed: Maximum speed at which a streak is valid.

    Returns:
        DataFrame with ``closest_player_streak``.
    """
    df = df.copy()

    streak = np.zeros(len(df), dtype=float)
    ids = df["closest_player_id"].to_numpy()
    speeds = df["ball_speed"].to_numpy()

    for i in range(1, len(df)):
        if np.isnan(ids[i]) or np.isnan(ids[i - 1]):
            streak[i] = np.nan
        elif np.isnan(speeds[i]) or speeds[i] >= max_ball_speed:
            streak[i] = 0
        elif ids[i] == ids[i - 1]:
            if np.isnan(streak[i - 1]):
                streak[i] = 1
            else:
                streak[i] = streak[i - 1] + 1
        else:
            streak[i] = 0

    df["closest_player_streak"] = streak
    return df


################################### EXTRACT WINDOW FEATURES ########################################


def extract_window_features(window: pd.DataFrame) -> dict:
    """Extract summary features from a fixed-length window of tracking frames.

    Args:
        window: DataFrame representing a window of tracking frames.

    Returns:
        A dictionary of aggregated features from the window.
    """
    features = {}

    # Position summaries
    ball_x = window["posX"]
    ball_y = window["posY"]
    features["x_center"] = ball_x.iloc[CENTER_FRAME]
    features["y_center"] = ball_y.iloc[CENTER_FRAME]
    features["ball_displacement_start_end"] = np.sqrt(
        (ball_x.iloc[END_FRAME] - ball_x.iloc[START_FRAME]) ** 2
        + (ball_y.iloc[END_FRAME] - ball_y.iloc[START_FRAME]) ** 2
    )
    features["ball_displacement_from_pitch_origin"] = np.sqrt(
        (ball_x.iloc[CENTER_FRAME] - PITCH_LENGTH_MIN) ** 2
        + (ball_y.iloc[CENTER_FRAME] - PITCH_WIDTH_MIN) ** 2
    )

    # Ball displacement
    displacement = window["ball_displacement"]
    features["displacement_std"] = displacement.std()
    features["displacement_var"] = displacement.var()

    # Ball speed
    speed = window["ball_speed"]
    features["speed_max"] = speed.max()
    features["speed_mean"] = speed.mean()
    features["speed_std"] = speed.std()
    features["ball_speed_center"] = speed.iloc[CENTER_FRAME]
    features["ball_speed_start"] = speed.iloc[START_FRAME]
    features["ball_speed_end"] = speed.iloc[END_FRAME]
    features["speed_end_minus_start"] = speed.iloc[END_FRAME] - speed.iloc[START_FRAME]

    # Ball acceleration
    acc = window["ball_acc"]
    features["acc_max"] = acc.max()
    features["acc_min"] = acc.min()
    features["acc_abs_max"] = acc.abs().max()
    features["acc_abs_mean"] = acc.abs().mean()

    # Ball previous speed
    speed_prev = window["ball_speed_prev"]
    features["speed_prev_mean"] = speed_prev.mean()
    features["speed_prev_std"] = speed_prev.std()

    # Ball direction
    direction = window["ball_direction_change"]
    features["direction_max"] = direction.max()
    features["direction_min"] = direction.min()
    features["direction_mean"] = direction.mean()
    features["direction_std"] = direction.std()

    # Distance to nearest player
    dist = window["dist_ball_nearest_player"]
    features["dist_max"] = dist.max()
    features["dist_min"] = dist.min()
    features["dist_mean"] = dist.mean()
    features["dist_center"] = dist.iloc[CENTER_FRAME]
    features["dist_start"] = dist.iloc[START_FRAME]
    features["dist_end"] = dist.iloc[END_FRAME]
    features["dist_end_minus_start"] = dist.iloc[END_FRAME] - dist.iloc[START_FRAME]
    features["dist_close_start"] = int(dist.iloc[START_FRAME] <= 1.5)
    features["dist_close_end"] = int(dist.iloc[END_FRAME] <= 1.5)

    # Closest player
    closest_player = window["closest_player_id"]
    features["same_closest_player_all_window"] = int(
        closest_player.nunique(dropna=True) == 1
    )

    closest_player_valid = closest_player.dropna()
    if len(closest_player_valid) > 1:
        features["closest_player_changes"] = int(
            (
                closest_player_valid.iloc[1:].values
                != closest_player_valid.iloc[:-1].values
            ).sum()
        )
    else:
        features["closest_player_changes"] = 0

    start_id = closest_player.iloc[START_FRAME]
    end_id = closest_player.iloc[END_FRAME]

    valid_start_end = pd.notna(start_id) and pd.notna(end_id)

    features["closest_player_start_end_valid"] = int(valid_start_end)

    if valid_start_end:
        features["closest_player_start_end_changed"] = int(start_id != end_id)
    else:
        features["closest_player_start_end_changed"] = 0

    # Closest team
    closest_team = window["closest_player_team"]
    team_counts = closest_team.value_counts(normalize=True)
    if len(team_counts) > 0:
        features["closest_team_share_max"] = team_counts.max()
    else:
        features["closest_team_share_max"] = 0

    closest_team_valid = closest_team.dropna()
    if len(closest_team_valid) > 1:
        features["closest_team_changes"] = int(
            (
                closest_team_valid.iloc[1:].values
                != closest_team_valid.iloc[:-1].values
            ).sum()
        )
    else:
        features["closest_team_changes"] = 0

    start_team = closest_team.iloc[START_FRAME]
    end_team = closest_team.iloc[END_FRAME]

    valid_team_start_end = pd.notna(start_team) and pd.notna(end_team)

    features["closest_team_start_end_valid"] = int(valid_team_start_end)

    if valid_team_start_end:
        features["closest_team_start_end_changed"] = int(start_team != end_team)
    else:
        features["closest_team_start_end_changed"] = 0

    # Ball and player near each other
    very_close = window["ball_player_close"]
    features["ball_player_close_share"] = very_close.mean()
    features["ball_player_close_any"] = int(very_close.any())

    # Closest player streak
    closest_streak = window["closest_player_streak"]
    features["streak_max"] = closest_streak.max()
    features["streak_min"] = closest_streak.min()
    features["streak_mean"] = closest_streak.mean()
    features["streak_std"] = closest_streak.std()
    features["streak_center"] = closest_streak.iloc[CENTER_FRAME]
    features["streak_start"] = closest_streak.iloc[START_FRAME]
    features["streak_end"] = closest_streak.iloc[END_FRAME]
    features["streak_end_minus_start"] = (
        closest_streak.iloc[END_FRAME] - closest_streak.iloc[START_FRAME]
    )

    # Own players near ball
    own_players = window["own_players_near_ball"]
    features["own_players_near_max"] = own_players.max()
    features["own_players_near_min"] = own_players.min()
    features["own_players_near_mean"] = own_players.mean()

    # Opposition players near ball
    oppo_players = window["opposition_players_near_ball"]
    features["oppo_players_near_max"] = oppo_players.max()
    features["oppo_players_near_min"] = oppo_players.min()
    features["oppo_players_near_mean"] = oppo_players.mean()

    # Distance change
    player_change = window["dist_ball_nearest_player_change"]
    features["dist_player_change_mean"] = player_change.mean()
    features["dist_player_change_max"] = player_change.max()
    features["dist_player_change_min"] = player_change.min()
    features["dist_player_change_std"] = player_change.std()

    # Straight-line movement
    dx = ball_x.diff()
    dy = ball_y.diff()

    step_dist = np.sqrt(dx**2 + dy**2)
    path_length = step_dist.sum(min_count=1)
    straight_distance = np.sqrt(
        (ball_x.iloc[END_FRAME] - ball_x.iloc[START_FRAME]) ** 2
        + (ball_y.iloc[END_FRAME] - ball_y.iloc[START_FRAME]) ** 2
    )

    features["ball_path_length"] = path_length
    features["ball_straight_distance"] = straight_distance
    features["ball_curve_distance"] = path_length - straight_distance
    features["ball_movement_directness"] = straight_distance / (path_length + SMOOTHING)
    features["ball_curve_ratio"] = (path_length - straight_distance) / (
        path_length + SMOOTHING
    )

    # Derived features
    features["speed_distance_interaction"] = speed.mean() * dist.mean()
    features["speed_max_per_path_length"] = speed.max() / (path_length + SMOOTHING)
    features["own_opposition_ratio_near_center"] = own_players.mean() / (
        oppo_players.mean() + SMOOTHING
    )

    features["nearest_dist_opening"] = dist.max() - dist.iloc[START_FRAME]
    features["nearest_dist_closing"] = dist.max() - dist.iloc[END_FRAME]
    features["nearest_dist_pass_arc"] = dist.max() - (
        (dist.iloc[START_FRAME] + dist.iloc[END_FRAME]) / 2
    )

    features["player_changed_x_speed_max"] = (
        features["closest_player_start_end_changed"] * speed.max()
    )
    features["player_changed_x_path_length"] = (
        features["closest_player_start_end_changed"] * path_length
    )
    features["player_changed_and_close_edges"] = int(
        (features["closest_player_start_end_changed"] == 1)
        and (dist.iloc[START_FRAME] <= 2.0)
        and (dist.iloc[END_FRAME] <= 2.0)
    )
    features["player_changed_and_high_speed"] = int(
        (features["closest_player_start_end_changed"] == 1) and (speed.max() >= 5.0)
    )

    features["team_changed_x_speed_max"] = (
        features["closest_team_start_end_changed"] * speed.max()
    )
    features["team_changed_x_path_length"] = (
        features["closest_team_start_end_changed"] * path_length
    )

    features["high_speed_far_from_player_share"] = (
        (window["ball_speed"] >= 5.0) & (window["dist_ball_nearest_player"] > 3.0)
    ).mean()
    features["high_speed_close_to_player_share"] = (
        (window["ball_speed"] >= 5.0) & (window["dist_ball_nearest_player"] <= 2.0)
    ).mean()

    return features
