"""End-to-end pipeline for loading, preprocessing, and feature extraction.

This module loads match tracking and event data, cleans and interpolates
coordinates, extracts motion and interaction features, and builds windowed
training datasets.
"""

import os
from typing import Any

from src.constants import (
    BASE_PATH,
    WINDOW_SIZE,
    STRIDE_LENGTH,
    MAX_GAP,
)

from src.data_processing.dataframe import build_player_team_map

from src.data_processing.dataframe import load_correct_tracking_sheet, csv_to_dataframe

from src.data_processing.preprocessing import (
    convert_to_numeric,
    sort_timestamps,
    remove_duplicate_timestamps,
    clean_out_of_bounds,
    interpolate_all_coordinates,
)
from src.features.events import label_passes
from src.features.features import (
    add_ball_displacement,
    add_ball_velocity,
    add_ball_acceleration,
    add_ball_speed_prev,
    add_ball_direction_change,
    add_distance_to_nearest_player,
    add_ball_player_proximity_flags,
    add_own_and_opposition_players_near_ball,
    add_distance_to_player_change,
    add_closest_player_streak,
)
from src.pipeline.dataset import build_window_meta_dataset


def extract_features(
    tracking_df_clean: Any,
    player_team_map: Any,
) -> Any:
    """Compute all derived tracking features for a match.

    Args:
        tracking_df_clean: Preprocessed tracking DataFrame.
        player_team_map: Mapping from player IDs to team labels.

    Returns:
        DataFrame containing the extracted feature columns.
    """
    tracking_df_features = add_ball_displacement(tracking_df_clean)
    tracking_df_features = add_ball_velocity(tracking_df_features)
    tracking_df_features = add_ball_acceleration(tracking_df_features)
    tracking_df_features = add_ball_speed_prev(tracking_df_features)

    tracking_df_features = add_ball_direction_change(tracking_df_features)

    tracking_df_features = add_distance_to_nearest_player(
        tracking_df_features,
        player_team_map,
    )
    tracking_df_features = add_ball_player_proximity_flags(tracking_df_features)
    tracking_df_features = add_own_and_opposition_players_near_ball(
        tracking_df_features,
        player_team_map,
    )
    tracking_df_features = add_distance_to_player_change(tracking_df_features)
    tracking_df_features = add_closest_player_streak(tracking_df_features)

    return tracking_df_features


def preprocess_matches(tracking_df: Any, pass_events_df: Any) -> Any:
    """Clean and prepare match tracking and pass event data.

    Args:
        tracking_df: Raw tracking DataFrame loaded from the match workbook.
        pass_events_df: Raw pass event DataFrame.

    Returns:
        Cleaned tracking DataFrame with labeled passes and interpolated coordinates.
    """
    tracking_df_clean = convert_to_numeric(tracking_df)
    tracking_df_clean = remove_duplicate_timestamps(tracking_df_clean)
    tracking_df_clean = sort_timestamps(tracking_df_clean)

    pass_events_df_clean = remove_duplicate_timestamps(pass_events_df)
    pass_events_df_clean = sort_timestamps(pass_events_df_clean)

    tracking_df_clean = label_passes(
        tracking_df_clean, pass_events_df_clean
    )  # label passes here before further processing

    tracking_df_clean = clean_out_of_bounds(tracking_df_clean)
    tracking_df_clean = interpolate_all_coordinates(tracking_df_clean, max_gap=MAX_GAP)

    return tracking_df_clean


def pipeline() -> list[dict[str, Any]]:
    """Load and process all match folders into windowed training datasets.

    This function scans ``BASE_PATH`` for match folders, loads tracking,
    pass event, and overall stats data for each match, preprocesses the data,
    extracts features, and builds slide-window datasets with metadata.

    Returns:
        A list of dictionaries, one per match, containing keys:
        ``match``, ``X``, ``y``, ``meta``, and ``real_passes``.
    """
    all_matches = []

    # Sort match folders from low to high for reproducibility
    match_folders = sorted(
        [x for x in os.listdir(BASE_PATH) if x.startswith("match_")],
        key=lambda x: int(x.split("_")[1]),
    )
    for match_folder in match_folders:
        match_path = os.path.join(BASE_PATH, match_folder)

        if not os.path.isdir(match_path):
            continue

        excel_file = None
        passes_csv_file = None
        overall_stats_csv_file = None

        for file in os.listdir(match_path):
            if file.endswith(".xlsx"):
                excel_file = os.path.join(match_path, file)
            elif file.endswith(".csv") and "passes" in file.lower():
                passes_csv_file = os.path.join(match_path, file)
            elif file.endswith(".csv") and "overall" in file.lower():
                overall_stats_csv_file = os.path.join(match_path, file)

        if (
            excel_file is None
            or passes_csv_file is None
            or overall_stats_csv_file is None
        ):
            print(f"Skipping {match_folder}, missing files.")
            continue

        # think about doing some testing?

        ### LOAD DATA ###
        tracking_df = load_correct_tracking_sheet(excel_file)
        pass_events_df = csv_to_dataframe(passes_csv_file)
        overall_stats = csv_to_dataframe(overall_stats_csv_file)
        print(f"Loaded: {match_folder}")

        ### BUILD PLAYER TEAM MAP ###
        player_team_map = build_player_team_map(overall_stats)
        print("Player Team Map build complete")

        ### PRE-PROCESS DATA ###
        tracking_df_clean = preprocess_matches(tracking_df, pass_events_df)
        print("Data pre-processing complete")

        ### FEATURE EXTRACTION ###
        tracking_df_features = extract_features(tracking_df_clean, player_team_map)
        tracking_df_features = tracking_df_features.reset_index(drop=True)
        print("Feature extraction complete")

        ### ACTUAL PASSES ###
        real_passes = tracking_df_features.loc[
            tracking_df_features["is_pass"] == 1, ["timestamp", "posX", "posY"]
        ].copy()
        real_passes["frame_idx"] = real_passes.index
        real_passes = real_passes.reset_index(drop=True)
        print(f"Actual passes: {len(real_passes)}")

        ### BUILD DATASET ###
        X, y, meta = build_window_meta_dataset(
            tracking_df_features, WINDOW_SIZE, STRIDE_LENGTH, label_mode="any"
        )
        print("Dataset build complete")

        all_matches.append(
            {
                "match": match_folder,
                "X": X,
                "y": y,
                "meta": meta,
                "real_passes": real_passes,
            }
        )

    print("Processing completed")
    return all_matches
