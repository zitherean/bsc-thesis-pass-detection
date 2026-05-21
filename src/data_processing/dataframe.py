"""Utilities to load tracking and statistics data into pandas DataFrames.

This module provides helpers to read Excel sheets and CSV files and to identify
the correct tracking statistics sheet based on required columns.
"""

from typing import Any, Optional
import pandas as pd

from src.constants import REQUIRED_COLUMNS


def excel_sheet_to_dataframe(file_path: str, position: str | int) -> pd.DataFrame:
    """Read a specific Excel sheet into a pandas DataFrame.

    Args:
        file_path: Path to the Excel workbook.
        position: Sheet name or index to load from the workbook.

    Returns:
        A pandas DataFrame containing the sheet's data.
    """
    df = pd.read_excel(file_path, sheet_name=position)
    return df


def csv_to_dataframe(file_path: str) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Args:
        file_path: Path to the CSV file.

    Returns:
        A pandas DataFrame with the CSV contents.
    """
    df = pd.read_csv(file_path)
    return df


def is_valid_tracking_df(df: pd.DataFrame) -> bool:
    """Check whether ``df`` contains all required tracking columns.

    Args:
        df: DataFrame to validate.

    Returns:
        True if all ``REQUIRED_COLUMNS`` are present in ``df.columns``, False otherwise.
    """
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            return False
    return True


def load_correct_tracking_sheet(file_path: str) -> Optional[pd.DataFrame]:
    """Find and load the workbook sheet that contains tracking statistics.

    The function iterates over all sheets in the workbook and returns the first
    sheet that contains the required tracking columns. If a valid sheet is found,
    it asserts that the sheet name contains the word 'statistics' (case-insensitive)
    as an additional sanity check.

    Args:
        file_path: Path to the Excel workbook.

    Returns:
        The first valid tracking DataFrame, or ``None`` if no valid sheet is found.
    """
    xls = pd.ExcelFile(file_path)

    for sheet_name in xls.sheet_names:
        # Read each sheet and validate.
        df = pd.read_excel(file_path, sheet_name=sheet_name)
        if is_valid_tracking_df(df):
            # Sanity check: sheet name should contain statistics.
            assert "statistics" in sheet_name.lower()
            print(f"Using sheet: {sheet_name}")
            return df

    print(f"No valid tracking sheet found in {file_path}")
    return None


def build_player_team_map(overall_stats: pd.DataFrame) -> dict[Any, str]:
    """Build a mapping from player ID to team name.

    Args:
        overall_stats: DataFrame containing ``playerId`` and ``Match Team``.

    Returns:
        Dictionary mapping each player ID to its team name.
    """
    required_cols = {"Match Team", "playerId"}
    if not required_cols.issubset(overall_stats.columns):
        print(f"Missing required columns: {required_cols}")

    teams = overall_stats["Match Team"].dropna().unique()
    if len(teams) != 2:
        print(f"Expected 2 teams, found {len(teams)}: {teams}")

    player_team_map = (
        overall_stats[["playerId", "Match Team"]]
        .dropna()
        .drop_duplicates()
        .set_index("playerId")["Match Team"]
        .to_dict()
    )

    return player_team_map
