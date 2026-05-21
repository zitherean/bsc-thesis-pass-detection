"""Project constants used throughout data loading, preprocessing, and modeling.

This module exposes settings for file paths, required data schema, pitch bounds,
interpolation/windowing parameters, and fixed seeds for deterministic behavior.
"""

# absolute path of match folder
BASE_PATH = "../data_matches"

# non numeric column that cannot be parsed with to_numeric
# In the future think about getting this directly from the file.
NON_NUMERIC_COLUMN = "ballId"

# timestamp column used in preprocessing
TIMESTAMP_COLUMN = "timestamp"

# minimum required columns for the dataframe to be accepted as tracking data
REQUIRED_COLUMNS = ["timestamp", "ballId", "posX", "posY", "playerId"]

# absolute position of the traccking data file
# only used for testing
MATCH_COORDINATES_ABSOLUTE_POSITION = 1

# assume GenGee pitch length and pitch width for all fields
PITCH_LENGTH_MAX = 105
PITCH_WIDTH_MAX = 68
PITCH_LENGTH_MIN = 0
PITCH_WIDTH_MIN = 0

# tolerance for pitch dimensions
TOLERANCE = 5

# smoothing value for mathematical calculations
SMOOTHING = 0.000001

# millisecond value for seconds calculations
MILLISECONDS_IN_SECOND = 1000

# maximum gap for interpolation
MAX_GAP = 5

# random state to replicate tests
RANDOM_STATE = 42

# how many frames in a window
WINDOW_SIZE = 15

# window stride length
STRIDE_LENGTH = 15

# center frame of the window
CENTER_FRAME = WINDOW_SIZE // 2

# start frame of the window
START_FRAME = 0

# end frame of the window
END_FRAME = WINDOW_SIZE - 1

# quality of tracking data (5hz)
# 5 frames in 1 second
FRAMES_PER_SECOND = 5
