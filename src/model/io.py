"""Model persistence helpers for saving and loading trained packages.

This module provides a small wrapper around joblib to keep model artifacts,
thresholds, and feature names together for later evaluation or deployment.
"""

from pathlib import Path
from typing import Any, Optional

import joblib

###################################### SAVE & LOAD MODELS #####################################


def save_model_package(
    model: Any,
    threshold: float,
    model_name: str,
    feature_names: Optional[list[str]] = None,
    save_dir: str = "../saved_models",
) -> None:
    """Save a trained model package to a joblib file.

    Args:
        model: Trained model object to save.
        threshold: Decision threshold associated with the model.
        model_name: Name used for the saved file.
        feature_names: Optional list of feature names used by the model.
        save_dir: Directory where the saved model file will be written.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    package = {
        "model": model,
        "threshold": threshold,
        "model_name": model_name,
        "feature_names": feature_names,
    }

    file_path = Path(save_dir) / f"{model_name}.joblib"
    joblib.dump(package, file_path)

    print(f"Model saved to: {file_path}")


def load_model_package(
    model_name: str, save_dir: str = "../saved_models"
) -> dict[str, Any]:
    """Load a saved model package from disk.

    Args:
        model_name: Name of the saved model file (without extension).
        save_dir: Directory where the saved model file is stored.

    Returns:
        Loaded model package dictionary containing the model and metadata.
    """
    file_path = Path(save_dir) / f"{model_name}.joblib"
    package = joblib.load(file_path)

    print(f"Model loaded from: {file_path}")

    return package
