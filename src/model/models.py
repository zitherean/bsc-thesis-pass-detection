"""Model construction, search, and evaluation helpers.

This module defines model pipelines, hyperparameter search routines, threshold
selection utilities, and evaluation helpers for classification experiments.
"""

from imblearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GroupKFold, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report, f1_score, ConfusionMatrixDisplay

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.constants import RANDOM_STATE

########################## MODELS #################################


def build_model(
    model_name: str,
    scale: float,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict]:
    """Create a machine learning pipeline and hyperparameter distribution.

    The available models are Logistic Regression, SVM, Decision Tree, Random
    Forest, XGBoost, and MLP. Each pipeline includes imputation and any required
    preprocessing steps.

    Args:
        model_name: Name of the model architecture to build.
        scale: Rescaling factor used for class weighting on the positive class.
        random_state: Random seed used for deterministic model behavior.

    Returns:
        A tuple containing the pipeline and a parameter distribution dictionary
        for randomized hyperparameter search.
    """
    if model_name == "Logistic Regression":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=10000, random_state=random_state),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__C": [
                0.000001,
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
                10000,
                100000,
            ],
            "model__class_weight": [
                "balanced",
                {0: 1, 1: scale},
                {0: 1, 1: 0.75},
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
        }
    elif model_name == "SVM":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="linear",
                        probability=True,
                        random_state=random_state,
                    ),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__C": [
                0.000001,
                0.00001,
                0.0001,
                0.001,
                0.01,
                0.1,
                1,
                10,
                100,
                1000,
                10000,
                100000,
            ],
            "model__class_weight": [
                "balanced",
                {0: 1, 1: scale},
                {0: 1, 1: 0.75},
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
        }

    elif model_name == "Decision Tree":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    DecisionTreeClassifier(random_state=random_state),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__criterion": [
                "gini",
                "entropy",
            ],
            "model__max_depth": [
                3,
                5,
                7,
                None,
            ],
            "model__min_samples_split": [
                2,
                10,
                20,
            ],
            "model__min_samples_leaf": [
                1,
                5,
                10,
            ],
            "model__class_weight": [
                "balanced",
                {0: 1, 1: scale},
                {0: 1, 1: 0.75},
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
        }
    elif model_name == "Random Forest":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    RandomForestClassifier(random_state=random_state),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__n_estimators": [100, 300],
            "model__criterion": ["gini", "entropy"],
            "model__max_depth": [5, 7, 10],
            "model__min_samples_split": [
                2,
                10,
            ],
            "model__min_samples_leaf": [1, 5],
            "model__class_weight": [
                "balanced",
                {0: 1, 1: scale},
                {0: 1, 1: 0.75},
                {0: 1, 1: 2},
                {0: 1, 1: 3},
            ],
        }
    elif model_name == "XGBoost":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                (
                    "model",
                    XGBClassifier(
                        random_state=random_state,
                        objective="binary:logistic",
                        eval_metric="logloss",
                        n_jobs=1,
                    ),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__n_estimators": [100, 300],
            "model__max_depth": [5, 7, 10],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
            "model__min_child_weight": [1, 5],
            "model__gamma": [0, 0.3],
            "model__reg_alpha": [0, 0.1],
            "model__reg_lambda": [1, 5],
            "model__scale_pos_weight": [0.75, 1, 2, 3, scale],
        }
    elif model_name == "MLP":
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer()),
                ("scaler", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        max_iter=1000, random_state=random_state, early_stopping=True
                    ),
                ),
            ]
        )

        param_dist = {
            "imputer": [
                SimpleImputer(strategy="median"),
                SimpleImputer(strategy="mean"),
            ],
            "model__hidden_layer_sizes": [
                (32,),
                (64,),
                (64, 32),
            ],
            "model__activation": [
                "relu",
                "tanh",
            ],
            "model__alpha": [
                0.0001,
                0.001,
                0.01,
            ],
            "model__learning_rate_init": [
                0.0001,
                0.001,
                0.01,
            ],
            "model__batch_size": [
                32,
                64,
            ],
        }

    else:
        pipe = None
        param_dist = None
        print(f"Unkown model: {model_name}")

    return pipe, param_dist


########################## TRAIN MODEL + HYPERPARAMETER SEARCH #################################


def run_random_search(
    pipe: Pipeline,
    param_dist: dict,
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series | None = None,
    n_splits: int = 5,
    scoring: str = "f1",
    n_iter: int = 50,
) -> RandomizedSearchCV:
    """Run randomized hyperparameter search.

    Uses grouped cross-validation when groups are provided.
    Otherwise, uses regular K-fold cross-validation.

    Args:
        pipe: Pipeline containing preprocessing and estimator steps.
        param_dist: Hyperparameter distributions for search.
        model_name: Human-readable model name for logging.
        X_train: Training features.
        y_train: Training labels.
        groups: Group labels to preserve in cross-validation splits.
        n_splits: Number of groups for cross-validation.
        scoring: Scoring metric used to rank models.
        n_iter: Number of randomized parameter settings to try.

    Returns:
        The fitted ``RandomizedSearchCV`` search object.
    """

    if groups is None:
        cv = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        fit_kwargs = {}
        print(f"\nRunning {model_name} with regular KFold CV\n")

    else:
        cv = GroupKFold(n_splits=n_splits)
        fit_kwargs = {"groups": groups}
        print(f"\nRunning {model_name} with GroupKFold CV\n")

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=5,
        verbose=10,
        random_state=RANDOM_STATE,
        refit=True,
    )

    search.fit(X_train, y_train, **fit_kwargs)

    print(f"\nBest params for {model_name}: {search.best_params_}\n")
    print(f"Best CV score: {search.best_score_:.4f}\n")

    return search


def find_best_threshold(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> float:
    """Search for the best probability threshold based on F1 score.

    Args:
        model: Trained classifier with ``predict_proba``.
        X_val: Validation features.
        y_val: Validation labels.

    Returns:
        Best threshold value found between 0.1 and 0.9.
    """
    y_probs = model.predict_proba(X_val)[:, 1]

    thresholds = np.linspace(0.1, 0.9, 17)
    best_thresh = 0.5
    best_score = 0

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        score = f1_score(y_val, y_pred)

        if score > best_score:
            best_score = score
            best_thresh = t

    print(f"Best threshold: {best_thresh:.3f}")
    print(f"Best F1: {best_score:.4f}")

    return best_thresh


########################## EVALUATION #################################


def evaluate_with_threshold(
    model: object,
    model_name: str,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5,
) -> None:
    """Evaluate a model on validation data using a fixed probability threshold.

    Args:
        model: Trained classifier with ``predict_proba``.
        model_name: Human-readable name for output labels.
        X_val: Validation features.
        y_val: Validation labels.
        threshold: Probability threshold for positive classification.
    """
    y_probs = model.predict_proba(X_val)[:, 1]
    y_pred = (y_probs >= threshold).astype(int)

    # Classification report
    print(f"{model_name} - Validation results:")
    print(classification_report(y_val, y_pred))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(
        y_val, y_pred, values_format=""
    )  # use values_format to format the CM display
    plt.title(f"{model_name} - Confusion Matrix (Validation)")
    plt.show()


def final_evaluation(
    model: object,
    model_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_threshold: float,
) -> dict:
    """Evaluate a final model on held-out test data.

    Args:
        model: Trained classifier with ``predict_proba``.
        model_name: Human-readable name for output labels.
        X_test: Test features.
        y_test: Test labels.
        best_threshold: Probability threshold for positive classification.

    Returns:
        Dictionary of classification metrics computed on the test set.
    """
    y_probs = model.predict_proba(X_test)[:, 1]
    y_pred = (y_probs >= best_threshold).astype(int)

    # Classification report
    print(f"{model_name} - Final results:")
    report = classification_report(y_test, y_pred, output_dict=True)

    print(classification_report(y_test, y_pred))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title(f"{model_name} - Confusion Matrix (Final)")
    plt.show()

    return {
        "accuracy": report["accuracy"],
        "precision_0": report["0"]["precision"],
        "recall_0": report["0"]["recall"],
        "f1_0": report["0"]["f1-score"],
        "precision_1": report["1"]["precision"],
        "recall_1": report["1"]["recall"],
        "f1_1": report["1"]["f1-score"],
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "weighted_precision": report["weighted avg"]["precision"],
        "weighted_recall": report["weighted avg"]["recall"],
        "weighted_f1": report["weighted avg"]["f1-score"],
    }


def plot_f1_vs_threshold(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> None:
    """Plot F1 score as a function of classification threshold.

    Args:
        model: Trained classifier with ``predict_proba``.
        X_val: Validation features.
        y_val: Validation labels.
    """
    y_probs = model.predict_proba(X_val)[:, 1]
    thresholds = np.linspace(0.0, 1.0, 50)

    f1_scores = []

    for t in thresholds:
        y_pred = (y_probs >= t).astype(int)
        f1_scores.append(f1_score(y_val, y_pred))

    plt.plot(thresholds, f1_scores)
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title("F1 vs Threshold")
    plt.show()


########################## FEATURE IMPORTANCE #################################


def evaluate_feature_importance(
    model: object,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    feature_names: list[str],
) -> pd.Series:
    """Compute and return permutation-based feature importance.

    Args:
        model: Trained classifier.
        X_val: Validation features.
        y_val: Validation labels.
        feature_names: Names of the input features.

    Returns:
        Series of feature importances sorted in descending order.
    """
    r = permutation_importance(
        model,
        X_val,
        y_val,
        n_repeats=30,
        random_state=RANDOM_STATE,
        n_jobs=6,
    )

    feat_imp = pd.Series(r.importances_mean, index=feature_names)
    feat_imp = feat_imp.sort_values(ascending=False)

    print(feat_imp.head(len(feature_names)))

    return feat_imp
