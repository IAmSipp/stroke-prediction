"""Leakage-safe preprocessing and model pipeline builders."""

from collections.abc import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def split_features_target(
        frame: pd.DataFrame, target: str = "stroke", drop_columns: Iterable[str] = ("id",),) -> tuple[pd.DataFrame, pd.Series]:
    """Return features and target while excluding identifier columns."""
    if target not in frame.columns:
        raise KeyError(f"Target column '{target}' was not found.")

    excluded = [target, *(column for column in drop_columns if column in frame.columns)]
    features = frame.drop(columns=excluded).copy()
    labels = frame[target].copy()
    return features, labels


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing that learns all statistics during ``fit`` only."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, make_column_selector(dtype_include="number")),
            ("categorical", categorical_pipeline, make_column_selector(dtype_exclude="number")),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_model_pipeline(
    estimator: BaseEstimator,
    feature_engineer: TransformerMixin | None = None,
    feature_selector: TransformerMixin | None = None,
) -> Pipeline:
    """Build feature engineering, logical selection, preprocessing, and model."""
    steps = []
    if feature_engineer is not None:
        steps.append(("feature_engineering", feature_engineer))
    if feature_selector is not None:
        steps.append(("feature_selection", feature_selector))
    steps.append(("preprocessor", build_preprocessor()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)
