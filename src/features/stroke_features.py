import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


ENGINEERED_FEATURE_DEPENDENCIES = {
    "age_squared": ("age",),
    "avg_glucose_log1p": ("avg_glucose_level",),
    "bmi_squared": ("bmi",),
    "age_glucose_interaction": ("age", "avg_glucose_level"),
    "bmi_glucose_interaction": ("bmi", "avg_glucose_level"),
    "comorbidity_count": ("hypertension", "heart_disease"),
}


def get_required_raw_features(
    selected_features,
    original_feature_order,
) -> list[str]:
    required = set()
    for feature in selected_features:
        dependencies = ENGINEERED_FEATURE_DEPENDENCIES.get(feature)
        if dependencies is None:
            required.add(feature)
        else:
            required.update(dependencies)

    return [feature for feature in original_feature_order if feature in required]


class StrokeFeatureEngineer(TransformerMixin, BaseEstimator):

    def fit(self, X: pd.DataFrame, y=None):
        self._validate_input(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(X)
        frame = X.copy()

        if "age" in frame:
            frame["age_squared"] = frame["age"].pow(2)

        if "avg_glucose_level" in frame:
            frame["avg_glucose_log1p"] = np.log1p(
                frame["avg_glucose_level"].clip(lower=0)
            )

        if "bmi" in frame:
            frame["bmi_squared"] = frame["bmi"].pow(2)

        if {"age", "avg_glucose_level"}.issubset(frame.columns):
            frame["age_glucose_interaction"] = (
                frame["age"] * frame["avg_glucose_level"]
            )

        if {"bmi", "avg_glucose_level"}.issubset(frame.columns):
            frame["bmi_glucose_interaction"] = (
                frame["bmi"] * frame["avg_glucose_level"]
            )

        risk_columns = [
            column
            for column in ("hypertension", "heart_disease")
            if column in frame.columns
        ]
        if risk_columns:
            frame["comorbidity_count"] = frame[risk_columns].sum(
                axis=1,
                min_count=len(risk_columns),
            )

        return frame

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("StrokeFeatureEngineer requires a pandas DataFrame.")
