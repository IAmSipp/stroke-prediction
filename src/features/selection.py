"""Feature selection at the human-readable column level."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils.validation import check_is_fitted


class LogicalFeatureSelector(TransformerMixin, BaseEstimator):
    """Select whole DataFrame columns using mutual information with the target.

    Text columns are ordinal-encoded temporarily for scoring only. ``transform``
    returns the selected original columns, so the normal preprocessing pipeline
    can impute and one-hot encode them afterwards.
    """

    def __init__(
        self,
        k: int | str = "all",
        discrete_unique_threshold: int = 10,
        random_state: int = 42,
    ):
        self.k = k
        self.discrete_unique_threshold = discrete_unique_threshold
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        """Score and select logical columns using training data only."""
        self._validate_input(X)
        encoded, discrete_mask = self._encode_for_scoring(X)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.scores_ = mutual_info_classif(
            encoded,
            y,
            discrete_features=discrete_mask,
            random_state=self.random_state,
        )

        feature_count = len(self.feature_names_in_)
        if self.k == "all":
            selected_count = feature_count
        elif isinstance(self.k, int) and 1 <= self.k <= feature_count:
            selected_count = self.k
        else:
            raise ValueError(
                f"k must be 'all' or an integer from 1 to {feature_count}; "
                f"received {self.k!r}."
            )

        ranked_indices = np.argsort(-self.scores_, kind="stable")
        selected_indices = ranked_indices[:selected_count]
        self.support_ = np.zeros(feature_count, dtype=bool)
        self.support_[selected_indices] = True
        self.selected_features_ = self.feature_names_in_[self.support_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return only the logical columns selected during ``fit``."""
        check_is_fitted(self, "selected_features_")
        self._validate_input(X)
        missing = set(self.selected_features_).difference(X.columns)
        if missing:
            raise ValueError(f"Input is missing selected columns: {sorted(missing)}")
        return X.loc[:, self.selected_features_].copy()

    def get_support(self) -> np.ndarray:
        """Return a mask aligned with ``feature_names_in_``."""
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the selected logical column names."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_.copy()

    def _encode_for_scoring(
        self,
        X: pd.DataFrame,
    ) -> tuple[pd.DataFrame, np.ndarray]:
        encoded = pd.DataFrame(index=X.index)
        discrete_mask = []

        for column in X.columns:
            series = X[column]
            is_numeric = pd.api.types.is_numeric_dtype(series)
            is_discrete = (not is_numeric) or (
                series.nunique(dropna=True) <= self.discrete_unique_threshold
            )

            if is_numeric:
                numeric = pd.to_numeric(series, errors="coerce")
                fill_value = numeric.median()
                encoded[column] = numeric.fillna(0 if pd.isna(fill_value) else fill_value)
            else:
                text = series.astype("string").fillna("__missing__")
                encoded[column] = pd.Categorical(text).codes

            discrete_mask.append(is_discrete)

        return encoded, np.asarray(discrete_mask, dtype=bool)

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("LogicalFeatureSelector requires a pandas DataFrame.")
        if X.empty:
            raise ValueError("LogicalFeatureSelector received an empty DataFrame.")


class FixedFeatureSelector(TransformerMixin, BaseEstimator):
    """Select a fixed list of logical features chosen in an earlier phase."""

    def __init__(self, features):
        self.features = features

    def fit(self, X: pd.DataFrame, y=None):
        """Validate that every frozen feature can be produced from the input."""
        self._validate_input(X)
        missing = set(self.features).difference(X.columns)
        if missing:
            raise ValueError(f"Input is missing frozen features: {sorted(missing)}")
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.selected_features_ = np.asarray(self.features, dtype=object)
        self.support_ = np.isin(self.feature_names_in_, self.selected_features_)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return the frozen logical features in their saved order."""
        check_is_fitted(self, "selected_features_")
        self._validate_input(X)
        missing = set(self.selected_features_).difference(X.columns)
        if missing:
            raise ValueError(f"Input is missing frozen features: {sorted(missing)}")
        return X.loc[:, self.selected_features_].copy()

    def get_support(self) -> np.ndarray:
        """Return a mask aligned with the columns seen during ``fit``."""
        check_is_fitted(self, "support_")
        return self.support_.copy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        """Return the frozen logical feature names."""
        check_is_fitted(self, "selected_features_")
        return self.selected_features_.copy()

    @staticmethod
    def _validate_input(X: pd.DataFrame) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FixedFeatureSelector requires a pandas DataFrame.")
        if X.empty:
            raise ValueError("FixedFeatureSelector received an empty DataFrame.")
