import numpy as np
import pandas as pd

from heart_mlops.train import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_preprocessor,
)


def sample_features():
    """Create representative input data with all expected feature columns."""
    return pd.DataFrame(
        [
            {
                "age": 55,
                "sex": 1,
                "cp": 2,
                "trestbps": 130,
                "chol": 250,
                "fbs": 0,
                "restecg": 1,
                "thalach": 150,
                "exang": 0,
                "oldpeak": 1.0,
                "slope": 2,
                "ca": 0,
                "thal": 3,
            },
            {
                "age": 62,
                "sex": 0,
                "cp": 3,
                "trestbps": np.nan,
                "chol": 220,
                "fbs": 0,
                "restecg": 0,
                "thalach": 140,
                "exang": 1,
                "oldpeak": 2.0,
                "slope": 1,
                "ca": np.nan,
                "thal": 3,
            },
        ]
    )


def test_preprocessor_returns_finite_numeric_features():
    X = sample_features()
    preprocessor = build_preprocessor()
    transformed = preprocessor.fit_transform(X)

    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert values.shape[0] == len(X)
    assert values.shape[1] > len(NUMERIC_FEATURES)
    assert np.isfinite(values).all()


def test_preprocessor_handles_missing_values():
    X = sample_features()
    transformed = build_preprocessor().fit_transform(X)

    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert np.isfinite(values).all()


def test_preprocessor_handles_unseen_categories():
    X_train = sample_features()
    preprocessor = build_preprocessor()
    preprocessor.fit(X_train)

    X_new = sample_features().iloc[[0]].copy()
    X_new.loc[X_new.index[0], "cp"] = 99

    transformed = preprocessor.transform(X_new)
    values = transformed.toarray() if hasattr(transformed, "toarray") else transformed

    assert values.shape[0] == 1
    assert np.isfinite(values).all()


def test_feature_columns_are_partitioned():
    assert set(NUMERIC_FEATURES).isdisjoint(CATEGORICAL_FEATURES)
    assert len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES) == 13
