import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import MeanImputer
from feature_engine.validation import (
    DistributionDriftWarning,
    InsufficientSampleWarning,
    MissingStatisticsError,
    StatisticsValidationWarning,
)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(10, 2, 2000),
            "b": rng.normal(-3, 1, 2000),
        }
    )
    X.loc[X.index[::10], "a"] = np.nan
    return X


@pytest.fixture()
def fitted(data):
    return MeanImputer(imputation_method="mean", validate_stats=True).fit(data)


def test_no_drift_on_same_distribution(fitted):
    rng = np.random.default_rng(1)
    X_new = pd.DataFrame({"a": rng.normal(10, 2, 2000), "b": rng.normal(-3, 1, 2000)})
    report = fitted.check_drift(X_new)
    assert report["status"] == "ok"
    assert all(not v["drift"] for v in report["variables"].values())


def test_drift_warning_on_shifted_distribution(fitted):
    rng = np.random.default_rng(1)
    X_new = pd.DataFrame({"a": rng.normal(30, 2, 2000), "b": rng.normal(-3, 1, 2000)})
    with pytest.warns(DistributionDriftWarning, match="drifted"):
        report = fitted.check_drift(X_new)
    assert report["status"] == "drift"
    assert report["variables"]["a"]["drift"] is True
    assert report["variables"]["a"]["psi"] > report["threshold"]
    assert report["variables"]["b"]["drift"] is False


def test_drift_check_on_transformed_data(fitted, data):
    X_new = data.copy()
    X_new["a"] = X_new["a"] + 20
    with pytest.warns(DistributionDriftWarning):
        report = fitted.check_drift(X_new, transformed=True)
    assert report["transformed"] is True
    assert report["status"] == "drift"


def test_insufficient_sample_gives_no_verdict(fitted, data):
    with pytest.warns(InsufficientSampleWarning, match="too small"):
        report = fitted.check_drift(data.iloc[:50])
    assert report["status"] == "insufficient_samples"
    assert report["n_samples"] == 50
    assert "variables" not in report


def test_drift_check_failure_warns_but_does_not_raise(fitted):
    # a dataframe without the fitted variables breaks the check internally
    bad = pd.DataFrame({"unrelated": np.zeros(500)})
    with pytest.warns(StatisticsValidationWarning, match="drift check failed"):
        report = fitted.check_drift(bad)
    assert report["status"] == "error"


def test_drift_check_without_snapshot_raises(data):
    imp = MeanImputer(imputation_method="mean").fit(data)
    with pytest.raises(MissingStatisticsError):
        imp.check_drift(data)


def test_drift_check_after_chunked_fit(data):
    chunks = [data.iloc[i : i + 500] for i in range(0, len(data), 500)]
    imp = MeanImputer(imputation_method="median", random_state=42)
    for chunk in chunks:
        imp.partial_fit(chunk)
    rng = np.random.default_rng(2)
    X_new = pd.DataFrame({"a": rng.normal(10, 2, 2000), "b": rng.normal(-3, 1, 2000)})
    report = imp.check_drift(X_new)
    assert report["status"] == "ok"
