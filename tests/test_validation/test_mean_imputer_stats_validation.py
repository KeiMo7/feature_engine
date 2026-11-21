import numpy as np
import pandas as pd
import pytest

from feature_engine.imputation import MeanImputer
from feature_engine.validation import (
    MissingStatisticsError,
    StatisticsMismatchWarning,
    StatisticsRefitError,
    StatisticsValidationWarning,
)


@pytest.fixture()
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        {
            "a": rng.normal(10, 2, 1000),
            "b": rng.normal(-3, 1, 1000),
        }
    )
    X.loc[X.index[::10], "a"] = np.nan
    return X


@pytest.fixture()
def shifted(data):
    X2 = data.copy()
    X2["a"] = X2["a"] + 10
    return X2


# ---- normal fit: statistics and provenance are recorded -------------------


def test_fit_records_statistics_and_provenance(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    snap = imp.statistics_snapshot_
    assert snap.scope == "full"
    assert snap.n_samples == len(data)
    assert snap.n_chunks == 1
    assert snap.exact == {"mean": True}
    assert snap.error_bounds == {}
    assert snap.statistics["a"]["mean"] == pytest.approx(data["a"].mean())
    assert snap.statistics["b"]["mean"] == pytest.approx(data["b"].mean())
    # a distribution summary is stored for drift detection
    assert set(snap.distribution_summary) == {"a", "b"}
    assert len(snap.distribution_summary["a"]["hist_counts"]) == 10


def test_fit_records_sampled_scope_when_declared(data):
    imp = MeanImputer(
        imputation_method="median", validate_stats=True, stats_scope="sampled"
    ).fit(data.sample(200, random_state=1))
    assert imp.statistics_snapshot_.scope == "sampled"


def test_fit_and_transform_results_unchanged_when_validation_on(data):
    imp_off = MeanImputer(imputation_method="median").fit(data)
    imp_on = MeanImputer(imputation_method="median", validate_stats=True).fit(data)
    assert imp_off.imputer_dict_ == imp_on.imputer_dict_
    pd.testing.assert_frame_equal(imp_off.transform(data), imp_on.transform(data))


# ---- frozen statistics: accidental refit is blocked ------------------------


def test_refit_at_inference_is_rejected(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with pytest.raises(StatisticsRefitError, match="refit"):
        imp.fit(data)
    # the statistics were not changed by the blocked call
    assert imp.statistics_snapshot_.statistics["a"]["mean"] == pytest.approx(
        data["a"].mean()
    )


def test_partial_fit_after_fit_is_rejected(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with pytest.raises(StatisticsRefitError):
        imp.partial_fit(data)


def test_explicit_refit_is_allowed(data, shifted):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    imp.refit(shifted)
    assert imp.statistics_snapshot_.statistics["a"]["mean"] == pytest.approx(
        shifted["a"].mean()
    )
    # and the guard is armed again after the explicit refit
    with pytest.raises(StatisticsRefitError):
        imp.fit(data)


def test_double_fit_allowed_when_validation_off(data):
    imp = MeanImputer(imputation_method="mean").fit(data)
    imp.fit(data)  # silently re-fits, as before
    assert not hasattr(imp, "statistics_snapshot_")


# ---- statistics consistency check ------------------------------------------


def test_check_statistics_ok_on_same_data(data):
    import warnings

    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        report = imp.check_statistics(X=data)
    assert report["status"] == "ok"
    assert not [w for w in record if issubclass(w.category, Warning)]


def test_check_statistics_warns_on_mismatch(data, shifted):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with pytest.warns(StatisticsMismatchWarning, match="a.mean"):
        report = imp.check_statistics(X=shifted)
    assert report["status"] == "mismatch"
    entry = report["variables"]["a"]["statistics"]["mean"]
    assert entry["fit"] == pytest.approx(data["a"].mean())
    assert entry["other"] == pytest.approx(shifted["a"].mean())
    assert entry["within_tolerance"] is False
    assert report["variables"]["b"]["status"] == "ok"


def test_check_statistics_against_saved_snapshot_without_data(data, shifted, tmp_path):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    path = tmp_path / "stats.json"
    imp.save_statistics(path)

    # in the inference environment: only the saved summary is needed
    imp_prod = MeanImputer(imputation_method="mean", validate_stats=True).fit(shifted)
    with pytest.warns(StatisticsMismatchWarning):
        report = imp_prod.check_statistics(other=path)
    assert report["status"] == "mismatch"
    assert report["n_samples_fit"] == len(shifted)
    assert report["n_samples_other"] == len(data)


def test_check_statistics_against_other_transformer(data, shifted):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    other = MeanImputer(imputation_method="mean", validate_stats=True).fit(shifted)
    with pytest.warns(StatisticsMismatchWarning):
        report = imp.check_statistics(other=other)
    assert report["status"] == "mismatch"


def test_check_statistics_requires_exactly_one_source(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with pytest.raises(ValueError, match="exactly one"):
        imp.check_statistics()
    with pytest.raises(ValueError, match="exactly one"):
        imp.check_statistics(other=imp, X=data)


def test_check_statistics_failure_warns_but_does_not_raise(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True).fit(data)
    with pytest.warns(StatisticsValidationWarning, match="check failed"):
        report = imp.check_statistics(other=object())
    assert report["status"] == "error"


def test_check_statistics_without_snapshot_raises(data):
    imp = MeanImputer(imputation_method="mean").fit(data)
    with pytest.raises(MissingStatisticsError):
        imp.check_statistics(X=data)


# ---- missing statistics at inference: explicit error, no fallback ---------


def test_transform_raises_when_statistics_missing(data):
    imp = MeanImputer(imputation_method="mean").fit(data)
    imp.validate_stats = True  # validation switched on after fitting without it
    with pytest.raises(MissingStatisticsError, match="no recorded statistics"):
        imp.transform(data)


def test_transform_works_when_validation_off(data):
    imp = MeanImputer(imputation_method="mean").fit(data)
    assert imp.transform(data)["a"].isna().sum() == 0


# ---- chunked fitting: exactness is labelled --------------------------------


def test_partial_fit_mean_is_exact(data):
    chunks = [data.iloc[i : i + 250] for i in range(0, len(data), 250)]
    imp = MeanImputer(imputation_method="mean", random_state=42)
    for chunk in chunks:
        imp.partial_fit(chunk)
    assert imp.imputer_dict_["a"] == pytest.approx(data["a"].mean(), rel=1e-12)
    snap = imp.statistics_snapshot_
    assert snap.scope == "chunked"
    assert snap.n_chunks == 4
    assert snap.n_samples == len(data)
    assert snap.exact == {"mean": True}
    assert snap.error_bounds == {}


def test_partial_fit_median_is_labelled_approximate_with_error_bound(data):
    chunks = [data.iloc[i : i + 250] for i in range(0, len(data), 250)]
    imp = MeanImputer(imputation_method="median", random_state=42)
    for chunk in chunks:
        imp.partial_fit(chunk)
    snap = imp.statistics_snapshot_
    assert snap.scope == "chunked"
    # the median is approximate and the error bound is recorded
    assert snap.exact == {"median": False}
    assert snap.error_bound_confidence == 0.95
    bound = snap.error_bounds["a"]["median"]
    assert bound > 0
    # and the bound actually covers the true full-data median
    assert abs(imp.imputer_dict_["a"] - data["a"].median()) <= bound


def test_partial_fit_deterministic_for_fixed_seed(data):
    chunks = [data.iloc[i : i + 100] for i in range(0, len(data), 100)]
    estimates = []
    for _ in range(2):
        imp = MeanImputer(imputation_method="median", random_state=42)
        for chunk in chunks:
            imp.partial_fit(chunk)
        estimates.append(imp.imputer_dict_)
    assert estimates[0] == estimates[1]


def test_partial_fit_transform_consistent_with_fit(data):
    imp = MeanImputer(imputation_method="mean", random_state=42)
    for chunk in [data.iloc[:500], data.iloc[500:]]:
        imp.partial_fit(chunk)
    out = imp.transform(data)
    assert out["a"].isna().sum() == 0
    assert out.loc[data.index[0], "a"] == pytest.approx(data["a"].mean())


def test_partial_fit_after_partial_fit_does_not_raise(data):
    imp = MeanImputer(imputation_method="mean", validate_stats=True, random_state=42)
    imp.partial_fit(data.iloc[:500])
    imp.partial_fit(data.iloc[500:])  # accumulation is not a re-fit
    assert imp.statistics_snapshot_.n_chunks == 2


# ---- init parameter validation ----------------------------------------------


def test_validate_stats_must_be_bool():
    with pytest.raises(ValueError, match="validate_stats"):
        MeanImputer(validate_stats="yes")


def test_stats_scope_must_be_valid():
    with pytest.raises(ValueError, match="stats_scope"):
        MeanImputer(stats_scope="chunked")
