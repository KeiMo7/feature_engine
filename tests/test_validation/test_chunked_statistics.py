import numpy as np
import pandas as pd
import pytest

from feature_engine.validation import ChunkedStatisticsAccumulator


@pytest.fixture()
def data():
    rng = np.random.default_rng(7)
    return pd.DataFrame(
        {
            "a": rng.normal(10, 2, 5000),
            "b": rng.exponential(1.5, 5000),
        }
    )


def _fit_in_chunks(acc, X, n_chunks):
    for chunk in np.array_split(X, n_chunks):
        acc.update(pd.DataFrame(chunk, columns=X.columns))
    return acc


def test_mean_and_std_are_exact(data):
    acc = ChunkedStatisticsAccumulator(["a", "b"], random_state=1)
    _fit_in_chunks(acc, data, 7)
    assert acc.mean("a") == pytest.approx(data["a"].mean(), rel=1e-12)
    assert acc.mean("b") == pytest.approx(data["b"].mean(), rel=1e-12)
    assert acc.std("a") == pytest.approx(data["a"].std(ddof=0), rel=1e-9)
    assert acc.n_chunks == 7
    assert acc.n_rows == len(data)


def test_min_max_are_exact(data):
    acc = ChunkedStatisticsAccumulator(["a"], random_state=1)
    _fit_in_chunks(acc, data, 3)
    assert acc.min("a") == data["a"].min()
    assert acc.max("a") == data["a"].max()


def test_quantile_is_approximate_and_error_bound_recorded(data):
    acc = ChunkedStatisticsAccumulator(["a", "b"], random_state=1)
    _fit_in_chunks(acc, data, 10)
    values, bounds = acc.quantiles_with_bounds((0.5,))
    for v in ("a", "b"):
        estimate = values[v]["q0.5"]
        bound = bounds[v]["q0.5"]
        true = data[v].median()
        # the bound is recorded, positive, and covers the true value
        assert bound > 0
        assert abs(estimate - true) <= bound


def test_error_bound_shrinks_with_bigger_reservoir(data):
    bounds = []
    for size in (100, 2000):
        acc = ChunkedStatisticsAccumulator(["a"], reservoir_size=size, random_state=1)
        _fit_in_chunks(acc, data, 5)
        _, b = acc.quantiles_with_bounds((0.5,))
        bounds.append(b["a"]["q0.5"])
    assert bounds[1] < bounds[0]


def test_deterministic_for_fixed_seed(data):
    estimates = []
    for _ in range(2):
        acc = ChunkedStatisticsAccumulator(["a"], random_state=3)
        _fit_in_chunks(acc, data, 4)
        estimates.append(acc.quantile("a", 0.5))
    assert estimates[0] == estimates[1]


def test_chunk_boundaries_change_approximation_but_not_exact_stats(data):
    acc1 = ChunkedStatisticsAccumulator(["a"], random_state=3)
    _fit_in_chunks(acc1, data, 2)
    acc2 = ChunkedStatisticsAccumulator(["a"], random_state=3)
    _fit_in_chunks(acc2, data, 9)
    assert acc1.mean("a") == acc2.mean("a")
    # reservoir content depends on chunk boundaries, estimates may differ,
    # but both stay within their error bound of the true median
    for acc in (acc1, acc2):
        bound = acc.quantile_error_bound("a", 0.5)
        assert abs(acc.quantile("a", 0.5) - data["a"].median()) <= bound


def test_missing_values_are_skipped():
    X = pd.DataFrame({"a": [1.0, np.nan, 3.0, np.nan, 5.0]})
    acc = ChunkedStatisticsAccumulator(["a"], random_state=1)
    acc.update(X)
    assert acc.n_samples["a"] == 3
    assert acc.mean("a") == 3.0


def test_reservoir_size_validation():
    with pytest.raises(ValueError, match="reservoir_size"):
        ChunkedStatisticsAccumulator(["a"], reservoir_size=5)
