import numpy as np
import pytest

from feature_engine.validation import StatisticsSnapshot


def _snapshot():
    return StatisticsSnapshot(
        statistics={"a": {"mean": 1.5}, "b": {"mean": -2.0}},
        n_samples=100,
        scope="full",
        transformer="MeanImputer",
        exact={"mean": True},
    )


def test_scope_must_be_valid():
    with pytest.raises(ValueError, match="scope must be one of"):
        StatisticsSnapshot(statistics={}, n_samples=1, scope="weekly")


def test_snapshot_records_provenance():
    snap = StatisticsSnapshot(
        statistics={"a": {"median": 0.5}},
        n_samples=1000,
        scope="chunked",
        exact={"median": False},
        error_bounds={"a": {"median": 0.1}},
        error_bound_confidence=0.95,
        n_chunks=4,
        random_state=42,
    )
    assert snap.scope == "chunked"
    assert snap.is_exact("median") is False
    assert snap.error_bounds == {"a": {"median": 0.1}}
    assert snap.error_bound_confidence == 0.95
    assert snap.n_chunks == 4


def test_to_dict_from_dict_roundtrip():
    snap = _snapshot()
    snap2 = StatisticsSnapshot.from_dict(snap.to_dict())
    assert snap2.statistics == snap.statistics
    assert snap2.scope == snap.scope
    assert snap2.n_samples == snap.n_samples
    assert snap2.exact == snap.exact


def test_to_json_from_json_roundtrip(tmp_path):
    snap = _snapshot()
    path = tmp_path / "snapshot.json"
    snap.to_json(path)
    snap2 = StatisticsSnapshot.from_json(path)
    assert snap2.to_dict() == snap.to_dict()


def test_compare_identical_snapshots_is_ok():
    report = _snapshot().compare(_snapshot())
    assert report["status"] == "ok"
    assert report["variables"]["a"]["statistics"]["mean"]["within_tolerance"]


def test_compare_reports_both_values_on_mismatch():
    other = StatisticsSnapshot(
        statistics={"a": {"mean": 3.0}, "b": {"mean": -2.0}},
        n_samples=50,
        scope="full",
    )
    report = _snapshot().compare(other, rtol=1e-3, atol=1e-9)
    assert report["status"] == "mismatch"
    entry = report["variables"]["a"]["statistics"]["mean"]
    assert entry["fit"] == 1.5
    assert entry["other"] == 3.0
    assert entry["abs_diff"] == 1.5
    assert entry["within_tolerance"] is False
    # variable b matches
    assert report["variables"]["b"]["status"] == "ok"


def test_compare_flags_missing_statistic():
    other = StatisticsSnapshot(statistics={"a": {}}, n_samples=10, scope="full")
    report = _snapshot().compare(other)
    assert report["status"] == "mismatch"
    entry = report["variables"]["a"]["statistics"]["mean"]
    assert entry["other"] is None


def test_compare_respects_tolerance():
    snap = _snapshot()
    other = StatisticsSnapshot(
        statistics={"a": {"mean": 1.5 + 1e-4}, "b": {"mean": -2.0}},
        n_samples=10,
        scope="full",
    )
    assert snap.compare(other, rtol=1e-3)["status"] == "ok"
    assert snap.compare(other, rtol=1e-6, atol=1e-12)["status"] == "mismatch"
