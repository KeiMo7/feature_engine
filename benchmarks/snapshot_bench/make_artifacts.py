"""Generate the sample snapshot and rollback record shipped in artifacts/.

Run from the repo root:
    PYTHONPATH=$PWD python -P benchmarks/snapshot_bench/make_artifacts.py
"""
import json
from pathlib import Path

import pandas as pd

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import RenameFeatures
from feature_engine.selection import DropFeatures
from feature_engine.snapshot import diff_snapshots, rollback

ARTIFACTS = Path(__file__).resolve().parents[2] / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

X = pd.DataFrame(
    {
        "x1": [1.0, 2.0, 3.0, 4.0],
        "x2": [5.0, 6.0, 7.0, 8.0],
        "x3": [9.0, 10.0, 11.0, 12.0],
    }
)

# ---- version 1 of the chain ------------------------------------------------
pipe_v1 = Pipeline(
    [
        ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
        ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"})),
        ("drop", DropFeatures(features_to_drop=["x2"])),
    ]
).fit(X)

snapshot_v1 = pipe_v1.create_feature_snapshot(X, name="feature-set-v1")
(ARTIFACTS / "snapshot_v1.json").write_text(
    json.dumps(snapshot_v1.to_dict(), indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# ---- version 2: rename the sum feature, add a max feature -----------------
pipe_v2 = Pipeline(
    [
        ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean", "max"])),
        (
            "rename",
            RenameFeatures(
                rename_dict={"x3": "x3_renamed", "sum_x1_x2": "total_x1_x2"}
            ),
        ),
        ("drop", DropFeatures(features_to_drop=["x2"])),
    ]
).fit(X)

snapshot_v2 = pipe_v2.create_feature_snapshot(X, name="feature-set-v2")
(ARTIFACTS / "snapshot_v2.json").write_text(
    json.dumps(snapshot_v2.to_dict(), indent=2, ensure_ascii=False),
    encoding="utf-8",
)

diff = diff_snapshots(snapshot_v1, snapshot_v2)
(ARTIFACTS / "diff_v1_v2.json").write_text(
    json.dumps(diff.to_dict(), indent=2, ensure_ascii=False),
    encoding="utf-8",
)

# ---- rollback to v1 and verify ---------------------------------------------
restored = rollback(snapshot_v1, X, verify=True)
rollback_output = restored.transform(X)
original_output = pipe_v1.transform(X)

record = {
    "rolled_back_to": "feature-set-v1",
    "restored_steps": [name for name, _ in restored.steps],
    "expected_feature_names": snapshot_v1.feature_names,
    "observed_feature_names": rollback_output.columns.tolist(),
    "feature_sets_match": rollback_output.columns.tolist()
    == snapshot_v1.feature_names,
    "values_match_original_chain": rollback_output.equals(original_output),
    "output_preview": rollback_output.to_dict(orient="list"),
}
(ARTIFACTS / "rollback_record.json").write_text(
    json.dumps(record, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("artifacts written to", ARTIFACTS)
print(json.dumps(record, indent=2))
