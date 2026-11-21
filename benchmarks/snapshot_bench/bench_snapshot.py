"""Time/memory comparison: fit+transform with and without a feature snapshot."""
import time
import tracemalloc

import numpy as np
import pandas as pd

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.preprocessing import RenameFeatures
from feature_engine.scaling import MeanNormalizationScaler
from feature_engine.selection import DropFeatures

rng = np.random.default_rng(0)
N_ROWS, N_COLS = 5000, 300
X = pd.DataFrame(
    rng.normal(size=(N_ROWS, N_COLS)),
    columns=[f"col_{i}" for i in range(N_COLS)],
)

pipe = Pipeline(
    [
        ("scale", MeanNormalizationScaler()),
        ("math", MathFeatures(variables=X.columns.tolist()[:50], func=["sum", "mean", "max"])),
        ("rename", RenameFeatures({"col_0": "col_0_renamed"})),
        ("drop", DropFeatures(features_to_drop=X.columns.tolist()[-10:])),
    ]
).fit(X)


def time_it(fn, repeats=5):
    fn()  # warm-up
    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    return (time.perf_counter() - start) / repeats


# baseline: transform only, no snapshot machinery
baseline = time_it(lambda: pipe.transform(X))

# snapshot with data (runs the chain once itself, plus tracing)
with_data = time_it(lambda: pipe.create_feature_snapshot(X))

# metadata-only snapshot: no data execution, get_feature_names_out only
meta_only = time_it(lambda: pipe.create_feature_snapshot())

tracemalloc.start()
pipe.transform(X)
_, peak_base = tracemalloc.get_traced_memory()
tracemalloc.stop()

tracemalloc.start()
pipe.create_feature_snapshot(X)
_, peak_snap = tracemalloc.get_traced_memory()
tracemalloc.stop()

snap = pipe.create_feature_snapshot(X)
import json
on_disk = len(json.dumps(snap.to_dict()).encode("utf-8"))

print(f"rows={N_ROWS}, cols(in)={N_COLS}, cols(out)={len(snap.feature_names)}")
print(f"transform only (baseline)        : {baseline*1000:8.2f} ms")
print(f"create_feature_snapshot(X)       : {with_data*1000:8.2f} ms")
print(f"  of which tracing overhead      : {(with_data-baseline)*1000:8.2f} ms")
print(f"create_feature_snapshot() meta   : {meta_only*1000:8.2f} ms")
print(f"peak python memory, transform    : {peak_base/1024/1024:8.2f} MiB")
print(f"peak python memory, snapshot     : {peak_snap/1024/1024:8.2f} MiB")
print(f"snapshot JSON size (no data)     : {on_disk/1024:8.1f} KiB")
