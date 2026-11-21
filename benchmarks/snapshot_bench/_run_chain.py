"""Run the chain under whichever feature_engine is first on PYTHONPATH.

Writes the transformed output to a parquet/csv path given as argv[1], and a
flag indicating whether RenameFeatures exists. Used for the before/after
acceptance comparison: once against the baseline package, once against the
patched package.
"""
import sys

import numpy as np
import pandas as pd

out_path = sys.argv[1]

from feature_engine.creation import MathFeatures
from feature_engine.pipeline import Pipeline
from feature_engine.selection import DropFeatures

rng = np.random.default_rng(42)
X = pd.DataFrame(
    {
        "x1": rng.normal(size=20),
        "x2": rng.normal(size=20),
        "x3": rng.integers(0, 100, size=20).astype(float),
    }
)

steps = [
    ("math", MathFeatures(variables=["x1", "x2"], func=["sum", "mean"])),
    ("drop", DropFeatures(features_to_drop=["x2"])),
]

try:
    from feature_engine.preprocessing import RenameFeatures

    steps.insert(
        1, ("rename", RenameFeatures(rename_dict={"x3": "x3_renamed"}))
    )
    has_rename = True
except ImportError:  # baseline package
    has_rename = False

pipe = Pipeline(steps).fit(X)
out = pipe.transform(X)
if not has_rename:
    # make the baseline chain semantically identical for the value comparison
    out = out.rename(columns={"x3": "x3_renamed"})

out.to_csv(out_path, index=False)
with open(out_path + ".meta", "w") as fh:
    fh.write(f"has_rename={has_rename}\n")
    fh.write("columns=" + "|".join(out.columns.tolist()) + "\n")
print("wrote", out_path, "has_rename=", has_rename)
