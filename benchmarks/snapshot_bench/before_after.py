"""Drive the before/after comparison in two isolated processes."""
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO = Path("/home/wll/uid147")
ART = REPO / "artifacts"
BENCH = REPO / "benchmarks" / "snapshot_bench"
SITE = "/home/wll/.local/lib/python3.14/site-packages"

runner = BENCH / "_run_chain.py"
old_csv = ART / "output_baseline.csv"
new_csv = ART / "output_patched.csv"

env_base = {
    "PYTHONPATH": f"/tmp/fe_baseline:{SITE}",
}
env_new = {
    "PYTHONPATH": f"{REPO}:{SITE}",
}


def run(csv_path, env):
    cmd = [sys.executable, "-P", str(runner), str(csv_path)]
    subprocess.run(cmd, check=True, env={**__import__("os").environ, **env})


run(old_csv, env_base)
run(new_csv, env_new)

out_old = pd.read_csv(old_csv)
out_new = pd.read_csv(new_csv)

labels_match = out_old.columns.tolist() == out_new.columns.tolist()
values_match = out_old.equals(out_new)
max_diff = (out_old - out_new).abs().max()

report = {
    "n_rows": int(len(out_new)),
    "baseline_columns": out_old.columns.tolist(),
    "patched_columns": out_new.columns.tolist(),
    "column_labels_identical": bool(labels_match),
    "all_values_identical": bool(values_match),
    "max_abs_diff_per_column": {k: float(v) for k, v in max_diff.items()},
    "first_row_baseline": out_old.head(1).to_dict(orient="records")[0],
    "first_row_patched": out_new.head(1).to_dict(orient="records")[0],
}
(ART / "before_after_comparison.json").write_text(
    json.dumps(report, indent=2), encoding="utf-8"
)
print(json.dumps(report, indent=2))
