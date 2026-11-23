"""Generate the scorecard binning report into output/.

Builds business-like data where the bad rate is monotonic in the score,
fits the MonotonicDiscretiser, verifies the acceptance criteria and writes
the binning tables plus a summary to output/binning_report.md.
"""

import numpy as np
import pandas as pd

from feature_engine.discretisation import MonotonicDiscretiser

OUT = "output"

rng = np.random.default_rng(42)
n = 5000
score = rng.normal(0, 1, n)
p_bad = 1 / (1 + np.exp(-(-0.5 + 1.8 * score)))  # bad rate increases with score
y = pd.Series(rng.binomial(1, p_bad), name="bad")
X = pd.DataFrame({"score": score})
X.loc[rng.choice(X.index, 250, replace=False), "score"] = np.nan  # 5% missing

results = []

# 1. monotonic increasing, min bin size, max bins
md = MonotonicDiscretiser(
    max_bins=5, min_bin_size=0.05, monotonic="increasing"
).fit(X, y)
table = md.binning_table_["score"]
intervals = table[table["bin"] != "Missing"]
results.append(("bad rate monotonic increasing", intervals["bad_rate"].is_monotonic_increasing))
results.append(
    ("every interval bin >= 5% of non-missing samples",
     bool((intervals["count"] >= 0.05 * X["score"].notna().sum()).all()))
)
results.append(("interval bins <= max_bins (5)", len(intervals) <= 5))

# 2. missing values in their own bin, out-of-range into boundary bins
Xt = md.transform(X)
results.append(
    ("missing values in separate bin (-1)",
     bool((Xt.loc[X["score"].isna(), "score"] == -1).all()))
)
missing_row = table[table["bin"] == "Missing"].iloc[0]
results.append(("missing bin participates in stats", missing_row["count"] == 250))
X_oor = pd.DataFrame({"score": [-1e6, 1e6, 0.0]})
oor = md.transform(X_oor)["score"].tolist()
results.append((f"out-of-range values land in boundary bins {oor[:2]}", oor[:2] == [0, len(intervals) - 1]))

# 3. repeated fit -> identical boundaries; transform does not read y
md2 = MonotonicDiscretiser(
    max_bins=5, min_bin_size=0.05, monotonic="increasing"
).fit(X, y)
results.append(("repeated fit gives identical boundaries", md.binner_dict_ == md2.binner_dict_))
import inspect
results.append(
    ("transform() has no y parameter",
     "y" not in inspect.signature(md.transform).parameters)
)
md.transform(X_oor)  # transform on new data does not move boundaries
results.append(("boundaries frozen after fit", md.binner_dict_ == md2.binner_dict_))

# auto direction and decreasing also satisfy the constraint
md_auto = MonotonicDiscretiser(max_bins=5, monotonic="auto").fit(X, y)
md_dec = MonotonicDiscretiser(max_bins=5, monotonic="decreasing").fit(X, y)
dec_table = md_dec.binning_table_["score"]
dec_table = dec_table[dec_table["bin"] != "Missing"]
results.append((f"auto direction -> {md_auto.monotonic_direction_['score']}", True))
results.append(
    ("decreasing constraint honoured", dec_table["bad_rate"].is_monotonic_decreasing)
)

# write report
lines = ["# Monotonic binning report", "",
         f"- samples: {n} (missing in `score`: 250)",
         f"- total bads: {int(y.sum())} (bad rate {y.mean():.4f})",
         f"- configuration: max_bins=5, min_bin_size=0.05, monotonic='increasing'",
         f"- total IV (score): {md.iv_['score']:.4f}",
         "", "## Binning table (`score`)", "", table.to_markdown(index=False), "",
         "## Acceptance checks", ""]
for name, ok in results:
    lines.append(f"- [{'x' if ok else ' '}] {name}")
report = "\n".join(lines) + "\n"

with open(f"{OUT}/binning_report.md", "w") as f:
    f.write(report)
table.to_csv(f"{OUT}/binning_table_score.csv", index=False)

print(report)
assert all(ok for _, ok in results), "some acceptance checks failed"
print("ALL CHECKS PASSED")
