# Monotonic binning report

- samples: 5000 (missing in `score`: 250)
- total bads: 2047 (bad rate 0.4094)
- configuration: max_bins=5, min_bin_size=0.05, monotonic='increasing'
- total IV (score): 1.7218

## Binning table (`score`)

| bin     |       lower |      upper |   count |   bad |   good |   bad_rate |       woe |          iv | is_boundary   |
|:--------|------------:|-----------:|--------:|------:|-------:|-----------:|----------:|------------:|:--------------|
| 0       | -inf        |  -0.529338 |    1425 |   119 |   1306 |  0.0835088 | -2.02915  | 0.779456    | True          |
| 1       |   -0.529338 |   0.114916 |    1187 |   356 |    831 |  0.299916  | -0.481253 | 0.0517326   | False         |
| 2       |    0.114916 |   0.793664 |    1188 |   679 |    509 |  0.571549  |  0.654619 | 0.104306    | False         |
| 3       |    0.793664 |   1.61658  |     712 |   571 |    141 |  0.801966  |  1.76508  | 0.40808     | False         |
| 4       |    1.61658  | inf        |     238 |   228 |     10 |  0.957983  |  3.49321  | 0.377253    | True          |
| Missing |  nan        | nan        |     250 |    94 |    156 |  0.376     | -0.140115 | 0.000967742 | False         |

## Acceptance checks

- [x] bad rate monotonic increasing
- [x] every interval bin >= 5% of non-missing samples
- [x] interval bins <= max_bins (5)
- [x] missing values in separate bin (-1)
- [x] missing bin participates in stats
- [x] out-of-range values land in boundary bins [0, 4]
- [x] repeated fit gives identical boundaries
- [x] transform() has no y parameter
- [x] boundaries frozen after fit
- [x] auto direction -> increasing
- [x] decreasing constraint honoured
