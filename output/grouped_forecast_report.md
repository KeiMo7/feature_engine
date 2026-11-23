Grouped forecasting features: output and verification report
==================================================================

1. Three entities, 3D real-time window
--------------------------------------

Input:
store         ts  sales  qty
    A 2022-01-01    1.0 10.0
    A 2022-01-02    2.0 20.0
    A 2022-01-03    3.0 30.0
    A 2022-01-06    6.0 60.0
    A 2022-01-07    7.0 70.0
    B 2022-01-01   10.0  1.0
    B 2022-01-03   30.0  3.0
    B 2022-01-05   50.0  5.0
    C 2022-01-02  100.0  2.0
    C 2022-01-04  200.0  4.0
    C 2022-01-09  700.0  9.0

GroupedWindowFeatures(window='3D', functions=['mean', 'sum']):
store         ts  sales  qty  sales_window_3D_mean  sales_window_3D_sum  qty_window_3D_mean  qty_window_3D_sum
    A 2022-01-01    1.0 10.0                   NaN                  NaN                 NaN                NaN
    A 2022-01-02    2.0 20.0                   1.0                  1.0                10.0               10.0
    A 2022-01-03    3.0 30.0                   1.5                  3.0                15.0               30.0
    A 2022-01-06    6.0 60.0                   3.0                  3.0                30.0               30.0
    A 2022-01-07    7.0 70.0                   6.0                  6.0                60.0               60.0
    B 2022-01-01   10.0  1.0                   NaN                  NaN                 NaN                NaN
    B 2022-01-03   30.0  3.0                  10.0                 10.0                 1.0                1.0
    B 2022-01-05   50.0  5.0                  30.0                 30.0                 3.0                3.0
    C 2022-01-02  100.0  2.0                   NaN                  NaN                 NaN                NaN
    C 2022-01-04  200.0  4.0                 100.0                100.0                 2.0                2.0
    C 2022-01-09  700.0  9.0                   NaN                  NaN                 NaN                NaN

Hand-computed reference for sales (mean / sum):
store         ts  expected_mean  expected_sum
    A 2022-01-01            NaN           NaN
    A 2022-01-02            1.0           1.0
    A 2022-01-03            1.5           3.0
    A 2022-01-06            3.0           3.0
    A 2022-01-07            6.0           6.0
    B 2022-01-01            NaN           NaN
    B 2022-01-03           10.0          10.0
    B 2022-01-05           30.0          30.0
    C 2022-01-02            NaN           NaN
    C 2022-01-04          100.0         100.0
    C 2022-01-09            NaN           NaN

Matches hand computation: True

2. Forecast horizon ('1D')
--------------------------

store         ts  sales  sales_window_3D_sum
    A 2022-01-01    1.0                  NaN
    A 2022-01-02    2.0                  1.0
    A 2022-01-03    3.0                  3.0
    A 2022-01-06    6.0                  5.0
    A 2022-01-07    7.0                  9.0
    B 2022-01-01   10.0                  NaN
    B 2022-01-03   30.0                 10.0
    B 2022-01-05   50.0                 40.0
    C 2022-01-02  100.0                  NaN
    C 2022-01-04  200.0                100.0
    C 2022-01-09  700.0                  NaN

For a row at t, cut-off is t - 1D and the window is [cut-off - 3D, cut-off). The current row and future rows never contribute.

3. Lag features inside each entity
----------------------------------

store         ts  sales  sales_lag_1  sales_lag_2
    A 2022-01-01    1.0          NaN          NaN
    A 2022-01-02    2.0          1.0          NaN
    A 2022-01-03    3.0          2.0          1.0
    A 2022-01-06    6.0          3.0          2.0
    A 2022-01-07    7.0          6.0          3.0
    B 2022-01-01   10.0          NaN          NaN
    B 2022-01-03   30.0         10.0          NaN
    B 2022-01-05   50.0         30.0         10.0
    C 2022-01-02  100.0          NaN          NaN
    C 2022-01-04  200.0        100.0          NaN
    C 2022-01-09  700.0        200.0        100.0

4. Expanding window per entity
------------------------------

store         ts  sales  sales_expanding_mean  sales_expanding_count
    A 2022-01-01    1.0                   NaN                    NaN
    A 2022-01-02    2.0                   1.0                    1.0
    A 2022-01-03    3.0                   1.5                    2.0
    A 2022-01-06    6.0                   2.0                    3.0
    A 2022-01-07    7.0                   3.0                    4.0
    B 2022-01-01   10.0                   NaN                    NaN
    B 2022-01-03   30.0                  10.0                    1.0
    B 2022-01-05   50.0                  20.0                    2.0
    C 2022-01-02  100.0                   NaN                    NaN
    C 2022-01-04  200.0                 100.0                    1.0
    C 2022-01-09  700.0                 150.0                    2.0

5. Cold start for an unseen entity
----------------------------------

store         ts  sales  sales_window_2_mean
    A 2022-01-01    1.0                  NaN
    A 2022-01-02    2.0                  NaN
    D 2022-01-03    3.0                 -1.0
    A 2022-01-06    6.0                  1.5
    A 2022-01-07    7.0                  4.0
    B 2022-01-01   10.0                  NaN
    B 2022-01-03   30.0                  NaN
    B 2022-01-05   50.0                 20.0
    C 2022-01-02  100.0                  NaN
    C 2022-01-04  200.0                  NaN
    C 2022-01-09  700.0                150.0

Default cold-start policy message: The dataframe to transform contains entities that were not seen during fit (1 row(s), entity keys (first 10): [{'store': 'D'}], time range 2022-01-03 00:00:00 to 2022-01-03 00:00:00)

6. Duplicate timestamps
-----------------------

store         ts  sales  sales_lag_1
    A 2022-01-01    1.0          NaN
    A 2022-01-02    2.0          1.0
    A 2022-01-02    4.0          1.0
    A 2022-01-03    8.0          4.0

Default duplicates='raise' message: Found 1 duplicated (entity, timestamp) combination(s) covering 2 row(s). Entity keys (first 10): [{'store': 'A'}]. Time range: 2022-01-02 00:00:00 to 2022-01-02 00:00:00. Original row positions (0-based, first 20): [1, 2]. Keep the duplicates with `duplicates='first'`, `duplicates='last'` or `duplicates='keep'`, or remove them before fitting.
