You are Claude Code. Create a robust replication kit (Python) that reproduces the scatter plots:
1) Δ Profit Share (y/y, pp) vs Productivity Growth (y/y, %)
2) Δ Wage Share (y/y, pp) vs Productivity Growth (y/y, %)
and annotates Corr, HAC(Newey–West) t-stat for slope (L=4), and R² on each plot. Regression lines must be red.
Also produce a binscatter (20 quantile bins) version and export all underlying data to CSV.

DATA SOURCES (FRED series IDs; include these in a config file and README with citations):
- Labor productivity: OPHNFB (BLS via FRED) — Nonfarm Business Sector: Labor Productivity (Output per Hour)
- GDP: GDP (BEA via FRED) — Gross Domestic Product
- Corporate profits with IVA and CCAdj: CPROFIT (BEA via FRED)
- Compensation of employees (wage share numerator): COE (BEA via FRED)
Optional: Recession indicator: USRECQ (FRED) for recession-colored dots or filtering

IMPLEMENTATION REQUIREMENTS
A) Repo structure (create all files):
- README.md (how to run; definitions; sources; outputs)
- pyproject.toml (or requirements.txt) with pinned deps
- src/
    - fred_client.py
    - build_dataset.py
    - analysis.py
    - plots.py
    - cli.py (entry point)
- data/
    - raw/ (cache FRED API responses as JSON/CSV)
    - processed/ (final merged dataset for analysis)
- figures/ (PNG outputs)
- results/ (regression tables as JSON/CSV)

B) FRED API usage:
- Require env var FRED_API_KEY
- Use https://api.stlouisfed.org/fred/series/observations
- Cache every download under data/raw/{series_id}.csv (or .json) so reruns don’t hit API.
- Include a function to fetch and store series metadata (title, units, frequency, seasonal adj, last updated) using /fred/series and save to data/raw/series_metadata.json.
- Add basic validation: all required series exist, are quarterly, and span the postwar sample (start 1947Q1). If mismatch, raise a clear error.

C) Sample + transformations (must match):
- Use quarterly data, postwar: observation_start=1947-01-01.
- Align series by quarter end date; inner-join to common sample.
- Productivity growth (y/y, %): prod_yoy_pct = 100 * (ln(OPHNFB_t) - ln(OPHNFB_{t-4}))
- Profit share level (% of GDP): profit_share_pct = 100 * CPROFIT / GDP
- Wage share level (% of GDP): wage_share_pct = 100 * COE / GDP
- Changes (y/y, percentage points):
    d_profit_share_yoy_pp = profit_share_pct - profit_share_pct.shift(4)
    d_wage_share_yoy_pp   = wage_share_pct   - wage_share_pct.shift(4)
- Final dataset columns:
    date, prod_yoy_pct, profit_share_pct, wage_share_pct,
    d_profit_share_yoy_pp, d_wage_share_yoy_pp
  Export to: data/processed/dshares_vs_prod.csv

D) Regressions + stats:
For each dependent variable y in {d_profit_share_yoy_pp, d_wage_share_yoy_pp}:
- Regress y_t on constant + prod_yoy_pct (OLS)
- Compute:
    - slope, intercept
    - R² (OLS)
    - Corr(x, y)
    - HAC(Newey–West) t-stat for slope with maxlags=4:
        statsmodels OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": 4})
- Save a machine-readable regression summary to results/regression_summary.json and .csv

E) Plots (must be publication-quality):
1) Side-by-side scatter panel (two separate plots shown side-by-side):
- Left: y = d_profit_share_yoy_pp, x = prod_yoy_pct
- Right: y = d_wage_share_yoy_pp, x = prod_yoy_pct
- Scatter points (alpha ~0.7)
- Red regression line
- Horizontal and vertical zero lines
- Annotation box in top-left containing:
    Corr = ...
    HAC t(slope) = ... (L=4)
    R² = ...
    Equation: y = intercept + slope*x
- Export: figures/scatter_profit_wage_side_by_side_HACstats.png

2) Binscatter side-by-side (20 quantile bins):
- Create 20 equal-count bins based on x (pd.qcut, drop duplicate bins if needed)
- For each bin compute mean x, mean y, n, std(y), se(y)=std/sqrt(n)
- Plot binned means with error bars (±1 SE) + red OLS line
- Same annotation box as above
- Export: figures/binscatter_profit_wage_side_by_side.png
- Export binned table to: data/processed/binscatter_binned_means_profit_wage.csv with columns:
    which, x_mean, y_mean, n, y_std, y_se

F) CLI:
- Provide a single command:
    python -m src.cli run-all
that:
    1) downloads/caches data from FRED
    2) builds processed dataset
    3) runs regressions (HAC)
    4) creates both figure panels + binscatter
    5) writes outputs to /data/processed, /figures, /results
- Print a short run summary to console (sample size, date range, key stats)

G) Reproducibility niceties:
- Deterministic output (set random seeds where relevant; binscatter is deterministic)
- Add a “no-network” mode that uses cached raw files if present.
- Include unit-ish tests or smoke tests (e.g., pytest) verifying:
    - expected columns exist
    - no missing values after final join
    - series date frequency is quarterly
    - output files are created

README CONTENT (must include):
- What the kit reproduces (figures + data)
- Exact variable definitions (formulas above)
- Series IDs + source notes:
    OPHNFB (BLS), GDP (BEA), CPROFIT (BEA), COE (BEA), all via FRED
- How to get a FRED API key and set FRED_API_KEY in .env file as in this directory
- How to run end-to-end and where outputs are saved

Deliver the full repo content (all files) and ensure the code runs as-is.
