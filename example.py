import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import statsmodels.api as sm

# Load the two datasets we previously created
profit_df = pd.read_csv("/mnt/data/dprofitshare_yoy_vs_prod_yoy_slice.csv")
wage_df = pd.read_csv("/mnt/data/dwageshare_yoy_vs_prod_yoy_data.csv")

profit_df["date"] = pd.to_datetime(profit_df["date"])
wage_df["date"] = pd.to_datetime(wage_df["date"])

# Merge to ensure identical sample (same quarters/dates)
df = profit_df.merge(
    wage_df[["date", "d_wage_share_yoy_pp"]],
    on="date",
    how="inner"
).dropna(subset=["prod_yoy_pct", "d_profit_share_yoy_pp", "d_wage_share_yoy_pp"]).copy()

x = df["prod_yoy_pct"].astype(float).values
X = sm.add_constant(x)

def fit_ols_hac(y, maxlags=4):
    """
    Fit OLS and report HAC(Newey-West) t-stat for slope.
    maxlags=4 corresponds to ~1 year of quarterly lags.
    """
    y = np.asarray(y, dtype=float)
    ols = sm.OLS(y, X).fit()
    hac = sm.OLS(y, X).fit(cov_type="HAC", cov_kwds={"maxlags": maxlags})
    intercept, slope = ols.params
    t_hac = hac.tvalues[1]
    r2 = ols.rsquared
    corr = np.corrcoef(x, y)[0, 1]
    return intercept, slope, t_hac, r2, corr

def make_plot(y, intercept, slope, t_hac, r2, corr, title, ylab, out_path, maxlags=4):
    fig, ax = plt.subplots(figsize=(6.6, 5.0))
    ax.scatter(x, y, alpha=0.7)

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xs, intercept + slope * xs, linewidth=2, color="red")

    ax.axhline(0, linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_xlabel("Productivity growth (y/y, %)")
    ax.set_ylabel(ylab)
    ax.set_title(title)

    eq = f"y = {intercept:.3f} + {slope:.3f}x"
    stats = f"Corr = {corr:.3f}\nHAC t(slope) = {t_hac:.2f} (L={maxlags})\n$R^2$ = {r2:.3f}\n{eq}"
    ax.text(
        0.03, 0.97, stats,
        transform=ax.transAxes,
        va="top", ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85)
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close(fig)

# --- Profit share regression ---
b0_p, b1_p, t_p_hac, r2_p, corr_p = fit_ols_hac(df["d_profit_share_yoy_pp"].values, maxlags=4)
p1 = "/mnt/data/scatter_dprofitshare_vs_prod_yoy_HACstats.png"
make_plot(
    df["d_profit_share_yoy_pp"].values, b0_p, b1_p, t_p_hac, r2_p, corr_p,
    "Δ Profit Share (y/y, pp) vs Productivity Growth (y/y, %)",
    "Δ Profit share (pp vs year ago)",
    p1,
    maxlags=4
)

# --- Wage share regression ---
b0_w, b1_w, t_w_hac, r2_w, corr_w = fit_ols_hac(df["d_wage_share_yoy_pp"].values, maxlags=4)
p2 = "/mnt/data/scatter_dwageshare_vs_prod_yoy_HACstats.png"
make_plot(
    df["d_wage_share_yoy_pp"].values, b0_w, b1_w, t_w_hac, r2_w, corr_w,
    "Δ Wage Share (y/y, pp) vs Productivity Growth (y/y, %)",
    "Δ Wage share (pp vs year ago)",
    p2,
    maxlags=4
)

# Stitch side-by-side (since we avoid subplots per chart guidelines)
img1 = Image.open(p1)
img2 = Image.open(p2)
combo = Image.new("RGB", (img1.width + img2.width, max(img1.height, img2.height)), (255, 255, 255))
combo.paste(img1, (0, 0))
combo.paste(img2, (img1.width, 0))

out_combo = "/mnt/data/scatter_profit_wage_side_by_side_HACstats.png"
combo.save(out_combo)

out_combo, {
    "profit": {"corr": corr_p, "slope": b1_p, "hac_t": t_p_hac, "r2": r2_p},
    "wage": {"corr": corr_w, "slope": b1_w, "hac_t": t_w_hac, "r2": r2_w},
}
