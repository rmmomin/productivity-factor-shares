"""Plotting functions for scatter and binscatter plots."""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

from .analysis import RegressionResult


def make_scatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    result: RegressionResult,
    title: str,
    ylabel: str,
    output_path: str,
    dpi: int = 200,
) -> Path:
    """Create scatter plot with regression line and statistics.

    Args:
        x: Independent variable values
        y: Dependent variable values
        result: RegressionResult with coefficients and statistics
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
        dpi: Output resolution

    Returns:
        Path to saved plot
    """
    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    ax.scatter(x, y, alpha=0.7)

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xs, result.intercept + result.slope * xs, linewidth=2, color="red")

    ax.axhline(0, linewidth=1, color="black")
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Productivity growth (y/y, %)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    eq = f"y = {result.intercept:.3f} + {result.slope:.3f}x"
    stats = (
        f"Corr = {result.correlation:.3f}\n"
        f"HAC t(slope) = {result.t_hac:.2f} (L={result.maxlags})\n"
        f"$R^2$ = {result.r2:.3f}\n"
        f"{eq}"
    )
    ax.text(
        0.03,
        0.97,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    return output_path


def compute_binscatter_data(
    x: np.ndarray,
    y: np.ndarray,
    n_bins: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute binscatter data with quantile bins.

    Args:
        x: Independent variable values
        y: Dependent variable values
        n_bins: Number of quantile bins

    Returns:
        Tuple of (bin_means_x, bin_means_y, bin_se_y)
    """
    x = np.asarray(x)
    y = np.asarray(y)

    quantiles = np.linspace(0, 100, n_bins + 1)
    bin_edges = np.percentile(x, quantiles)
    bin_edges[-1] = bin_edges[-1] + 1e-10

    bin_indices = np.digitize(x, bin_edges) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_means_x = np.zeros(n_bins)
    bin_means_y = np.zeros(n_bins)
    bin_se_y = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_means_x[i] = np.mean(x[mask])
            bin_means_y[i] = np.mean(y[mask])
            if mask.sum() > 1:
                bin_se_y[i] = np.std(y[mask], ddof=1) / np.sqrt(mask.sum())
            else:
                bin_se_y[i] = 0

    return bin_means_x, bin_means_y, bin_se_y


def make_binscatter_plot(
    x: np.ndarray,
    y: np.ndarray,
    result: RegressionResult,
    title: str,
    ylabel: str,
    output_path: str,
    n_bins: int = 20,
    dpi: int = 200,
) -> tuple[Path, pd.DataFrame]:
    """Create binscatter plot with error bars and regression line.

    Args:
        x: Independent variable values
        y: Dependent variable values
        result: RegressionResult with coefficients and statistics
        title: Plot title
        ylabel: Y-axis label
        output_path: Path to save the plot
        n_bins: Number of quantile bins
        dpi: Output resolution

    Returns:
        Tuple of (plot_path, binscatter_data_df)
    """
    bin_means_x, bin_means_y, bin_se_y = compute_binscatter_data(x, y, n_bins)

    fig, ax = plt.subplots(figsize=(6.6, 5.0))

    ax.errorbar(
        bin_means_x,
        bin_means_y,
        yerr=bin_se_y,
        fmt="o",
        capsize=3,
        alpha=0.7,
    )

    xs = np.linspace(np.nanmin(x), np.nanmax(x), 200)
    ax.plot(xs, result.intercept + result.slope * xs, linewidth=2, color="red")

    ax.axhline(0, linewidth=1, color="black")
    ax.axvline(0, linewidth=1, color="black")
    ax.set_xlabel("Productivity growth (y/y, %)")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title} (Binscatter, {n_bins} bins)")

    eq = f"y = {result.intercept:.3f} + {result.slope:.3f}x"
    stats = (
        f"Corr = {result.correlation:.3f}\n"
        f"HAC t(slope) = {result.t_hac:.2f} (L={result.maxlags})\n"
        f"$R^2$ = {result.r2:.3f}\n"
        f"{eq}"
    )
    ax.text(
        0.03,
        0.97,
        stats,
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=dpi)
    plt.close(fig)

    binscatter_df = pd.DataFrame(
        {
            "bin_mean_x": bin_means_x,
            "bin_mean_y": bin_means_y,
            "bin_se_y": bin_se_y,
        }
    )

    return output_path, binscatter_df


def stitch_images_side_by_side(
    image_paths: list[Path],
    output_path: str,
) -> Path:
    """Stitch multiple images side by side using PIL.

    Args:
        image_paths: List of paths to images
        output_path: Path to save stitched image

    Returns:
        Path to saved stitched image
    """
    images = [Image.open(p) for p in image_paths]

    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)

    combined = Image.new("RGB", (total_width, max_height), (255, 255, 255))

    x_offset = 0
    for img in images:
        combined.paste(img, (x_offset, 0))
        x_offset += img.width

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.save(output_path)

    for img in images:
        img.close()

    return output_path


def create_all_plots(
    df: pd.DataFrame,
    results: dict[str, RegressionResult],
    figures_dir: str = "figures",
    processed_dir: str = "data/processed",
) -> dict[str, Path]:
    """Create all plots for the analysis.

    Args:
        df: Analysis DataFrame
        results: Dictionary of RegressionResults
        figures_dir: Directory for figure outputs
        processed_dir: Directory for binscatter CSV outputs

    Returns:
        Dictionary mapping plot names to paths
    """
    x = df["prod_yoy_pct"].values
    output_paths = {}

    profit_scatter = make_scatter_plot(
        x,
        df["d_profit_share_yoy_pp"].values,
        results["profit"],
        "Delta Profit Share (y/y, pp) vs Productivity Growth (y/y, %)",
        "Delta Profit share (pp vs year ago)",
        f"{figures_dir}/scatter_profit_share.png",
    )
    output_paths["scatter_profit"] = profit_scatter

    wage_scatter = make_scatter_plot(
        x,
        df["d_wage_share_yoy_pp"].values,
        results["wage"],
        "Delta Wage Share (y/y, pp) vs Productivity Growth (y/y, %)",
        "Delta Wage share (pp vs year ago)",
        f"{figures_dir}/scatter_wage_share.png",
    )
    output_paths["scatter_wage"] = wage_scatter

    combined_scatter = stitch_images_side_by_side(
        [profit_scatter, wage_scatter],
        f"{figures_dir}/scatter_combined.png",
    )
    output_paths["scatter_combined"] = combined_scatter

    profit_binscatter, profit_bins_df = make_binscatter_plot(
        x,
        df["d_profit_share_yoy_pp"].values,
        results["profit"],
        "Delta Profit Share (y/y, pp) vs Productivity Growth (y/y, %)",
        "Delta Profit share (pp vs year ago)",
        f"{figures_dir}/binscatter_profit_share.png",
    )
    output_paths["binscatter_profit"] = profit_binscatter
    profit_bins_df.to_csv(f"{processed_dir}/binscatter_profit.csv", index=False)

    wage_binscatter, wage_bins_df = make_binscatter_plot(
        x,
        df["d_wage_share_yoy_pp"].values,
        results["wage"],
        "Delta Wage Share (y/y, pp) vs Productivity Growth (y/y, %)",
        "Delta Wage share (pp vs year ago)",
        f"{figures_dir}/binscatter_wage_share.png",
    )
    output_paths["binscatter_wage"] = wage_binscatter
    wage_bins_df.to_csv(f"{processed_dir}/binscatter_wage.csv", index=False)

    combined_binscatter = stitch_images_side_by_side(
        [profit_binscatter, wage_binscatter],
        f"{figures_dir}/binscatter_combined.png",
    )
    output_paths["binscatter_combined"] = combined_binscatter

    return output_paths
