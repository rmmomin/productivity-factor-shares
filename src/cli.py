"""CLI entry point for the replication kit."""

import argparse
import sys
from pathlib import Path

from .analysis import export_results, run_regressions
from .build_dataset import build_analysis_dataset
from .plots import create_all_plots


def run_all(
    no_network: bool = False,
    cache_dir: str = "data/raw",
    output_dir: str = "data/processed",
    figures_dir: str = "figures",
    results_dir: str = "results",
) -> None:
    """Run the complete analysis pipeline.

    Args:
        no_network: If True, use only cached data
        cache_dir: Directory for raw FRED data cache
        output_dir: Directory for processed data output
        figures_dir: Directory for figure outputs
        results_dir: Directory for regression results
    """
    print("Building analysis dataset...")
    df = build_analysis_dataset(
        no_network=no_network,
        cache_dir=cache_dir,
        output_dir=output_dir,
    )
    print(f"  Dataset shape: {df.shape}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")

    print("\nRunning regressions...")
    results = run_regressions(df, maxlags=4)
    for name, result in results.items():
        print(f"  {name}: slope={result.slope:.4f}, HAC t={result.t_hac:.2f}, R2={result.r2:.3f}")

    print("\nExporting regression results...")
    json_path, csv_path = export_results(results, output_dir=results_dir)
    print(f"  JSON: {json_path}")
    print(f"  CSV: {csv_path}")

    print("\nCreating plots...")
    plot_paths = create_all_plots(
        df,
        results,
        figures_dir=figures_dir,
        processed_dir=output_dir,
    )
    for name, path in plot_paths.items():
        print(f"  {name}: {path}")

    print("\nPipeline complete!")
    print(f"\nOutputs:")
    print(f"  Processed data: {output_dir}/dshares_vs_prod.csv")
    print(f"  Regression results: {results_dir}/")
    print(f"  Figures: {figures_dir}/")


def main() -> int:
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Productivity and Factor Shares Replication Kit"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run-all", help="Run the complete analysis pipeline")
    run_parser.add_argument(
        "--no-network",
        action="store_true",
        help="Use only cached data (no network requests)",
    )
    run_parser.add_argument(
        "--cache-dir",
        default="data/raw",
        help="Directory for raw FRED data cache (default: data/raw)",
    )
    run_parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for processed data output (default: data/processed)",
    )
    run_parser.add_argument(
        "--figures-dir",
        default="figures",
        help="Directory for figure outputs (default: figures)",
    )
    run_parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory for regression results (default: results)",
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    if args.command == "run-all":
        run_all(
            no_network=args.no_network,
            cache_dir=args.cache_dir,
            output_dir=args.output_dir,
            figures_dir=args.figures_dir,
            results_dir=args.results_dir,
        )
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
