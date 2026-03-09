from __future__ import annotations

import argparse
import re
import statistics
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


FILE_NAME_PATTERN = re.compile(r"fileName:([^,]+)")


@dataclass
class DatasetRecord:
    name: str
    path: Path
    total_length: int
    anomaly_start: int
    anomaly_end: int
    anomaly_length: int
    anomaly_ratio: float


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Create dataset structure plots from an Isolation Forest results file."
    )
    parser.add_argument(
        "--results-file",
        type=Path,
        default=script_dir / "ISOLATION_FOREST_method.txt",
        help="Path to a results text file containing fileName entries.",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=script_dir.parent / "datasets" / "virgin",
        help="Directory that contains dataset .txt files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "dataset_structure_plots",
        help="Directory where plot images will be saved.",
    )
    parser.add_argument(
        "--linear-x",
        action="store_true",
        help="Use linear x-axis on the length vs anomaly ratio scatter plot.",
    )
    return parser.parse_args()


def extract_dataset_names(results_file: Path) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()

    with results_file.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            match = FILE_NAME_PATTERN.search(line)
            if not match:
                continue
            name = match.group(1).strip()
            if name and name not in seen:
                names.append(name)
                seen.add(name)

    return names


def parse_anomaly_bounds(dataset_name: str) -> tuple[int, int]:
    parts = dataset_name.split("_")
    if len(parts) < 6:
        raise ValueError(
            f"Dataset name does not match expected pattern: '{dataset_name}'"
        )
    try:
        anomaly_start = int(parts[-2])
        anomaly_end = int(parts[-1])
    except ValueError as exc:
        raise ValueError(
            f"Could not parse anomaly bounds from dataset name: '{dataset_name}'"
        ) from exc
    return anomaly_start, anomaly_end


def count_datapoints(dataset_file: Path) -> int:
    with dataset_file.open("r", encoding="utf-8", errors="ignore") as handle:
        return sum(1 for _ in handle)


def build_records(results_file: Path, dataset_dir: Path) -> tuple[list[DatasetRecord], list[str]]:
    records: list[DatasetRecord] = []
    issues: list[str] = []

    for dataset_name in extract_dataset_names(results_file):
        dataset_file = dataset_dir / f"{dataset_name}.txt"
        if not dataset_file.exists():
            issues.append(f"Missing dataset file: {dataset_file}")
            continue

        try:
            anomaly_start, anomaly_end = parse_anomaly_bounds(dataset_name)
        except ValueError as exc:
            issues.append(str(exc))
            continue

        total_length = count_datapoints(dataset_file)
        anomaly_length = max(0, anomaly_end - anomaly_start)
        anomaly_ratio = (anomaly_length / total_length) if total_length > 0 else 0.0

        records.append(
            DatasetRecord(
                name=dataset_name,
                path=dataset_file,
                total_length=total_length,
                anomaly_start=anomaly_start,
                anomaly_end=anomaly_end,
                anomaly_length=anomaly_length,
                anomaly_ratio=anomaly_ratio,
            )
        )

    return records, issues


def plot_dataset_length_histogram(lengths: list[int], output_path: Path) -> None:
    mean_length = statistics.fmean(lengths)
    median_length = statistics.median(lengths)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lengths, bins="auto", color="#4C78A8", edgecolor="black", alpha=0.85)
    ax.axvline(
        mean_length,
        color="#F58518",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean_length:,.0f}",
    )
    ax.axvline(
        median_length,
        color="#54A24B",
        linestyle="-.",
        linewidth=2,
        label=f"Median: {median_length:,.0f}",
    )
    ax.set_title("Histogram of Dataset Lengths")
    ax.set_xlabel("Number of datapoints")
    ax.set_ylabel("Number of datasets")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_anomaly_ratio_histogram(ratios: list[float], output_path: Path) -> None:
    positive_ratios = [ratio for ratio in ratios if ratio > 0]
    if not positive_ratios:
        raise ValueError("Cannot use log x-axis for anomaly ratios because all ratios are zero.")

    min_positive = min(positive_ratios)
    epsilon = min_positive / 10.0
    plot_values = [ratio if ratio > 0 else epsilon for ratio in ratios]
    max_ratio = max(plot_values)
    bins = np.logspace(np.log10(epsilon), np.log10(max_ratio), 30)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(plot_values, bins=bins, color="#72B7B2", edgecolor="black", alpha=0.85)
    ax.set_xscale("log")
    ax.set_xlim(epsilon, max_ratio)
    ax.set_title("Histogram of Anomaly Ratio (Anomaly Length / Dataset Length)")
    ax.set_xlabel("Anomaly ratio (log scale)")
    ax.set_ylabel("Number of datasets")
    ax.grid(axis="y", alpha=0.2)

    zero_count = len(ratios) - len(positive_ratios)
    if zero_count > 0:
        ax.text(
            0.99,
            0.95,
            f"{zero_count} zero-ratio datasets plotted at {epsilon:.1e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_length_vs_ratio_scatter(
    lengths: list[int], ratios: list[float], output_path: Path, linear_x: bool
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        lengths,
        ratios,
        color="#E45756",
        alpha=0.75,
        s=40,
        edgecolors="white",
        linewidths=0.4,
    )
    if not linear_x:
        ax.set_xscale("log")
    ax.set_title("Dataset Length vs Anomaly Ratio")
    ax.set_xlabel("Dataset length")
    ax.set_ylabel("Anomaly ratio")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    results_file = args.results_file.resolve()
    dataset_dir = args.dataset_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    records, issues = build_records(results_file, dataset_dir)
    if not records:
        raise RuntimeError("No valid dataset records were found.")

    output_dir.mkdir(parents=True, exist_ok=True)

    lengths = [record.total_length for record in records]
    ratios = [record.anomaly_ratio for record in records]

    length_hist_path = output_dir / "dataset_length_histogram.png"
    ratio_hist_path = output_dir / "anomaly_ratio_histogram.png"
    scatter_path = output_dir / "length_vs_anomaly_ratio_scatter.png"

    plot_dataset_length_histogram(lengths, length_hist_path)
    plot_anomaly_ratio_histogram(ratios, ratio_hist_path)
    plot_length_vs_ratio_scatter(lengths, ratios, scatter_path, linear_x=args.linear_x)

    print(f"Processed datasets: {len(records)}")
    print(f"Length mean: {statistics.fmean(lengths):.2f}")
    print(f"Length median: {statistics.median(lengths):.2f}")
    print(f"Anomaly ratio mean: {statistics.fmean(ratios):.6f}")
    print(f"Anomaly ratio median: {statistics.median(ratios):.6f}")
    print(f"Saved: {length_hist_path}")
    print(f"Saved: {ratio_hist_path}")
    print(f"Saved: {scatter_path}")

    if issues:
        print("\nWarnings:")
        for issue in issues:
            print(f"- {issue}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
