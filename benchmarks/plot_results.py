"""Generate static PNG plots from a benchmark CSV for README embedding.

Usage:
    python benchmarks/plot_results.py --csv benchmarks/results_full.csv --out assets/plots/
"""
import argparse
from pathlib import Path

import numpy as np  # noqa: F401 — imported for pandas numeric ops
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
except ImportError:
    plt = None


def _load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["n"] = pd.to_numeric(df["n"], errors="coerce")
    df["mean"] = pd.to_numeric(df["mean"], errors="coerce")
    df["min"] = pd.to_numeric(df.get("min", pd.Series(dtype=float)), errors="coerce")
    df["max"] = pd.to_numeric(df.get("max", pd.Series(dtype=float)), errors="coerce")
    return df.dropna(subset=["n", "mean"])


def plot_timing_by_family(df: pd.DataFrame, out_dir: Path) -> None:
    """One PNG per algorithm family (unique prefix before '_') — mean ± band."""
    # Group algorithms by family: first word before underscore or the full name
    df = df.copy()
    df["family"] = df["algorithm"].str.split("_").str[0]
    families = sorted(df["family"].unique())

    for family in families:
        sub = df[df["family"] == family]
        algos = sorted(sub["algorithm"].unique())
        backends = sorted(sub["backend"].unique())
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        color_map = {a: colors[i % len(colors)] for i, a in enumerate(algos)}
        dash_map = {"cpu": "-", "gpu": "--"}

        fig, ax = plt.subplots(figsize=(10, 5))
        for algo in algos:
            for backend in backends:
                grp = sub[(sub["algorithm"] == algo) & (sub["backend"] == backend)].sort_values("n")
                if grp.empty:
                    continue
                x = grp["n"].values
                y = grp["mean"].values
                col = color_map[algo]
                ls = dash_map.get(backend, ":")
                ax.plot(x, y, ls, color=col, marker="o", markersize=4, label=f"{algo} ({backend})")
                if "min" in grp.columns and "max" in grp.columns:
                    ax.fill_between(x, grp["min"].values, grp["max"].values, alpha=0.12, color=col)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n (input size)")
        ax.set_ylabel("mean time (s)")
        ax.set_title(f"Timing — {family}")
        ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
        ax.legend(fontsize=8, loc="upper left", ncol=2)
        fig.tight_layout()
        out_path = out_dir / f"timing_{family}.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  wrote {out_path}")


def _canonical(name: str) -> str:
    """Strip backend-specific suffixes so CPU and GPU rows join on operation family."""
    for suffix in ("_numpy", "_cupy", "_numba", "_tiled"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return {"bfs_frontier": "bfs"}.get(name, name)


def plot_speedup(df: pd.DataFrame, out_dir: Path) -> None:
    """Single PNG: CPU÷GPU speedup vs n, one line per algorithm family."""
    d = df.copy()
    d["_key"] = d["algorithm"].apply(_canonical)
    cpu = d[d["backend"] == "cpu"][["_key", "n", "mean"]].rename(columns={"mean": "cpu_mean"})
    gpu = d[d["backend"] == "gpu"][["_key", "n", "mean"]].rename(columns={"mean": "gpu_mean"})
    merged = pd.merge(cpu, gpu, on=["_key", "n"])
    if merged.empty:
        print("  no paired CPU/GPU rows — skipping speedup plot")
        return
    merged["speedup"] = merged["cpu_mean"] / merged["gpu_mean"]
    merged = merged.sort_values(["_key", "n"])

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    algos = sorted(merged["_key"].unique())
    color_map = {a: colors[i % len(colors)] for i, a in enumerate(algos)}

    fig, ax = plt.subplots(figsize=(12, 6))
    for algo in algos:
        grp = merged[merged["_key"] == algo]
        ax.plot(grp["n"].values, grp["speedup"].values, "-o", color=color_map[algo], markersize=5, label=algo)

    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="break-even (1×)")
    ax.set_xscale("log")
    ax.set_xlabel("n (input size)", fontsize=13)
    ax.set_ylabel("Speedup (cpu / gpu)", fontsize=13)
    ax.set_title("GPU Speedup over CPU", fontsize=14)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.legend(fontsize=8, loc="upper left", ncol=3)
    fig.tight_layout()
    out_path = out_dir / "speedup_all.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_p95_vs_mean(df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter: p95 vs mean per (algorithm, backend, n) to reveal tail latency."""
    if "p95" not in df.columns or "mean" not in df.columns:
        print("  p95 column not present — skipping tail latency plot")
        return
    d = df.dropna(subset=["p95", "mean"]).copy()
    if d.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    backends = sorted(d["backend"].unique())
    colors = {"cpu": "#2196F3", "gpu": "#E91E63"}
    markers = {"cpu": "o", "gpu": "^"}
    for backend in backends:
        sub = d[d["backend"] == backend]
        ax.scatter(
            sub["mean"], sub["p95"],
            c=colors.get(backend, "#888"),
            marker=markers.get(backend, "s"),
            alpha=0.6, s=30, label=backend,
        )
    # diagonal y=x reference line
    lims = [min(d["mean"].min(), d["p95"].min()), max(d["mean"].max(), d["p95"].max())]
    ax.plot(lims, lims, "k--", linewidth=0.8, label="p95 = mean")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("mean time (s)")
    ax.set_ylabel("p95 time (s)")
    ax.set_title("Tail Latency: p95 vs mean")
    ax.legend()
    fig.tight_layout()
    out_path = out_dir / "tail_latency.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate static PNG plots from benchmark CSV")
    parser.add_argument("--csv", default="benchmarks/results_full.csv")
    parser.add_argument("--out", default="assets/plots")
    args = parser.parse_args()

    if plt is None:
        print("matplotlib is not installed. Run: pip install matplotlib")
        return

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"CSV not found: {csv_path}")
        return

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = _load(str(csv_path))
    print(f"Loaded {len(df)} rows from {csv_path}")

    plot_timing_by_family(df, out_dir)
    plot_speedup(df, out_dir)
    plot_p95_vs_mean(df, out_dir)

    print(f"\nDone. PNGs written to {out_dir}/")
    print("To embed in README:\n  ![Speedup](assets/plots/speedup_all.png)")


if __name__ == "__main__":
    main()
