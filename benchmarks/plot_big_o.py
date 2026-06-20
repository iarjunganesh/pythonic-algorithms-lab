"""
Generate Big-O complexity landscape chart for all 66 algorithms.

CPU panel: 45 algorithms grouped by category
GPU panel: 21 kernels grouped by sub-package

Output: assets/plots/big_o_landscape.png
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "assets" / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Complexity tiers (left → right = faster → slower) ──────────────────────
TIERS = [
    "O(1)",
    "O(log n)",
    "O(√n)",
    "O(n)",
    "O(n log log n)",
    "O(n log n)",
    "O(n²)",
    "O(n³)",
    "O(n!)",
]
CX = {t: i for i, t in enumerate(TIERS)}
X_MAX = len(TIERS) - 1

# ── Category → colour ───────────────────────────────────────────────────────
CAT_COLOR = {
    "Sorting":          "#3A6EA8",
    "Searching":        "#D4722A",
    "Graph":            "#3A8A5A",
    "Dyn. Prog.":       "#B03040",
    "Data Struct.":     "#6B5FA0",
    "Strings":          "#7A6048",
    "Geometry":         "#B060A0",
    "Math":             "#505050",
    "Backtracking":     "#A07030",
    "Greedy":           "#3A9070",
    "GPU Sorting":      "#1255A0",
    "GPU Signal":       "#145A14",
    "GPU LinAlg":       "#A01010",
    "GPU Reduction":    "#C04800",
    "GPU Elementwise":  "#600090",
    "GPU Scan":         "#005060",
    "GPU Sparse":       "#40281A",
    "GPU Graph":        "#283040",
}

# ── CPU algorithm groups: (display_name, complexity_tier) ───────────────────
#   Complexity = worst-case unless noted
CPU_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("Sorting", [
        ("Timsort",         "O(n log n)"),    # O(n) best; O(n log n) avg+worst
        ("Merge Sort",      "O(n log n)"),
        ("Heap Sort",       "O(n log n)"),
        ("Quick Sort †",    "O(n log n)"),    # O(n²) worst; plotted at average — marked †
        ("Counting Sort",   "O(n)"),          # O(n + k); k ∝ n in this lab
        ("Radix Sort",      "O(n)"),          # O(d·n); d constant
        ("Shell Sort",      "O(n²)"),         # halving-gap variant; worst O(n²)
        ("Insertion Sort",  "O(n²)"),
        ("Selection Sort",  "O(n²)"),
        ("Bubble Sort",     "O(n²)"),
    ]),
    ("Searching", [
        ("Binary Search",       "O(log n)"),
        ("Fibonacci Search",    "O(log n)"),
        ("Jump Search",         "O(√n)"),
        ("Linear Search",       "O(n)"),
    ]),
    ("Graph", [
        ("Dijkstra",    "O(n log n)"),   # O((V+E) log V)
        ("BFS",         "O(n)"),         # O(V+E)
        ("DFS",         "O(n)"),         # O(V+E)
    ]),
    ("Dyn. Prog.", [
        ("Fibonacci (memo)",    "O(n)"),
        ("0/1 Knapsack",        "O(n²)"),    # O(n·W); W treated as O(n)
    ]),
    ("Data Struct.", [
        ("Union-Find",      "O(1)"),     # amortized; inverse-Ackermann
        ("Bloom Filter",    "O(1)"),     # O(k) hash fns; k is constant
        ("Stack",           "O(1)"),     # push/pop
        ("Queue",           "O(1)"),     # enqueue/dequeue
        ("Linked List",     "O(1)"),     # head insert
        ("Doubly LL",       "O(1)"),     # head/tail insert
        ("AVL Tree",        "O(log n)"),
        ("Min-Heap",        "O(log n)"), # push/pop
        ("Skip List",       "O(log n)"), # avg
        ("BST",             "O(n)"),     # worst (unbalanced)
        ("Trie",            "O(n)"),     # O(key_length)
    ]),
    ("Strings", [
        ("Suffix Array",    "O(n log n)"),
        ("KMP Search",      "O(n)"),         # O(n + m)
        ("Rabin-Karp †",    "O(n)"),         # avg; O(nm) worst
    ]),
    ("Geometry", [
        ("Convex Hull",     "O(n log n)"),   # monotone chain
        ("Closest Pair",    "O(n²)"),        # brute-force impl
    ]),
    ("Math", [
        ("Matmul (NumPy)",      "O(n³)"),
        ("FFT (NumPy)",         "O(n log n)"),
        ("Sieve of Erath.",     "O(n log log n)"),
        ("SpMV (CPU)",          "O(n)"),     # O(nnz)
        ("Prefix Sum (CPU)",    "O(n)"),
        ("Sum Reduce (CPU)",    "O(n)"),
        ("Max Reduce (CPU)",    "O(n)"),
        ("GCD (Euclidean)",     "O(log n)"),
    ]),
    ("Backtracking", [
        ("N-Queens",        "O(n!)"),
    ]),
    ("Greedy", [
        ("Coin Change",     "O(n log n)"),
    ]),
]

# ── GPU kernel groups ────────────────────────────────────────────────────────
GPU_GROUPS: list[tuple[str, list[tuple[str, str]]]] = [
    ("GPU Sorting", [
        ("GPU Sort",        "O(n log n)"),
        ("CuPy Sort",       "O(n log n)"),
        ("Radix (CuPy)",    "O(n log n)"),
    ]),
    ("GPU Signal", [
        ("FFT (CuPy)",      "O(n log n)"),
        ("Convolve",        "O(n)"),
    ]),
    ("GPU LinAlg", [
        ("Matmul (CuPy)",   "O(n³)"),
        ("Outer Product",   "O(n²)"),
        ("L2 Norm",         "O(n)"),
    ]),
    ("GPU Reduction", [
        ("Sum Reduce",      "O(n)"),
        ("Max Reduce",      "O(n)"),
        ("Min Reduce",      "O(n)"),
        ("Mean Reduce",     "O(n)"),
        ("Dot Product",     "O(n)"),
    ]),
    ("GPU Scan", [
        ("Prefix Sum",      "O(n)"),
        ("Diff (1st order)","O(n)"),
    ]),
    ("GPU Elementwise", [
        ("ArgSort",         "O(n log n)"),
        ("Clip",            "O(n)"),
        ("Histogram",       "O(n)"),
        ("Softmax",         "O(n)"),
    ]),
    ("GPU Sparse", [
        ("SpMV (GPU)",      "O(n)"),     # O(nnz)
    ]),
    ("GPU Graph", [
        ("BFS Frontier",    "O(n)"),     # O(V+E); naive impl
    ]),
]


# ── Build flat row list for each panel ──────────────────────────────────────
def build_rows(groups):
    """Return [(label, cat_or_None, cx_or_None)] top-to-bottom."""
    rows: list[tuple[str, str | None, int | None]] = []
    for cat, algos in groups:
        rows.append((cat, None, None))          # group header
        for name, tier in algos:
            rows.append((name, cat, CX[tier]))
    return rows


cpu_rows = build_rows(CPU_GROUPS)
gpu_rows = build_rows(GPU_GROUPS)

# ── Constants ────────────────────────────────────────────────────────────────
ROW_H  = 0.40   # inches per row
FIG_W  = 16

BAND_COLORS = ["#EEF2F7", "#F8F9FA"]   # alternating column bands


def draw_panel(ax: plt.Axes, rows: list, title: str) -> None:
    n = len(rows)
    ax.set_xlim(-0.55, X_MAX + 0.55)
    ax.set_ylim(-0.6, n - 0.4)
    ax.invert_yaxis()
    ax.set_facecolor("#F8F9FA")

    # Alternating vertical bands per complexity tier
    for xi, _tier in enumerate(TIERS):
        ax.axvspan(xi - 0.5, xi + 0.5, color=BAND_COLORS[xi % 2], zorder=0)

    # Vertical guide lines
    for xi in range(len(TIERS)):
        ax.axvline(xi, color="#D0D8E0", linewidth=0.6, zorder=1, linestyle=":")

    ytick_pos: list[float] = []
    ytick_lbl: list[str]   = []
    ytick_cat: list[str | None] = []

    for yi, (label, cat, cx) in enumerate(rows):
        ytick_pos.append(yi)
        if cat is None:
            # ── Group header ──────────────────────────────────────────────
            ytick_lbl.append(f"▸ {label}")
            ytick_cat.append(None)
            ax.axhline(yi - 0.45, color="#B8C4D0", linewidth=0.8, zorder=2)
        else:
            # ── Algorithm row ─────────────────────────────────────────────
            ytick_lbl.append(f"  {label}")
            ytick_cat.append(cat)
            color = CAT_COLOR[cat]

            # Full-width track (very faint)
            ax.plot(
                [-0.5, X_MAX + 0.5], [yi, yi],
                color="#E0E0E0", linewidth=0.8, zorder=2,
            )
            # Filled bar from x=0 to cx (alpha fill to show "reaching" complexity)
            if cx > 0:
                ax.barh(
                    yi, cx, left=0,
                    height=0.42,
                    color=color, alpha=0.12, zorder=3,
                )
            # Dot at complexity position
            ax.plot(
                cx, yi, "o",
                color=color, markersize=11,
                markeredgecolor="white", markeredgewidth=1.0,
                zorder=5,
            )

    # Y-axis tick labels
    ax.set_yticks(ytick_pos)
    ax.set_yticklabels(ytick_lbl, fontsize=12.0)

    tick_labels = ax.get_yticklabels()
    for tick_label, cat in zip(tick_labels, ytick_cat):
        if cat is None:
            tick_label.set_fontweight("bold")
            tick_label.set_fontsize(13.5)
            tick_label.set_color("#2C3E50")
        else:
            tick_label.set_color(CAT_COLOR[cat])

    # X-axis ticks
    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels(TIERS, fontsize=11.0, rotation=32, ha="right")
    ax.tick_params(axis="x", pad=4)
    ax.set_xlabel("Time Complexity  (worst case; † = average case plotted)", fontsize=13, labelpad=8)
    ax.set_title(title, fontsize=16, fontweight="bold", pad=12, color="#1A2533")

    for spine in ax.spines.values():
        spine.set_edgecolor("#C0CAD5")
        spine.set_linewidth(0.8)


def make_panel_fig(
    rows: list,
    groups,
    panel_title: str,
    suptitle_line1: str,
    out_name: str,
) -> None:
    n_rows = len(rows)
    fig_h  = n_rows * ROW_H + 3.2
    top_frac    = 1.7 / fig_h
    bottom_frac = 3.5 / fig_h

    fig, ax = plt.subplots(1, 1, figsize=(FIG_W, fig_h))
    fig.patch.set_facecolor("#F8F9FA")

    draw_panel(ax, rows, panel_title)

    subtitle = "pythonic-algorithms-lab  ·  RTX 5070 Laptop GPU  ·  Python 3.14"
    fig.suptitle(
        f"{suptitle_line1}\n{subtitle}",
        fontsize=20, fontweight="bold", color="#1A2533",
        y=1.0 - 0.12 / fig_h,
        linespacing=1.5,
    )

    # Legend — only categories present in this panel
    cats_in_panel = {cat for _, cat, _ in rows if cat is not None}
    legend_patches = [
        mpatches.Patch(facecolor=CAT_COLOR[cat], alpha=0.75, label=cat)
        for cat in CAT_COLOR if cat in cats_in_panel
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=11.0,
        frameon=True,
        framealpha=0.95,
        edgecolor="#C0CAD5",
        bbox_to_anchor=(0.5, 0.01),
        title="Category",
        title_fontsize=13.0,
        handlelength=1.2,
    )

    # Complexity distribution box (right margin)
    dist: dict[str, int] = {t: 0 for t in TIERS}
    for _cat, algos in groups:
        for _name, tier in algos:
            dist[tier] += 1
    n_algos = sum(dist.values())
    summary_lines = [f"Tier distribution ({n_algos}):"] + [
        f"  {t:18s} {dist[t]:2d}" for t in TIERS if dist[t] > 0
    ]
    fig.text(
        0.995, 0.5,
        "\n".join(summary_lines),
        ha="right", va="center",
        fontsize=10.5,
        color="#40505F",
        fontfamily="monospace",
        transform=fig.transFigure,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#EEF2F7", edgecolor="#C0CAD5", linewidth=0.8),
    )

    fig.subplots_adjust(
        left=0.17, right=0.92,
        top=1.0 - top_frac,
        bottom=bottom_frac,
    )

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out}")


make_panel_fig(
    cpu_rows, CPU_GROUPS,
    panel_title="CPU Algorithms  (45)",
    suptitle_line1="Big-O Complexity Landscape — CPU  (45 Algorithms)",
    out_name="big_o_cpu.png",
)

make_panel_fig(
    gpu_rows, GPU_GROUPS,
    panel_title="GPU Kernels  (21)",
    suptitle_line1="Big-O Complexity Landscape — GPU  (21 Kernels)",
    out_name="big_o_gpu.png",
)

# Print combined distribution for verification
print("\nComplexity distribution (combined):")
dist_all: dict[str, int] = {t: 0 for t in TIERS}
for groups in (CPU_GROUPS, GPU_GROUPS):
    for _cat, algos in groups:
        for _name, tier in algos:
            dist_all[tier] += 1
for tier in TIERS:
    if dist_all[tier]:
        print(f"  {tier:20s}  {dist_all[tier]:2d}")
print(f"  {'TOTAL':20s}  {sum(dist_all.values()):2d}")
