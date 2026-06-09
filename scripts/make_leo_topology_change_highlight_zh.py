from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


Node = tuple[int, int]
Edge = tuple[Node, Node]


def edge_key(a: Node, b: Node) -> Edge:
    return tuple(sorted((a, b)))


def base_edges(planes: int, sats: int) -> set[Edge]:
    edges: set[Edge] = set()
    for p in range(planes):
        for s in range(sats - 1):
            edges.add(edge_key((p, s), (p, s + 1)))
    return edges


def transition_edges() -> tuple[set[Edge], set[Edge]]:
    common = base_edges(4, 5)
    t0 = common | {
        edge_key((0, 1), (1, 1)),
        edge_key((1, 2), (2, 2)),
        edge_key((2, 3), (3, 3)),
    }
    t1 = common | {
        edge_key((0, 2), (1, 2)),
        edge_key((1, 3), (2, 3)),
        edge_key((2, 1), (3, 1)),
    }
    return t0, t1


def draw_nodes(ax: plt.Axes, positions: dict[Node, tuple[float, float]], label_nodes: bool = True) -> None:
    for (p, s), (x, y) in positions.items():
        ax.scatter(x, y, s=135, facecolors="white", edgecolors="#111111", linewidths=1.05, zorder=4)
        if label_nodes:
            ax.text(x, y, f"S$_{{{p + 1},{s + 1}}}$", ha="center", va="center", fontsize=5.9, zorder=5)


def draw_edges(
    ax: plt.Axes,
    edges: set[Edge],
    positions: dict[Node, tuple[float, float]],
    color: str,
    lw: float,
    linestyle="-",
    alpha: float = 1.0,
    zorder: int = 1,
) -> None:
    for a, b in sorted(edges):
        x1, y1 = positions[a]
        x2, y2 = positions[b]
        ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=linestyle, alpha=alpha, zorder=zorder)


def setup_axis(ax: plt.Axes, title: str) -> None:
    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(-4.42, 0.42)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(i) for i in range(1, 5)])
    ax.set_yticks([0, -1, -2, -3, -4])
    ax.set_yticklabels([str(i) for i in range(1, 6)])
    ax.tick_params(axis="both", labelsize=7, length=2.3, width=0.65, pad=1.2)
    ax.grid(True, color="#E7E7E7", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=10.2, pad=5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("#222222")


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "SimSun",
            "font.sans-serif": ["SimSun", "SimHei", "Microsoft YaHei", "Arial"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    positions = {(p, s): (p, -s) for p in range(4) for s in range(5)}
    t0, t1 = transition_edges()
    kept = t0 & t1
    added = t1 - t0
    removed = t0 - t1

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 3.08), dpi=300)

    setup_axis(axes[0], "（a）当前拓扑快照 $t_k$")
    draw_edges(axes[0], t0, positions, "#7F7F7F", 1.35, alpha=0.85)
    draw_nodes(axes[0], positions)

    setup_axis(axes[1], "（b）星间链路切换")
    draw_edges(axes[1], kept, positions, "#C9C9C9", 0.8, alpha=0.34)
    draw_edges(axes[1], removed, positions, "#D55E00", 2.45, linestyle=(0, (3.2, 2.0)), alpha=1.0, zorder=3)
    draw_edges(axes[1], added, positions, "#009E73", 2.75, alpha=1.0, zorder=4)
    draw_nodes(axes[1], positions)
    setup_axis(axes[2], "（c）下一拓扑快照 $t_{k+1}$")
    draw_edges(axes[2], t1, positions, "#7F7F7F", 1.35, alpha=0.85)
    draw_edges(axes[2], added, positions, "#009E73", 2.15, alpha=0.95, zorder=3)
    draw_nodes(axes[2], positions)

    axes[0].set_ylabel("轨道内卫星编号", fontsize=9.3)
    for ax in axes:
        ax.set_xlabel("轨道面编号", fontsize=9.0, labelpad=2)

    handles = [
        Line2D([0], [0], color="#7F7F7F", lw=1.7, label="原有链路"),
        Line2D([0], [0], color="#009E73", lw=2.6, label="新建链路"),
        Line2D([0], [0], color="#D55E00", lw=2.3, linestyle=(0, (3.2, 2.0)), label="断开链路"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=5.0,
            lw=0,
            label="卫星节点",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.005), ncol=4, frameon=False, fontsize=8.8)
    fig.subplots_adjust(left=0.065, right=0.99, top=0.88, bottom=0.25, wspace=0.22)

    output_dir = Path("paper_figures/leo_topology_change_highlight")
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "leo_topology_change_highlight_zh"
    for suffix in [".svg", ".pdf", ".png"]:
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


if __name__ == "__main__":
    main()
