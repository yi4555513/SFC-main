from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def edge_key(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    return tuple(sorted((a, b)))


def base_edges(planes: int, sats: int) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    edges: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for p in range(planes):
        for s in range(sats - 1):
            edges.add(edge_key((p, s), (p, s + 1)))
    return edges


def scenario_edges() -> list[set[tuple[tuple[int, int], tuple[int, int]]]]:
    planes, sats = 4, 5
    common = base_edges(planes, sats)
    return [
        common
        | {
            edge_key((0, 1), (1, 1)),
            edge_key((1, 2), (2, 2)),
            edge_key((2, 3), (3, 3)),
        },
        common
        | {
            edge_key((0, 2), (1, 2)),
            edge_key((1, 3), (2, 3)),
            edge_key((2, 1), (3, 1)),
        },
        common
        | {
            edge_key((0, 3), (1, 3)),
            edge_key((1, 1), (2, 1)),
            edge_key((2, 2), (3, 2)),
        },
        common
        | {
            edge_key((0, 1), (1, 2)),
            edge_key((1, 2), (2, 3)),
            edge_key((2, 3), (3, 4)),
        },
    ]


def draw_panel(ax: plt.Axes, idx: int, edges: set, previous: set | None, chinese: bool = False) -> None:
    positions = {(p, s): (p, -s) for p in range(4) for s in range(5)}

    if previous is None:
        kept = edges
        added: set = set()
        removed: set = set()
    else:
        kept = edges & previous
        added = edges - previous
        removed = previous - edges

    def draw_edges(edge_set: set, color: str, lw: float, style="-", alpha=1.0, zorder=1) -> None:
        for a, b in sorted(edge_set):
            x1, y1 = positions[a]
            x2, y2 = positions[b]
            ax.plot([x1, x2], [y1, y2], color=color, lw=lw, linestyle=style, alpha=alpha, zorder=zorder)

    draw_edges(kept, "#7F7F7F", 1.35, "-", 0.72, 1)
    draw_edges(removed, "#D55E00", 1.85, (0, (3.2, 2.2)), 0.98, 2)
    draw_edges(added, "#009E73", 2.05, "-", 0.98, 3)

    for (p, s), (x, y) in positions.items():
        ax.scatter(x, y, s=110, facecolors="white", edgecolors="#111111", linewidths=1.0, zorder=4)
        ax.text(x, y, f"S$_{{{p + 1},{s + 1}}}$", ha="center", va="center", fontsize=6.5, zorder=5)

    ax.set_xlim(-0.42, 3.42)
    ax.set_ylim(-4.42, 0.42)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(i) for i in range(1, 5)])
    ax.set_yticks([0, -1, -2, -3, -4])
    ax.set_yticklabels([str(i) for i in range(1, 6)])
    ax.tick_params(axis="both", labelsize=7, length=2.5, width=0.65, pad=1.5)
    ax.grid(True, color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.7)

    if chinese:
        if previous is None:
            title = f"（{chr(97 + idx)}）快照 $t_{{{idx}}}$"
        else:
            title = f"（{chr(97 + idx)}）快照 $t_{{{idx}}}$（+{len(added)}，-{len(removed)}）"
    elif previous is None:
        title = f"({chr(97 + idx)}) Snapshot $t_{idx}$"
    else:
        title = f"({chr(97 + idx)}) Snapshot $t_{idx}$  (+{len(added)}, -{len(removed)})"
    ax.set_title(title, fontsize=9.5, pad=4)


def make_figure(chinese: bool = False) -> None:
    font_family = "SimSun" if chinese else "Times New Roman"
    plt.rcParams.update(
        {
            "font.family": font_family,
            "font.sans-serif": ["SimSun", "SimHei", "Microsoft YaHei", "Arial"],
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": False,
        }
    )

    edges_by_time = scenario_edges()
    fig, axes = plt.subplots(2, 2, figsize=(6.65, 5.25), dpi=300)
    axes = axes.flatten()

    previous = None
    for idx, (ax, edges) in enumerate(zip(axes, edges_by_time)):
        draw_panel(ax, idx, edges, previous, chinese=chinese)
        previous = edges

    if chinese:
        fig.supxlabel("轨道面编号", fontsize=9.5, y=0.06)
        fig.supylabel("轨道内卫星编号", fontsize=9.5, x=0.055)
    else:
        fig.supxlabel("Orbital plane index", fontsize=9.5, y=0.06)
        fig.supylabel("Satellite index within an orbital plane", fontsize=9.5, x=0.055)

    labels = (
        ("持续星间链路", "新建星间链路", "断开星间链路", "卫星节点")
        if chinese
        else ("Persistent ISL", "Newly established ISL", "Disconnected ISL", "Satellite node")
    )
    handles = [
        Line2D([0], [0], color="#7F7F7F", lw=1.7, label=labels[0]),
        Line2D([0], [0], color="#009E73", lw=2.1, label=labels[1]),
        Line2D([0], [0], color="#D55E00", lw=1.9, linestyle=(0, (3.2, 2.2)), label=labels[2]),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            markerfacecolor="white",
            markeredgewidth=1.0,
            markersize=5.2,
            lw=0,
            label=labels[3],
        ),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.006),
        ncol=4,
        frameon=False,
        fontsize=8.4,
        handlelength=2.5,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.105, right=0.985, top=0.94, bottom=0.165, wspace=0.17, hspace=0.25)

    output_dir = Path("paper_figures/leo_topology_transition_schematic")
    output_dir.mkdir(parents=True, exist_ok=True)
    base_name = "leo_topology_transition_schematic_zh" if chinese else "leo_topology_transition_schematic"
    base = output_dir / base_name
    for suffix in [".svg", ".pdf", ".png"]:
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    make_figure(chinese=False)
    make_figure(chinese=True)


if __name__ == "__main__":
    main()
