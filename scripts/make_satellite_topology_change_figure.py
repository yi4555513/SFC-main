from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.lines import Line2D


def edge_key(edge: tuple[str, str]) -> tuple[str, str]:
    return tuple(sorted(edge))


def snapshot_index(path: Path) -> int:
    match = re.search(r"_t(\d+)\.gml$", path.name)
    if not match:
        raise ValueError(f"Cannot parse snapshot index from {path}")
    return int(match.group(1))


def read_snapshots(snapshot_dir: Path) -> dict[int, nx.Graph]:
    files = sorted(snapshot_dir.glob("satellite_graph_t*.gml"), key=snapshot_index)
    if not files:
        raise FileNotFoundError(f"No snapshot GML files found in {snapshot_dir}")
    return {snapshot_index(path): nx.read_gml(path) for path in files}


def build_grid_positions(graph: nx.Graph) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {}
    for node, attrs in graph.nodes(data=True):
        _, plane, sat = attrs["sequence_num"]
        positions[node] = (float(plane), float(-sat))
    return positions


def draw_snapshot(
    ax: plt.Axes,
    graph: nx.Graph,
    positions: dict[str, tuple[float, float]],
    current_edges: set[tuple[str, str]],
    previous_edges: set[tuple[str, str]] | None,
    index: int,
) -> tuple[int, int, int]:
    if previous_edges is None:
        kept = current_edges
        added: set[tuple[str, str]] = set()
        removed: set[tuple[str, str]] = set()
    else:
        kept = current_edges & previous_edges
        added = current_edges - previous_edges
        removed = previous_edges - current_edges

    for edges, color, width, style, alpha, zorder in [
        (kept, "#8B8B8B", 0.74, "-", 0.62, 1),
        (removed, "#D55E00", 1.12, (0, (3.0, 2.2)), 0.95, 2),
        (added, "#009E73", 1.20, "-", 0.95, 3),
    ]:
        nx.draw_networkx_edges(
            graph,
            positions,
            edgelist=list(edges),
            ax=ax,
            edge_color=color,
            width=width,
            style=style,
            alpha=alpha,
            arrows=False,
            node_size=18,
        )
        for line in ax.collections[-1:]:
            line.set_zorder(zorder)

    xs = [positions[node][0] for node in graph.nodes]
    ys = [positions[node][1] for node in graph.nodes]
    ax.scatter(xs, ys, s=18, facecolors="white", edgecolors="#111111", linewidths=0.75, zorder=4)

    planes = sorted({int(graph.nodes[node]["sequence_num"][1]) for node in graph.nodes})
    sats = sorted({int(graph.nodes[node]["sequence_num"][2]) for node in graph.nodes})
    ax.set_xlim(min(planes) - 0.55, max(planes) + 0.55)
    ax.set_ylim(-max(sats) - 0.55, -min(sats) + 0.55)
    ax.set_xticks(planes)
    ax.set_yticks([-sat for sat in sats])
    ax.set_yticklabels([str(sat) for sat in sats])
    ax.tick_params(axis="both", which="major", labelsize=7, length=2.5, width=0.6, pad=1.5)
    ax.grid(True, color="#E8E8E8", linewidth=0.45)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#333333")

    if previous_edges is None:
        subtitle = f"Snapshot $t_{index}$"
    else:
        subtitle = f"Snapshot $t_{index}$  (+{len(added)}, -{len(removed)})"
    ax.set_title(subtitle, fontsize=9.5, pad=4)
    return len(kept), len(added), len(removed)


def make_figure(snapshot_dir: Path, output_dir: Path, selected: list[int]) -> None:
    snapshots = read_snapshots(snapshot_dir)
    missing = [idx for idx in selected if idx not in snapshots]
    if missing:
        raise ValueError(f"Missing selected snapshots: {missing}")

    first_graph = snapshots[selected[0]]
    positions = build_grid_positions(first_graph)

    plt.rcParams.update(
        {
            "font.family": "Times New Roman",
            "mathtext.fontset": "stix",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(7.05, 5.45), dpi=300)
    axes = axes.flatten()

    summary: list[tuple[int, int, int, int]] = []
    for ax, idx in zip(axes, selected):
        graph = snapshots[idx]
        current_edges = {edge_key(edge) for edge in graph.edges}
        previous_edges = None
        if idx > 0 and (idx - 1) in snapshots:
            previous_edges = {edge_key(edge) for edge in snapshots[idx - 1].edges}
        kept, added, removed = draw_snapshot(ax, graph, positions, current_edges, previous_edges, idx)
        summary.append((idx, kept, added, removed))

    for ax in axes[len(selected) :]:
        ax.axis("off")

    fig.supxlabel("Orbital plane index", fontsize=9.5, y=0.055)
    fig.supylabel("Satellite index within an orbital plane", fontsize=9.5, x=0.055)

    legend_handles = [
        Line2D([0], [0], color="#8B8B8B", lw=1.4, label="Persistent ISL"),
        Line2D([0], [0], color="#009E73", lw=1.8, label="Newly established ISL"),
        Line2D([0], [0], color="#D55E00", lw=1.7, linestyle=(0, (3.0, 2.2)), label="Disconnected ISL"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#111111",
            markerfacecolor="white",
            markeredgewidth=0.75,
            markersize=4.7,
            lw=0,
            label="Satellite node",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=4,
        frameon=False,
        fontsize=8.5,
        handlelength=2.6,
        columnspacing=1.2,
    )

    fig.subplots_adjust(left=0.105, right=0.985, top=0.945, bottom=0.16, wspace=0.16, hspace=0.24)

    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "leo_dynamic_topology_snapshots"
    for suffix in [".svg", ".pdf", ".png"]:
        fig.savefig(base.with_suffix(suffix), bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)

    summary_path = output_dir / "leo_dynamic_topology_snapshots_summary.txt"
    lines = [
        "Dynamic topology snapshot summary",
        "Format: snapshot, persistent links, newly established links, disconnected links",
    ]
    lines.extend(f"t{idx}: kept={kept}, added={added}, removed={removed}" for idx, kept, added, removed in summary)
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, default=Path("datasets/topology/snapshots"))
    parser.add_argument("--output-dir", type=Path, default=Path("paper_figures/leo_dynamic_topology"))
    parser.add_argument("--selected", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()
    make_figure(args.snapshot_dir, args.output_dir, args.selected)


if __name__ == "__main__":
    main()
