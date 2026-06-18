from __future__ import annotations

from itertools import islice
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np

from main.utils import path_to_links

QOS_DIMS = ("latency", "reliability", "bandwidth", "computing")
SERVICE_TO_DIM = {
    "delay_sensitive": "latency",
    "reliability_sensitive": "reliability",
    "bandwidth_sensitive": "bandwidth",
    "compute_sensitive": "computing",
    "computing_sensitive": "computing",
    # Backward compatible alias for older generated CSV/config.
    "cost_sensitive": "computing",
}


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if hasattr(obj, "get"):
        return obj.get(key, default)
    return getattr(obj, key, default)


def get_service_qos_config(config: Any) -> Dict[str, Any]:
    v_setting = _get(config, "v_sim_setting", {})
    qos = _get(v_setting, "service_qos", {})
    return dict(qos) if qos is not None else {}


def is_service_qos_enabled(config: Any) -> bool:
    return bool(get_service_qos_config(config).get("enabled", False))


def build_phi(service_type: str, qos_cfg: Dict[str, Any]) -> Dict[str, float]:
    dominant_weight = float(qos_cfg.get("dominant_weight", 0.7))
    other_weight = (1.0 - dominant_weight) / (len(QOS_DIMS) - 1)
    dim = SERVICE_TO_DIM.get(service_type, "latency")
    phi = {d: other_weight for d in QOS_DIMS}
    phi[dim] = dominant_weight
    total = sum(phi.values()) or 1.0
    return {k: float(v) / total for k, v in phi.items()}


def get_thresholds(v_net: Any, qos_cfg: Dict[str, Any]) -> Dict[str, float]:
    """Return the concrete QoS thresholds carried by one SFC request.

    Chapter 4 uses per-request thresholds sampled from Table 4-1 ranges.  The
    simulator stores those concrete values on the virtual-network graph as
    ``qos_*`` attributes; this function first reads those values, then falls
    back to the YAML defaults for compatibility with older generated datasets.
    """
    service_type = v_net.graph.get("service_type", "delay_sensitive")
    default = dict(qos_cfg.get("default_thresholds", {}))
    service_thresholds = dict(qos_cfg.get("thresholds", {}).get(service_type, {}))
    default.update(service_thresholds)

    # Concrete per-request values generated from thesis Table 4-1.
    graph_key_map = {
        "max_latency": ["qos_max_latency", "max_latency"],
        "min_bandwidth": ["qos_min_bandwidth", "min_bandwidth"],
        "min_path_success": ["qos_min_path_success", "min_path_success"],
        "max_compute_delay": ["qos_max_compute_delay", "max_compute_delay"],
    }
    for key, graph_keys in graph_key_map.items():
        for graph_key in graph_keys:
            value = v_net.graph.get(graph_key)
            if value is not None:
                default[key] = float(value)
                break

    # Backward-compatible aliases.
    if "min_bw_ratio" in default and "min_bandwidth" not in default:
        default["min_bandwidth"] = default["min_bw_ratio"]
    if "max_cost_ratio" in default and "max_compute_delay" not in default:
        default["max_compute_delay"] = default["max_cost_ratio"]

    default.setdefault("max_latency", 300.0)
    default.setdefault("min_bandwidth", 15.0)
    default.setdefault("min_path_success", 0.95)
    default.setdefault("max_compute_delay", 80.0)
    return default


def _edge_data(net: Any, edge: Tuple[int, int]) -> Dict[str, Any]:
    if edge in net.edges:
        return net.edges[edge]
    rev = (edge[1], edge[0])
    if rev in net.edges:
        return net.edges[rev]
    return {}


def _solution_edges(solution: Any) -> List[Tuple[int, int]]:
    edges: List[Tuple[int, int]] = []
    for paths in solution.get("link_paths", {}).values():
        for p_link in paths or []:
            if isinstance(p_link, tuple) and len(p_link) == 2:
                edges.append(p_link)
            elif isinstance(p_link, list) and len(p_link) == 2:
                edges.append((p_link[0], p_link[1]))
    return edges


def _shortest_edges(net: Any, src: int, dst: int) -> List[Tuple[int, int]]:
    if src is None or dst is None or src == dst:
        return []
    try:
        path = next(islice(nx.shortest_simple_paths(net, src, dst, weight="ltc"), 1))
        return path_to_links(path)
    except Exception:
        return []


def compute_path_success_prob(config: Any, p_net: Any, v_net: Any, solution: Any) -> float:
    edges = _solution_edges(solution)
    if bool(_get(_get(config, "solver", {}), "include_endpoint_latency", True)):
        slots = solution.get("node_slots", {})
        if slots:
            first_v = 0
            last_v = v_net.num_nodes - 1
            edges += _shortest_edges(p_net, v_net.graph.get("src"), slots.get(first_v))
            edges += _shortest_edges(p_net, slots.get(last_v), v_net.graph.get("dst"))
    success = 1.0
    for edge in edges:
        loss = float(_edge_data(p_net, edge).get("loss", 0.0))
        loss = min(max(loss, 0.0), 0.999999)
        success *= (1.0 - loss)
    return float(success)


def compute_effective_bandwidth(p_net: Any, v_net: Any, solution: Any) -> float:
    """Request-level effective bandwidth B^e from thesis Eq. (4-14)(4-15).

    It is **not an average**.  For each virtual link, take the minimum residual
    bandwidth on its mapped physical path; for the whole SFC, take the minimum
    over all virtual links.
    """
    link_bottlenecks: List[float] = []
    for p_links in solution.get("link_paths", {}).values():
        residual_bws = []
        for p_link in p_links or []:
            data = _edge_data(p_net, p_link)
            residual_bw = float(data.get("bw", data.get("max_bw", 0.0)) or 0.0)
            residual_bws.append(residual_bw)
        if residual_bws:
            link_bottlenecks.append(min(residual_bws))
    return float(min(link_bottlenecks)) if link_bottlenecks else 0.0


def compute_bw_ratio(p_net: Any, v_net: Any, solution: Any) -> float:
    min_bandwidth = 0.0
    try:
        min_bandwidth = float(get_thresholds(v_net, get_service_qos_config({})).get("min_bandwidth", 0.0))
    except Exception:
        min_bandwidth = 0.0
    effective_bandwidth = compute_effective_bandwidth(p_net, v_net, solution)
    return float(effective_bandwidth / max(min_bandwidth, 1e-9)) if min_bandwidth > 0 else 1.0


def compute_bandwidth_margin(effective_bandwidth: float, min_bandwidth: float) -> float:
    """Bandwidth satisfaction margin from thesis Eq. (4-22)."""
    if effective_bandwidth <= 0:
        return 0.0
    return float(max(0.0, (effective_bandwidth - min_bandwidth) / effective_bandwidth))


def _critical_compute_vnode(v_net: Any) -> Any:
    candidates = []
    for v_node in v_net.nodes:
        cpu = float(v_net.nodes[v_node].get("cpu", 0.0) or 0.0)
        chi = int(float(v_net.nodes[v_node].get("compute_sensitive", 0) or 0))
        candidates.append((chi * cpu, cpu, v_node))
    if not candidates:
        return None
    return max(candidates)[2]


def compute_computing_delay(config: Any, p_net: Any, v_net: Any, solution: Any) -> float:
    """Compute processing delay from thesis Eq. (4-8)--(4-11).

    mu_i = kappa * CPU_i^res, workload_j = eta * cpu_j, and
    t_comp(j,i) = t0 + chi_j * workload_j / mu_i.
    """
    qos_cfg = get_service_qos_config(config)
    base_delay = float(qos_cfg.get("base_processing_delay", 1.0))
    service_rate_factor = float(qos_cfg.get("compute_service_rate_factor", 1.0))
    workload_factor = float(qos_cfg.get("compute_workload_factor", 1.0))
    total = 0.0
    for v_node, p_node in solution.get("node_slots", {}).items():
        if v_node not in v_net.nodes or p_node not in p_net.nodes:
            continue
        demand = float(v_net.nodes[v_node].get("cpu", 0.0) or 0.0)
        chi = int(float(v_net.nodes[v_node].get("compute_sensitive", 0) or 0))
        residual_cpu = float(p_net.nodes[p_node].get("cpu", p_net.nodes[p_node].get("max_cpu", 0.0)) or 0.0)
        mu = max(service_rate_factor * residual_cpu, 1e-9)
        total += base_delay + chi * (workload_factor * demand) / mu
    return float(total)


def compute_computing_margin(p_net: Any, v_net: Any, solution: Any) -> float:
    """Critical-computing VNF CPU carrying margin from thesis Eq. (4-24)--(4-26)."""
    critical = _critical_compute_vnode(v_net)
    node_slots = solution.get("node_slots", {})
    if critical is None or critical not in node_slots:
        return 0.0
    p_node = node_slots[critical]
    if p_node not in p_net.nodes:
        return 0.0
    demand_on_node = 0.0
    for v_node, mapped_p in node_slots.items():
        if mapped_p == p_node and v_node in v_net.nodes:
            demand_on_node += float(v_net.nodes[v_node].get("cpu", 0.0) or 0.0)
    post_cpu = float(p_net.nodes[p_node].get("cpu", 0.0) or 0.0)
    before_cpu = post_cpu + demand_on_node
    return float(max(0.0, min(1.0, post_cpu / max(before_cpu, 1e-9))))


def compute_service_qos_metrics(config: Any, p_net: Any, v_net: Any, solution: Any) -> Dict[str, Any]:
    qos_cfg = get_service_qos_config(config)
    service_type = v_net.graph.get("service_type", qos_cfg.get("default_service_type", "delay_sensitive"))
    if all(v_net.graph.get(f"phi_{dim}") is not None for dim in QOS_DIMS):
        phi = {dim: float(v_net.graph.get(f"phi_{dim}")) for dim in QOS_DIMS}
    else:
        phi = build_phi(service_type, qos_cfg)
    thresholds = get_thresholds(v_net, qos_cfg)

    # `total_latency` produced by the placement/routing code is the communication
    # delay: src -> first VNF + SFC links + last VNF -> dst.  In Chapter 4, the
    # end-to-end service delay should also include VNF processing/computing
    # delay, so we keep the communication part and use the combined value for
    # QoS latency scoring/threshold checking.
    communication_latency = float(solution.get("communication_latency", solution.get("total_latency", v_net.graph.get("max_latency", 0.0))) or 0.0)
    path_success = compute_path_success_prob(config, p_net, v_net, solution)
    effective_bandwidth = compute_effective_bandwidth(p_net, v_net, solution)
    min_bandwidth = float(thresholds["min_bandwidth"])
    bw_ratio = float(effective_bandwidth / max(min_bandwidth, 1e-9))
    bandwidth_margin = compute_bandwidth_margin(effective_bandwidth, min_bandwidth)
    compute_delay = compute_computing_delay(config, p_net, v_net, solution)
    computing_margin = compute_computing_margin(p_net, v_net, solution)
    latency = communication_latency + compute_delay

    latency_score = max(0.0, min(1.0, 1.0 - latency / max(float(thresholds["max_latency"]), 1e-9)))
    reliability_score = max(0.0, min(1.0, (path_success - float(thresholds["min_path_success"])) / max(1.0 - float(thresholds["min_path_success"]), 1e-9)))
    bandwidth_score = max(0.0, min(1.0, bandwidth_margin))
    computing_score = max(0.0, min(1.0, computing_margin))

    scores = {
        "latency": latency_score,
        "reliability": reliability_score,
        "bandwidth": bandwidth_score,
        "computing": computing_score,
    }
    qos_score = sum(float(phi[d]) * scores[d] for d in QOS_DIMS)

    # Chapter 4 is service-dominant QoS optimization.  By default, admission
    # uses the dominant QoS dimension of the current service as the hard
    # threshold; the other dimensions still contribute to qos_score/reward.
    # Set service_qos.enforce_mode=all if all four dimensions must be hard
    # constraints simultaneously.
    dim_satisfied = {
        "latency": latency <= float(thresholds["max_latency"]),
        "reliability": path_success >= float(thresholds["min_path_success"]),
        "bandwidth": effective_bandwidth >= float(thresholds["min_bandwidth"]),
        "computing": compute_delay <= float(thresholds["max_compute_delay"]),
    }
    enforce_mode = str(qos_cfg.get("enforce_mode", "dominant")).lower()
    dominant_dim = SERVICE_TO_DIM.get(service_type, "latency")
    if enforce_mode == "all":
        qos_satisfied = all(dim_satisfied.values())
    else:
        qos_satisfied = bool(dim_satisfied.get(dominant_dim, True))
    metrics = {
        "service_type": service_type,
        "path_success_prob": path_success,
        "bw_ratio": bw_ratio,
        "effective_bandwidth": effective_bandwidth,
        "bandwidth_margin": bandwidth_margin,
        "compute_delay": compute_delay,
        "communication_latency": communication_latency,
        "service_total_latency": latency,
        "total_latency": latency,
        "computing_margin": computing_margin,
        "qos_score": float(qos_score),
        "qos_satisfied": bool(qos_satisfied),
        "qos_enforce_mode": enforce_mode,
        "qos_dominant_dim": dominant_dim,
    }
    for dim, value in dim_satisfied.items():
        metrics[f"qos_{dim}_satisfied"] = bool(value)
    for dim in QOS_DIMS:
        metrics[f"phi_{dim}"] = float(phi[dim])
        metrics[f"qos_{dim}_score"] = float(scores[dim])
    for key, value in thresholds.items():
        metrics[f"qos_{key}"] = value
    return metrics


def apply_service_qos_to_solution(config: Any, p_net: Any, v_net: Any, solution: Any) -> None:
    if not is_service_qos_enabled(config):
        return
    metrics = compute_service_qos_metrics(config, p_net, v_net, solution)
    solution.update(metrics)
    if solution.get("result", False) and not metrics["qos_satisfied"] and bool(get_service_qos_config(config).get("enforce_thresholds", False)):
        solution.update({
            "result": False,
            "route_result": False,
            "description": "service QoS threshold violation",
        })
