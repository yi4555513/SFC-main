from datetime import timedelta
from pathlib import Path

import networkx as nx
import numpy as np
from skyfield.api import EarthSatellite, load

LIGHT_SPEED_KM_PER_S = 3e5


class SatelliteTracker:
    def __init__(self, tle_filepath, max_satellites=None):
        tle_lines = Path(tle_filepath).read_text(encoding="utf-8").splitlines()
        if max_satellites is not None:
            tle_lines = tle_lines[: int(max_satellites) * 3]
        self.satellites = [
            EarthSatellite(tle_lines[i + 1], tle_lines[i + 2], tle_lines[i])
            for i in range(0, len(tle_lines), 3)
        ]

    def generate_satellite_dict(self, time):
        sat_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            eci_position = geocentric.position.km
            subpoint = geocentric.subpoint()
            orbit_altitude, orbit_number, sat_number = [int(s) for s in sat.name.split("_")[1:]]
            sat_dict[sat.name] = [
                eci_position,
                orbit_altitude,
                orbit_number,
                sat_number,
                subpoint.latitude.degrees,
                subpoint.longitude.degrees,
                subpoint.elevation.km,
            ]
        return sat_dict

    def get_max_orbit_number(self):
        return max(int(sat.name.split("_")[2]) for sat in self.satellites)

    def get_max_satellite_number(self):
        return max(int(sat.name.split("_")[3]) for sat in self.satellites)


class SatelliteGraph:
    def __init__(self, p_net_setting):
        self.p_net_setting = p_net_setting
        self.fixed_node_attrs = {}
        self.fixed_edge_attrs = {}
        self.fixed_edges = None

    @staticmethod
    def _distance(pos1, pos2):
        return float(sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5)

    @staticmethod
    def _attr_range(setting, section, name):
        for attr in setting.get(section, []):
            if attr.get("name") == name:
                return attr.get("low"), attr.get("high")
        raise KeyError(f"Cannot find {name!r} in {section}.")

    def build_graph_with_fixed_edges(
        self,
        satellite_tracker,
        time,
        pole=False,
        distance_threshold=None,
        snapshot_index=0,
    ):
        satellite_dict = satellite_tracker.generate_satellite_dict(time)
        graph = nx.Graph()
        graph.add_nodes_from(satellite_dict.keys())

        low_cpu, high_cpu = self._attr_range(self.p_net_setting, "node_attrs_setting", "cpu")
        low_ram, high_ram = self._attr_range(self.p_net_setting, "node_attrs_setting", "ram")
        low_bw, high_bw = self._attr_range(self.p_net_setting, "link_attrs_setting", "bw")

        if snapshot_index == 0:
            self.fixed_node_attrs = {}
            for sat_name in satellite_dict:
                cpu = int(np.random.randint(low_cpu, high_cpu))
                ram = int(np.random.randint(low_ram, high_ram))
                self.fixed_node_attrs[sat_name] = {
                    "cpu": cpu,
                    "max_cpu": cpu,
                    "ram": ram,
                    "max_ram": ram,
                }

        for sat_name, position in satellite_dict.items():
            graph.nodes[sat_name]["pos"] = position[0].tolist()
            graph.nodes[sat_name]["sequence_num"] = position[1:4]
            graph.nodes[sat_name]["pos_0"] = position[4:]
            graph.nodes[sat_name].update(self.fixed_node_attrs[sat_name])

        max_orbit_number = satellite_tracker.get_max_orbit_number()
        max_satellite_number = satellite_tracker.get_max_satellite_number()

        if snapshot_index == 0:
            self.fixed_edges = []
            self.fixed_edge_attrs = {}
            for sat_name, sat_data in satellite_dict.items():
                same_orbit_neighbors = [
                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number}",
                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{(sat_data[3] % max_satellite_number) + 1}",
                ]
                self._maybe_add_edges(graph, sat_name, same_orbit_neighbors, low_bw, high_bw, distance_threshold)

                next_orbit_number = sat_data[2] + 1 if pole else (sat_data[2] % max_orbit_number) + 1
                if not pole or next_orbit_number <= max_orbit_number:
                    next_sat_num = sat_data[3] if next_orbit_number % 2 == 0 else (
                        sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number
                    )
                    next_sat = f"Satellite_{sat_data[1]}_{next_orbit_number}_{next_sat_num}"
                    if next_sat in graph.nodes:
                        if pole and (
                            abs(graph.nodes[sat_name]["pos"][2]) >= 6000
                            or abs(graph.nodes[next_sat]["pos"][2]) >= 6000
                        ):
                            continue
                        self._maybe_add_edges(graph, sat_name, [next_sat], low_bw, high_bw, distance_threshold)

        for sat_name, neighbor in self.fixed_edges:
            distance = self._distance(graph.nodes[sat_name]["pos"], graph.nodes[neighbor]["pos"])
            if distance_threshold is None or distance <= distance_threshold:
                ltc = 1000 * distance / LIGHT_SPEED_KM_PER_S
                graph.add_edge(
                    sat_name,
                    neighbor,
                    pos_a=graph.nodes[sat_name]["pos"],
                    pos_b=graph.nodes[neighbor]["pos"],
                    distance=distance,
                    ltc=float(ltc),
                    **self.fixed_edge_attrs[(sat_name, neighbor)],
                )

        graph.graph["node_attrs_setting"] = [
            {"name": "cpu", "type": "resource", "owner": "node", "distribution": "uniform", "dtype": "int", "generative": True, "high": high_cpu, "low": low_cpu},
            {"name": "max_cpu", "type": "extrema", "owner": "node", "originator": "cpu"},
            {"name": "ram", "type": "resource", "owner": "node", "distribution": "uniform", "dtype": "int", "generative": True, "high": high_ram, "low": low_ram},
            {"name": "max_ram", "type": "extrema", "owner": "node", "originator": "ram"},
            {"name": "pos", "type": "attribute", "owner": "node", "dtype": "list", "generative": False},
            {"name": "sequence_num", "type": "attribute", "owner": "node", "dtype": "list", "generative": False},
            {"name": "pos_0", "type": "attribute", "owner": "node", "dtype": "list", "generative": False},
        ]
        graph.graph["link_attrs_setting"] = [
            {"name": "bw", "type": "resource", "owner": "link", "distribution": "uniform", "dtype": "int", "generative": True, "high": high_bw, "low": low_bw},
            {"name": "max_bw", "type": "extrema", "owner": "link", "originator": "bw"},
            {"name": "distance", "type": "attribute", "owner": "link", "dtype": "float", "generative": False},
            {"name": "ltc", "type": "latency", "owner": "link", "distribution": "customized", "generative": False, "max": 10.0, "min": 0.0},
            {"name": "pos_a", "type": "attribute", "owner": "link", "dtype": "list", "generative": False},
            {"name": "pos_b", "type": "attribute", "owner": "link", "dtype": "list", "generative": False},
        ]
        return graph

    def _maybe_add_edges(self, graph, sat_name, neighbors, low_bw, high_bw, distance_threshold):
        for neighbor in neighbors:
            if neighbor not in graph.nodes:
                continue
            distance = self._distance(graph.nodes[sat_name]["pos"], graph.nodes[neighbor]["pos"])
            if distance_threshold is not None and distance > distance_threshold:
                continue
            edge = (sat_name, neighbor)
            if edge in self.fixed_edge_attrs:
                continue
            self.fixed_edges.append(edge)
            bw = int(np.random.randint(low_bw, high_bw))
            self.fixed_edge_attrs[edge] = {"bw": bw, "max_bw": bw}


def custom_stringizer(value):
    if isinstance(value, (list, np.ndarray)):
        return f"[{','.join(str(float(x)) for x in value)}]"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return str(float(value))
    return str(value)


def generate_multi_snapshot_gml(
    tle_filepath,
    output_dir,
    p_net_setting,
    num_snapshots=10,
    time_interval_seconds=10,
    max_satellites=None,
    distance_threshold=10000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_snapshot in output_dir.glob("*.gml"):
        old_snapshot.unlink()

    ts = load.timescale()
    start_time = ts.utc(2023, 5, 1)
    tracker = SatelliteTracker(tle_filepath, max_satellites=max_satellites)
    graph_builder = SatelliteGraph(p_net_setting)

    for i in range(int(num_snapshots)):
        current_time = ts.utc(start_time.utc_datetime() + timedelta(seconds=i * time_interval_seconds))
        graph = graph_builder.build_graph_with_fixed_edges(
            tracker,
            current_time,
            pole=True,
            distance_threshold=distance_threshold,
            snapshot_index=i,
        )
        output_gml = output_dir / f"satellite_graph_t{i}.gml"
        nx.write_gml(graph, output_gml, stringizer=custom_stringizer)
        print(f"Generated snapshot {i}: {output_gml}")
