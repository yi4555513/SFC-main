
from skyfield.api import EarthSatellite, load
import networkx as nx
import numpy as np
import os
from datetime import timedelta

# 定义光速 (km/s)
LIGHT_SPEED = 3e5  # 300,000 km/s


# SatelliteTracker 类（保持不变）
class SatelliteTracker:
    def __init__(self, tle_filepath, max_satellites=None):
        with open(tle_filepath) as f:
            tle_data = f.read()
        tle_lines = tle_data.splitlines()
        if max_satellites is not None:
            tle_lines = tle_lines[:max_satellites * 3]
        self.satellites = [EarthSatellite(tle_lines[i + 1], tle_lines[i + 2], tle_lines[i]) for i in
                           range(0, len(tle_lines), 3)]

    def generate_satellite_dict(self, time):
        sat_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            eci_position = geocentric.position.km
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km
            sat_name = sat.name
            orbit_altitude, orbit_number, sat_number = [int(s) for s in sat_name.split('_')[1:]]
            sat_dict[sat_name] = [eci_position, orbit_altitude, orbit_number, sat_number, lat, lon, alt]
        return sat_dict

    def get_max_orbit_number(self):
        max_orbit_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            orbit_number = int(sat_name.split('_')[2])
            max_orbit_number = max(max_orbit_number, orbit_number)
        return max_orbit_number

    def get_max_satellite_number(self):
        max_satellite_number = 0
        for sat in self.satellites:
            sat_name = sat.name
            satellite_number = int(sat_name.split('_')[3])
            max_satellite_number = max(max_satellite_number, satellite_number)
        return max_satellite_number

    def generate_satellite_LLA_dict(self, time):
        sat_LLA_dict = {}
        for sat in self.satellites:
            geocentric = sat.at(time)
            subpoint = geocentric.subpoint()
            lat = subpoint.latitude.degrees
            lon = subpoint.longitude.degrees
            alt = subpoint.elevation.km
            sat_LLA_dict[sat.name] = {"latitude": lat, "longitude": lon, "altitude": alt}
        return sat_LLA_dict


# 修改后的 SatelliteGraph 类
class SatelliteGraph:
    def __init__(self, p_net_setting=None, v_sim_setting=None):
        self.fixed_node_attrs = {}  # fixed node attributes
        self.fixed_edge_attrs = {}  # fixed edge attributes
        self.fixed_edges = None  # fixed edge set
        self.p_net_setting = p_net_setting
        self.v_sim_setting = v_sim_setting

    def _distance(self, pos1, pos2):
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def build_graph_with_fixed_edges(self, satellite_tracker, time, pole=False, distance_threshold=None,
                                     snapshot_index=0):
        """
        构建卫星网络图，节点属性和边带宽固定，distance 和 ltc 随时间变化。

        Args:
            satellite_tracker: SatelliteTracker 实例
            time: 当前时间 (Skyfield Time 对象)
            pole: 是否包含极地轨道
            distance_threshold: 最大连接距离（km）
            snapshot_index: 当前快照索引（用于决定是否初始化固定属性）

        Returns:
            nx.Graph: 卫星网络图
        """
        satellite_dict = satellite_tracker.generate_satellite_dict(time)
        graph = nx.Graph()
        graph.add_nodes_from(satellite_dict.keys())

        if self.p_net_setting is None or self.v_sim_setting is None:
            import yaml
            with open(r'settings\p_net_setting\satellite_snap.yaml', 'r') as f:
                config = yaml.safe_load(f)
            with open(r'settings\v_sim_setting\satellite.yaml', 'r', encoding='utf-8') as f:
                config2 = yaml.safe_load(f)
        else:
            config = self.p_net_setting
            config2 = self.v_sim_setting
        num_snapshots = config['topology']['num_snapshots'] # 总快照数量
        snapshot_duration = config2['num_v_nets'] / (
                1000 * num_snapshots * config2['arrival_rate']['lam'])  # 单位 s
        low_cpu = None
        for attribute in config['node_attrs_setting']:
            if attribute['name'] == 'cpu':
                low_cpu = attribute['low']
                break
        high_cpu = None
        for attribute in config['node_attrs_setting']:
            if attribute['name'] == 'cpu':
                high_cpu = attribute['high']
                break
        low_ram = None
        for attribute in config['node_attrs_setting']:
            if attribute['name'] == 'ram':
                low_ram = attribute['low']
                break
        high_ram = None
        for attribute in config['node_attrs_setting']:
            if attribute['name'] == 'ram':
                high_ram = attribute['high']
                break
        low_bw = None
        for attribute in config['link_attrs_setting']:
            if attribute['name'] == 'bw':
                low_bw = attribute['low']
                break
        high_bw = None
        for attribute in config['link_attrs_setting']:
            if attribute['name'] == 'bw':
                high_bw = attribute['high']
                break
        # 初始化或重用节点属性
        if snapshot_index == 0:
            self.fixed_node_attrs = {}
            for sat_name, position in satellite_dict.items():
                cpu = int(np.random.randint(low_cpu, high_cpu))  # 固定 cpu
                ram = int(np.random.randint(low_ram, high_ram))  # 固定 ram
                self.fixed_node_attrs[sat_name] = {
                    'cpu': cpu,
                    'max_cpu': cpu,
                    'ram': ram,
                    'max_ram': ram
                }

        # 设置节点属性
        for sat_name, position in satellite_dict.items():
            graph.nodes[sat_name]['pos'] = position[0].tolist()  # ECI 位置（随时间变化）
            graph.nodes[sat_name]['sequence_num'] = position[1:4]  # 轨道高度、轨道编号、卫星编号
            graph.nodes[sat_name]['pos_0'] = position[4:]  # 纬度、经度、高度
            # 使用固定属性
            graph.nodes[sat_name].update(self.fixed_node_attrs[sat_name])

        max_orbit_number = satellite_tracker.get_max_orbit_number()
        max_satellite_number = satellite_tracker.get_max_satellite_number()

        # 初始化或重用边集和边属性
        if snapshot_index == 0:
            self.fixed_edges = []
            self.fixed_edge_attrs = {}
            for sat_name, sat_data in satellite_dict.items():
                # 同轨道邻居
                same_orbit_neighbors = [
                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number}",
                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{(sat_data[3] % max_satellite_number) + 1}"
                ]
                for neighbor in same_orbit_neighbors:
                    if neighbor in graph.nodes:
                        distance = self._distance(graph.nodes[sat_name]['pos'], graph.nodes[neighbor]['pos'])
                        if distance_threshold is None or distance <= distance_threshold:
                            self.fixed_edges.append((sat_name, neighbor))
                            bw = int(np.random.randint(low_bw, high_bw))  # 固定带宽
                            self.fixed_edge_attrs[(sat_name, neighbor)] = {'bw': bw, 'max_bw': bw}

                # 跨轨道邻居
                if pole:
                    next_orbit_number = sat_data[2] + 1
                else:
                    next_orbit_number = (sat_data[2] % max_orbit_number) + 1
                if not pole:
                    next_orbit_satellite = f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number % 2 == 0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                    if next_orbit_satellite in graph.nodes:
                        distance = self._distance(graph.nodes[sat_name]['pos'],
                                                  graph.nodes[next_orbit_satellite]['pos'])
                        if distance_threshold is None or distance <= distance_threshold:
                            self.fixed_edges.append((sat_name, next_orbit_satellite))
                            bw = int(np.random.randint(low_bw, high_bw))
                            self.fixed_edge_attrs[(sat_name, next_orbit_satellite)] = {'bw': bw, 'max_bw': bw}
                elif next_orbit_number <= max_orbit_number:
                    next_orbit_satellite = f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number % 2 == 0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                    if next_orbit_satellite in graph.nodes and abs(graph.nodes[sat_name]['pos'][2]) < 6000 and abs(
                            graph.nodes[next_orbit_satellite]['pos'][2]) < 6000:
                        distance = self._distance(graph.nodes[sat_name]['pos'],
                                                  graph.nodes[next_orbit_satellite]['pos'])
                        if distance_threshold is None or distance <= distance_threshold:
                            self.fixed_edges.append((sat_name, next_orbit_satellite))
                            bw = int(np.random.randint(low_bw, high_bw))
                            self.fixed_edge_attrs[(sat_name, next_orbit_satellite)] = {'bw': bw, 'max_bw': bw}

        # 添加固定边集并更新 distance 和 ltc
        for edge in self.fixed_edges:
            sat_name, neighbor = edge
            distance = self._distance(graph.nodes[sat_name]['pos'], graph.nodes[neighbor]['pos'])
            if distance_threshold is None or distance <= distance_threshold:
                ltc = 1000 * distance / LIGHT_SPEED  # 时延（毫秒）
                graph.add_edge(sat_name, neighbor,
                               pos_a=graph.nodes[sat_name]['pos'],
                               pos_b=graph.nodes[neighbor]['pos'],
                               distance=float(distance),
                               ltc=float(ltc),
                               **self.fixed_edge_attrs[edge])

        # 设置图属性（兼容 ViRNE）
        graph.graph["node_attrs_setting"] = [
            {"name": "cpu", "type": "resource", "owner": "node", "distribution": "uniform", "dtype": "int",
             "generative": True, "high": high_cpu, "low": low_cpu},
            {"name": "max_cpu", "type": "extrema", "owner": "node", "originator": "cpu"},
            {"name": "ram", "type": "resource", "owner": "node", "distribution": "uniform", "dtype": "int",
             "generative": True, "high": high_ram, "low": low_ram},
            {"name": "max_ram", "type": "extrema", "owner": "node", "originator": "ram"},
            {"name": "pos", "type": "attribute", "owner": "node", "dtype": "list", "generative": False},
            {"name": "sequence_num", "type": "attribute", "owner": "node", "dtype": "list", "generative": False},
            {"name": "pos_0", "type": "attribute", "owner": "node", "dtype": "list", "generative": False}
        ]
        graph.graph["link_attrs_setting"] = [
            {"name": "bw", "type": "resource", "owner": "link", "distribution": "uniform", "dtype": "int",
             "generative": True, "high": high_bw, "low": low_bw},
            {"name": "max_bw", "type": "extrema", "owner": "link", "originator": "bw"},
            {"name": "distance", "type": "attribute", "owner": "link", "dtype": "float", "generative": False},
            {"name": "ltc", "type": "constraint", "owner": "link", "checking_level": "path", "dtype": "float",
             "generative": False},
            {"name": "pos_a", "type": "attribute", "owner": "link", "dtype": "list", "generative": False},
            {"name": "pos_b", "type": "attribute", "owner": "link", "dtype": "list", "generative": False}
        ]
        return graph


# 自定义 GML 序列化函数
def custom_stringizer(value):
    if isinstance(value, (list, np.ndarray)):
        return f"[{','.join(str(float(x)) for x in value)}]"
    elif isinstance(value, (int, np.integer)):
        return str(int(value))
    elif isinstance(value, (float, np.floating)):
        return str(float(value))
    return str(value)


# 修改后的生成多快照函数
def generate_multi_snapshot_gml(tle_filepath, output_dir="snapshots", num_snapshots=10, time_interval_seconds=10,
                                max_satellites=None, p_net_setting=None, v_sim_setting=None,
                                distance_threshold=10000):
    """
    生成多个时间快照的 GML 文件，节点属性和链路带宽固定，distance 和 ltc 随时间变化。

    Args:
        tle_filepath: TLE 文件路径
        output_dir: 输出 GML 文件目录
        num_snapshots: 快照数量
        time_interval_minutes: 快照时间间隔（分钟）
        max_satellites: 最大卫星数量（可选）
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = load.timescale()
    start_time = ts.utc(2023, 5, 1)
    tracker = SatelliteTracker(tle_filepath, max_satellites=max_satellites)
    graph_builder = SatelliteGraph(p_net_setting=p_net_setting, v_sim_setting=v_sim_setting)

    for i in range(num_snapshots):
        current_time = ts.utc(start_time.utc_datetime() + timedelta(seconds=i * time_interval_seconds))
        graph = graph_builder.build_graph_with_fixed_edges(tracker, current_time, pole=True,
                                                           distance_threshold=distance_threshold, snapshot_index=i)

        # 调试信息
        # print(f"t{i} sample node attributes:", graph.nodes[list(graph.nodes)[0]])
        # print(f"t{i} sample edge attributes:", graph.edges[list(graph.edges)[0]] if graph.edges else "No edges")
        # print(f"t{i} node_attrs_setting:", graph.graph["node_attrs_setting"])
        # print(f"t{i} link_attrs_setting:", graph.graph["link_attrs_setting"])

        output_gml = os.path.join(output_dir, f"satellite_graph_t{i}.gml")
        nx.write_gml(graph, output_gml, stringizer=custom_stringizer)

        G = nx.read_gml(output_gml, label='id')
        # print(f"t{i} loaded node attributes:", G.nodes[list(G.nodes)[0]])
        # print(f"t{i} loaded edge attributes:", G.edges[list(G.edges)[0]] if G.edges else "No edges")
        # print(f"Generated snapshot {i} at {current_time.utc_iso()} to {output_gml}")

if __name__ == "__main__":
    import yaml
    with open(r'settings\p_net_setting\satellite_snap.yaml', 'r',encoding='utf-8') as f:
        config = yaml.safe_load(f)
    planes = config['topology']['planes']
    nums_per_plane = config['topology']['nums_per_plane']
    print(f"planes:{planes}")
    print(f"nums_per_plane:{nums_per_plane}")
    tle_filepath = fr"TLE\theory\Satellite_Data/60Degree_500_{planes}x{nums_per_plane}_tles_1.txt"

    with open(r'settings\v_sim_setting\satellite.yaml', 'r',
              encoding='utf-8') as f:
        config2 = yaml.safe_load(f)
    num_snapshots = config['topology']['num_snapshots']  # 总快照数量
    snapshot_duration = config2['num_v_nets'] / (
            1000 * num_snapshots * config2['arrival_rate']['lam'])  # 单位 s
    generate_multi_snapshot_gml(tle_filepath, output_dir="snapshots", num_snapshots=10, time_interval_seconds=snapshot_duration,
                                max_satellites=1000)
