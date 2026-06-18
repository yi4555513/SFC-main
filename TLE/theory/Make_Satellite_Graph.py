from skyfield.api import EarthSatellite

class SatelliteTracker:
    def __init__(self, tle_filepath):
        with open(tle_filepath) as f:
            tle_data = f.read()

        tle_lines = tle_data.splitlines()
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
            sat_dict[sat_name] = [eci_position,orbit_altitude , orbit_number, sat_number,lat,lon,alt]
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

            sat_LLA_dict[sat.name]={"latitude": lat, "longitude": lon, "altitude": alt}
        return sat_LLA_dict

import networkx as nx

class SatelliteGraph:
    def __init__(self):
        pass

    def _distance(self, pos1, pos2):
        return sum((a - b) ** 2 for a, b in zip(pos1, pos2)) ** 0.5

    def build_graph_with_fixed_edges(self, satellite_tracker, time, pole=False):
        satellite_dict = satellite_tracker.generate_satellite_dict(time)
        graph = nx.Graph()
        graph.add_nodes_from(satellite_dict.keys())
        for sat_name, position in satellite_dict.items():
            graph.nodes[sat_name]['pos'] = position[0]
            graph.nodes[sat_name]['sequence_num'] = position[1:4]
            graph.nodes[sat_name]['pos_0'] = position[4:]
        max_orbit_number = satellite_tracker.get_max_orbit_number()
        max_satellite_number = satellite_tracker.get_max_satellite_number()

        for sat_name, sat_data in satellite_dict.items():
            same_orbit_neighbors = [f"Satellite_{sat_data[1]}_{sat_data[2]}_{sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number}",
                                    f"Satellite_{sat_data[1]}_{sat_data[2]}_{(sat_data[3] % max_satellite_number) + 1}"]
            for neighbor in same_orbit_neighbors:
                if neighbor in satellite_dict:
                    graph.add_edge(sat_name, neighbor,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[neighbor]['pos'])

        for sat_name, sat_data in satellite_dict.items():
            if pole:
                next_orbit_number = sat_data[2] + 1
            else:
                next_orbit_number = (sat_data[2] % max_orbit_number) + 1
            if not pole:
                next_orbit_satellite =f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number%2==0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                graph.add_edge(sat_name, next_orbit_satellite,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[next_orbit_satellite]['pos'])

            elif next_orbit_number<= max_orbit_number:
                next_orbit_satellite = f"Satellite_{sat_data[1]}_{next_orbit_number}_{sat_data[3] if next_orbit_number%2==0 else (sat_data[3] - 1 if sat_data[3] != 1 else max_satellite_number)}"
                if abs(graph.nodes[sat_name]['pos'][2])<6000 and abs(graph.nodes[next_orbit_satellite]['pos'][2])<6000:
                    graph.add_edge(sat_name, next_orbit_satellite,pos_a=graph.nodes[sat_name]['pos'], pos_b=graph.nodes[next_orbit_satellite]['pos'])

        return graph

from skyfield.api import load

def get_adjacency_matrix(graph):
    """返回邻接矩阵表示"""
    # 获取按名称排序的节点列表
    nodes = sorted(graph.nodes())
    n = len(nodes)

    # 创建节点名称到索引的映射
    node_to_index = {node: i for i, node in enumerate(nodes)}

    # 创建邻接矩阵
    adj_matrix = [[0 for _ in range(n)] for _ in range(n)]

    # 填充邻接矩阵
    for u, v in graph.edges():
        i, j = node_to_index[u], node_to_index[v]
        adj_matrix[i][j] = 1
        adj_matrix[j][i] = 1  # 无向图

    return adj_matrix, nodes


def get_adjacency_list(graph):
    """返回邻接表表示"""
    adj_list = {}
    for node in graph.nodes():
        adj_list[node] = list(graph.neighbors(node))

    return adj_list


if __name__ == '__main__':
    # 创建时间对象
    ts = load.timescale()
    time = ts.utc(2025, 7, 2)  # 使用当前日期

    # 加载卫星TLE数据
    tle_filepath = r"TLE\theory\Satellite_Data\60Degree_500_12x24_tles_1.txt"  # 替换为实际TLE文件路径
    tracker = SatelliteTracker(tle_filepath)

    # 创建卫星连接图
    satellite_graph = SatelliteGraph()
    graph = satellite_graph.build_graph_with_fixed_edges(tracker, time)

    # 获取邻接矩阵
    adj_matrix, node_names = get_adjacency_matrix(graph)
    print(f"邻接矩阵大小: {len(adj_matrix)}x{len(adj_matrix[0])}")

    # 获取邻接表
    adj_list = get_adjacency_list(graph)
    print(f"节点数量: {len(adj_list)}")
    print(f"第一个节点的连接: {list(adj_list.items())[0]}")
