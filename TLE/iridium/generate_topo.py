import numpy as np
from skyfield.api import load, EarthSatellite
import networkx as nx
import os
import json
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import logging
import time
from datetime import datetime
# 在代码顶部导入模块后添加以下代码
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import platform

# 设置中文字体支持
if platform.system() == 'Windows':
    # Windows系统
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体为黑体
elif platform.system() == 'Darwin':
    # macOS系统
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'STHeiti']
else:
    # Linux系统
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei']  # 文泉驿微米黑

# 解决负号显示问题
plt.rcParams['axes.unicode_minus'] = False

# 解决KMeans警告
os.environ["OMP_NUM_THREADS"] = "1"  # 设置OMP线程数为1，避免内存泄漏

# 设置日志
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"iridium_simulation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def load_and_parse_tle(tle_file_path):
    """加载并解析TLE文件"""
    logger.info(f"开始从 {tle_file_path} 加载TLE数据...")

    with open(tle_file_path, 'r') as f:
        tle_data = f.read()

    lines = tle_data.strip().split('\n')
    all_satellites = []
    i = 0

    while i < len(lines):
        if i + 2 < len(lines):
            name = lines[i].strip()
            line1 = lines[i + 1].strip()
            line2 = lines[i + 2].strip()

            try:
                sat = EarthSatellite(line1, line2, name)

                # 提取轨道参数
                inclination = float(line2[8:16])
                raan = float(line2[17:25])  # 升交点赤经

                all_satellites.append({
                    'name': name,
                    'sat': sat,
                    'line1': line1,
                    'line2': line2,
                    'inclination': inclination,
                    'raan': raan,
                    'orbit_plane': None  # 稍后通过聚类确定
                })

            except Exception as e:
                logger.warning(f"跳过无效TLE: {name}, 错误: {e}")
        i += 3

    logger.info(f"总共解析了 {len(all_satellites)} 颗卫星的TLE数据")
    return all_satellites


def cluster_orbit_planes(satellites, n_clusters=6):
    """使用RAAN和倾角更准确地分组卫星"""
    # 提取倾角和RAAN
    features = np.array([[sat['inclination'], sat['raan']] for sat in satellites])

    # 处理RAAN在0-360范围内的循环性质
    if np.max(features[:, 1]) - np.min(features[:, 1]) > 180:
        features[:, 1] = np.where(features[:, 1] < 180, features[:, 1] + 360, features[:, 1])

    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)

    # 将聚类标签映射回卫星
    for i, sat in enumerate(satellites):
        sat['orbit_plane'] = f"Plane {labels[i] + 1}"

    return satellites


def select_active_satellites(satellites, satellites_per_plane=11):
    """选择活跃的66颗卫星（每个轨道面11颗）"""
    logger.info(f"选择活跃卫星，每个轨道面{satellites_per_plane}颗...")

    # 按轨道面分组
    planes = {}
    for sat in satellites:
        plane = sat['orbit_plane']
        if plane not in planes:
            planes[plane] = []
        planes[plane].append(sat)

    # 选择每个轨道面的卫星
    active_satellites = []
    for plane_name in sorted(planes.keys()):
        plane_sats = planes[plane_name]

        # 按RAAN排序，以确保选择分布均匀的卫星
        sorted_sats = sorted(plane_sats, key=lambda x: x['raan'])

        # 选择卫星
        selected = sorted_sats[:min(satellites_per_plane, len(sorted_sats))]
        active_satellites.extend(selected)
        logger.info(f"从 {plane_name} 选择了 {len(selected)} 颗卫星")

    # 限制为66颗
    active_satellites = active_satellites[:66]
    logger.info(f"最终选择了 {len(active_satellites)} 颗活跃卫星用于仿真")

    return active_satellites


def calculate_satellite_positions(satellites, time):
    """计算指定时间点的卫星位置"""
    positions = []
    for sat in satellites:
        try:
            geocentric = sat.at(time)
            x, y, z = geocentric.position.km
            positions.append((x, y, z))
        except Exception as e:
            logger.error(f"计算位置错误 {sat.name}: {e}")
            positions.append((0, 0, 0))  # 添加默认位置
    return positions


def order_satellites_in_plane(satellites_in_plane, positions):
    """根据卫星在轨道上的相对位置排序"""
    # 提取这些卫星的位置
    sat_indices = [s['index'] for s in satellites_in_plane]
    sat_positions = [positions[i] for i in sat_indices]

    # 计算轨道平面的法向量
    if len(sat_positions) < 3:
        return sat_indices  # 如果卫星不足3个，无法确定平面

    # 选择3个点确定平面
    p1, p2, p3 = np.array(sat_positions[0]), np.array(sat_positions[1]), np.array(sat_positions[2])
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)

    # 计算地球中心到轨道平面的垂线方向
    # 地球中心在原点(0,0,0)

    # 选择一个参考卫星
    ref_sat = np.array(sat_positions[0])

    # 计算每个卫星相对于参考卫星的角度
    angles = []
    for pos in sat_positions:
        pos = np.array(pos)
        # 投影到垂直于法向量的平面
        projected = pos - np.dot(pos, normal) * normal
        # 计算与参考方向的角度
        angle = np.arctan2(np.linalg.norm(np.cross(ref_sat, projected)), np.dot(ref_sat, projected))
        # 确定角度符号
        if np.dot(np.cross(ref_sat, projected), normal) < 0:
            angle = 2 * np.pi - angle
        angles.append(angle)

    # 根据角度排序
    sorted_indices = [sat_indices[i] for i in np.argsort(angles)]
    return sorted_indices


def create_network_topology(active_satellites, positions, max_link_distance=4000):
    G = nx.Graph()

    # 添加节点
    for i, sat_info in enumerate(active_satellites):
        G.add_node(i, name=sat_info['name'].strip(),
                   position=positions[i],
                   orbit_plane=sat_info['orbit_plane'])

    # 按轨道面组织卫星
    plane_indices = {}
    for i, sat_info in enumerate(active_satellites):
        plane = sat_info['orbit_plane']
        if plane not in plane_indices:
            plane_indices[plane] = []
        plane_indices[plane].append(i)

    logger.info("轨道平面分组情况:")
    for plane, indices in plane_indices.items():
        logger.info(f"{plane}: {len(indices)} 颗卫星")

    # 1. 确保同一轨道平面内的卫星形成环形连接，这是最高优先级
    for plane, indices in plane_indices.items():
        if len(indices) > 1:
            # 使用更简单的算法对同一平面内卫星排序
            # 这里我们假设同一平面卫星在轨道上均匀分布

            # 先对卫星按照某个轨道参数排序(如平近地点角或平近点角)
            # 这个信息可以从TLE提取，但简化起见，我们可以使用位置的某个特征

            # 确保以相对固定的顺序连接同一平面内的卫星
            sorted_indices = indices

            # 打印每个卫星对之间的距离(用于调试)
            for j in range(len(sorted_indices)):
                sat1 = sorted_indices[j]
                sat2 = sorted_indices[(j + 1) % len(sorted_indices)]
                pos1 = positions[sat1]
                pos2 = positions[sat2]
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))
                logger.info(f"平面{plane}: 卫星{sat1}-{sat2}距离: {dist:.2f}km")

            # 强制连接同一平面内的所有相邻卫星(形成环形)
            for j in range(len(sorted_indices)):
                sat1 = sorted_indices[j]
                sat2 = sorted_indices[(j + 1) % len(sorted_indices)]
                pos1 = positions[sat1]
                pos2 = positions[sat2]
                dist = np.linalg.norm(np.array(pos1) - np.array(pos2))

                # 无论距离如何，都添加同平面相邻连接
                G.add_edge(sat1, sat2, link_type="intra-plane", weight=dist)
                logger.info(f"添加同平面链路: {sat1}-{sat2}, 距离: {dist:.2f}km")

    # 2. 添加跨轨道平面链路
    # 只在相邻轨道平面的相应位置的卫星之间添加链路
    plane_names = sorted(plane_indices.keys())

    for i in range(len(plane_names)):
        current_plane = plane_names[i]
        next_plane = plane_names[(i + 1) % len(plane_names)]

        current_indices = plane_indices[current_plane]
        next_indices = plane_indices[next_plane]

        # 确保两个平面有相同数量的卫星
        min_sats = min(len(current_indices), len(next_indices))

        # 为每对对应位置的卫星添加链路
        for j in range(min_sats):
            sat1 = current_indices[j]
            sat2 = next_indices[j]
            pos1 = positions[sat1]
            pos2 = positions[sat2]
            dist = np.linalg.norm(np.array(pos1) - np.array(pos2))

            # 只添加距离合理的跨平面链路
            if dist <= max_link_distance:
                G.add_edge(sat1, sat2, link_type="inter-plane", weight=dist)
                logger.info(f"添加跨平面链路: {sat1}-{sat2}, 距离: {dist:.2f}km")

    return G


# 添加地球遮挡检查函数
def is_earth_blocking(pos1, pos2):
    """检查地球是否遮挡两个卫星之间的链路"""
    earth_radius = 6371  # 地球半径（公里）

    # 计算两点连线与地球中心的最短距离
    # 使用向量公式计算点到线的距离
    p1 = np.array(pos1)
    p2 = np.array(pos2)

    # 两点之间的向量
    v = p2 - p1
    # 向量长度
    v_length = np.linalg.norm(v)
    # 单位向量
    v_unit = v / v_length

    # 从p1指向地球中心的向量
    p1_to_center = -p1  # 地球中心在(0,0,0)

    # 计算p1_to_center在v_unit方向上的投影长度
    projection_length = np.dot(p1_to_center, v_unit)

    # 确保投影点在线段p1-p2上
    if 0 <= projection_length <= v_length:
        # 计算投影点
        projection_point = p1 + projection_length * v_unit

        # 计算投影点到地球中心的距离
        distance_to_center = np.linalg.norm(projection_point)

        # 如果距离小于地球半径，则被地球遮挡
        return distance_to_center < earth_radius

    return False

def visualize_network(G, positions, snapshot_id, output_dir):
    """可视化网络拓扑"""
    # 创建可视化文件夹
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)

    # 2D可视化
    plt.figure(figsize=(12, 10))

    # 准备节点颜色（按轨道面）
    node_colors = []
    plane_labels = sorted(set(nx.get_node_attributes(G, 'orbit_plane').values()))
    color_map = plt.cm.get_cmap('tab10', len(plane_labels))

    color_dict = {plane: color_map(i) for i, plane in enumerate(plane_labels)}

    for node in G.nodes():
        plane = G.nodes[node]['orbit_plane']
        node_colors.append(color_map(plane_labels.index(plane)))

    # 划分边的类型
    intra_edges = [(u, v) for u, v, d in G.edges(data=True) if d['link_type'] == 'intra-plane']
    inter_edges = [(u, v) for u, v, d in G.edges(data=True) if d['link_type'] == 'inter-plane']

    # 准备节点位置
    node_positions = {i: (pos[0], pos[1]) for i, pos in enumerate(positions)}

    # 绘制图
    nx.draw_networkx_nodes(G, pos=node_positions, node_color=node_colors, node_size=100, alpha=0.8)
    nx.draw_networkx_edges(G, pos=node_positions, edgelist=intra_edges, edge_color='blue', alpha=0.5,
                           label='同轨道面链路')
    nx.draw_networkx_edges(G, pos=node_positions, edgelist=inter_edges, edge_color='red', alpha=0.5,
                           label='跨轨道面链路')

    # 添加图例
    plt.legend()
    plt.title(f"Iridium Constellation Network Topology - Snapshot {snapshot_id}")
    plt.axis('off')

    # 保存图像
    plt.savefig(os.path.join(vis_dir, f"network_2d_{snapshot_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 3D可视化
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制地球（简化为球体）
    earth_radius = 6371  # 地球半径（公里）
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = earth_radius * np.cos(u) * np.sin(v)
    y = earth_radius * np.sin(u) * np.sin(v)
    z = earth_radius * np.cos(v)
    ax.plot_surface(x, y, z, color='c', alpha=0.1)

    # 添加卫星
    for i, pos in enumerate(positions):
        plane = G.nodes[i]['orbit_plane']
        color = color_dict[plane]
        ax.scatter(pos[0], pos[1], pos[2], c=[color], s=50, alpha=0.8)

    # 添加链路
    for u, v, data in G.edges(data=True):
        pos_u = positions[u]
        pos_v = positions[v]
        if data['link_type'] == 'intra-plane':
            color = 'blue'
        else:
            color = 'red'
        ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], [pos_u[2], pos_v[2]], color=color, alpha=0.3)

    ax.set_title(f"Iridium Constellation Network Topology 3D View - Snapshot {snapshot_id}")

    # 设置显示范围
    max_range = max([max(abs(np.array(positions)[:, i])) for i in range(3)]) * 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    # 保存3D图
    plt.savefig(os.path.join(vis_dir, f"network_3d_{snapshot_id}.png"), dpi=300, bbox_inches='tight')
    plt.close()

    logger.info(f"已保存快照 {snapshot_id} 的2D和3D可视化图像")


def save_network_data(G, positions, snapshot_id, output_dir):
    """保存网络数据到文件"""
    # 保存为邻接矩阵 CSV 格式
    adj_matrix = nx.to_numpy_array(G)
    adj_csv_file = os.path.join(output_dir, f"adjacency_{snapshot_id}.csv")
    np.savetxt(adj_csv_file, adj_matrix, fmt='%d', delimiter=',')

    # 保存为带权重的邻接矩阵（距离作为权重）
    weight_adj_matrix = nx.to_numpy_array(G, weight='weight')
    weight_adj_csv_file = os.path.join(output_dir, f"weighted_adjacency_{snapshot_id}.csv")
    np.savetxt(weight_adj_csv_file, weight_adj_matrix, fmt='%.2f', delimiter=',')

    # 保存为边列表 TXT 格式
    edge_list = list(G.edges())
    edge_list_file = os.path.join(output_dir, f"edge_list_{snapshot_id}.txt")
    with open(edge_list_file, 'w') as f:
        for u, v in edge_list:
            f.write(f"{u} {v}\n")

    # 计算并保存带距离的边列表
    edge_list_with_dist_file = os.path.join(output_dir, f"edge_list_with_dist_{snapshot_id}.txt")
    with open(edge_list_with_dist_file, 'w') as f:
        for u, v in G.edges():
            distance = G.edges[u, v]['weight']  # 使用已保存的权重
            f.write(f"{u} {v} {distance:.2f}\n")

    # 保存节点属性（轨道面信息） JSON
    node_planes = {node_id: attrs['orbit_plane'] for node_id, attrs in G.nodes(data=True)}
    node_planes_file = os.path.join(output_dir, f"node_planes_{snapshot_id}.json")
    with open(node_planes_file, 'w') as f:
        json.dump(node_planes, f, indent=4)


def analyze_network(G, snapshot_id):
    """分析网络并返回统计信息"""
    stats = {}
    stats["nodes"] = G.number_of_nodes()
    stats["edges"] = G.number_of_edges()
    stats["avg_degree"] = 2 * stats["edges"] / stats["nodes"]
    stats["is_connected"] = nx.is_connected(G)

    # 计算链路距离统计
    distances = [data['weight'] for _, _, data in G.edges(data=True)]
    if distances:
        stats["max_distance"] = max(distances)
        stats["avg_distance"] = sum(distances) / len(distances)

        # 链路类型统计
        stats["intra_plane_links"] = sum(1 for _, _, data in G.edges(data=True) if data['link_type'] == 'intra-plane')
        stats["inter_plane_links"] = sum(1 for _, _, data in G.edges(data=True) if data['link_type'] == 'inter-plane')

    # 度数分布
    degree_counts = [0] * 6  # [度=0, 度=1, 度=2, 度=3, 度=4, 度>4]
    for _, degree in G.degree():
        if degree > 4:
            degree_counts[5] += 1
        else:
            degree_counts[degree] += 1
    stats["degree_distribution"] = degree_counts

    return stats


def main():
    # 开始计时
    start_time = time.time()

    # 1. 读取TLE数据
    tle_file_path = r"/TLE/iridium/iridium_latest.txt"
    all_satellites = load_and_parse_tle(tle_file_path)

    # 2. 使用聚类确定轨道面
    all_satellites = cluster_orbit_planes(all_satellites, n_clusters=6)

    # 3. 选择活跃卫星
    active_satellites = select_active_satellites(all_satellites, satellites_per_plane=11)

    # 提取卫星对象列表
    satellites = [sat_info['sat'] for sat_info in active_satellites]

    # 4. 设置时间范围
    ts = load.timescale()
    start_sim_time = ts.now()
    snapshot_interval = 10  # 每10分钟一次快照
    num_snapshots = 10  # 共10次快照

    # 创建输出目录
    output_dir = os.path.join(os.path.dirname(tle_file_path), "network_snapshots")
    os.makedirs(output_dir, exist_ok=True)

    # 设置链路距离限制
    MAX_LINK_DISTANCE = 4000  # 4000公里距离限制

    # 存储快照数据
    network_snapshots = []
    positions_snapshots = []
    network_stats = []

    # 5. 循环生成快照
    for snapshot in range(num_snapshots):
        logger.info(f"正在生成快照 {snapshot + 1}/{num_snapshots}...")

        # 计算当前时间
        current_time = ts.tt_jd(start_sim_time.tt + (snapshot * snapshot_interval / (24 * 60)))

        # 计算卫星位置
        positions = calculate_satellite_positions(satellites, current_time)

        # 创建网络拓扑
        G = create_network_topology(active_satellites, positions, MAX_LINK_DISTANCE)

        # 保存网络数据
        save_network_data(G, positions, snapshot + 1, output_dir)

        # 可视化网络
        visualize_network(G, positions, snapshot + 1, output_dir)

        # 分析网络
        stats = analyze_network(G, snapshot + 1)
        network_stats.append(stats)

        # 存储快照
        network_snapshots.append(G)
        positions_snapshots.append(positions)

        # 打印链路统计
        logger.info(f"快照 {snapshot + 1} 完成: 节点数={stats['nodes']}, 边数={stats['edges']}")
        logger.info(f"  同轨道面内链路: {stats['intra_plane_links']}, 跨轨道面链路: {stats['inter_plane_links']}")
        logger.info(f"  是否连通: {stats['is_connected']}, 最大链路距离: {stats['max_distance']:.2f} km")

    # 6. 输出汇总统计信息
    logger.info("\n网络快照统计信息:")
    logger.info("=" * 60)
    logger.info(f"{'快照':^8}|{'节点数':^8}|{'边数':^8}|{'平均度':^8}|{'是否连通':^8}|{'最大链路距离':^12}")
    logger.info("-" * 60)

    for i, stats in enumerate(network_stats):
        logger.info(
            f"{i + 1:^8}|{stats['nodes']:^8}|{stats['edges']:^8}|{stats['avg_degree']:^8.2f}|{stats['is_connected']:^8}|{stats['max_distance']:^12.2f}")

    # 节点度数分布
    logger.info("\n节点度数分布:")
    logger.info("=" * 50)
    logger.info(f"{'快照':^10}|{'度=0':^8}|{'度=1':^8}|{'度=2':^8}|{'度=3':^8}|{'度=4':^8}|{'度>4':^8}")
    logger.info("-" * 50)

    for i, stats in enumerate(network_stats):
        degree_counts = stats["degree_distribution"]
        logger.info(
            f"{i + 1:^10}|{degree_counts[0]:^8}|{degree_counts[1]:^8}|{degree_counts[2]:^8}|{degree_counts[3]:^8}|{degree_counts[4]:^8}|{degree_counts[5]:^8}")

    # 链路距离分布分析
    logger.info("\n链路距离分布:")
    logger.info("=" * 50)
    distance_ranges = [(0, 1000), (1000, 2000), (2000, 3000), (3000, 4000), (4000, float('inf'))]
    range_labels = ["0-1000km", "1000-2000km", "2000-3000km", "3000-4000km", ">4000km"]

    for i, G in enumerate(network_snapshots):
        logger.info(f"快照 {i + 1} 链路距离分布:")
        distances = [data['weight'] for _, _, data in G.edges(data=True)]

        # 计算各距离范围的链路数量
        dist_counts = [sum(1 for d in distances if r[0] <= d < r[1]) for r in distance_ranges]

        for j, label in enumerate(range_labels):
            if len(distances) > 0:
                logger.info(f"  {label}: {dist_counts[j]} 条链路 ({dist_counts[j] / len(distances) * 100:.1f}%)")

        if len(distances) > 0:
            logger.info(f"  平均链路距离: {sum(distances) / len(distances):.2f} km")
            logger.info(f"  最大链路距离: {max(distances):.2f} km")
        logger.info("-" * 30)

    # 7. 保存元数据
    metadata = {
        "num_satellites": len(active_satellites),
        "num_snapshots": num_snapshots,
        "snapshot_interval_minutes": snapshot_interval,
        "orbit_planes": list(set(sat['orbit_plane'] for sat in active_satellites)),
        "satellites_per_plane": 11,
        "link_types": ["intra-plane", "inter-plane"],
        "max_links_per_satellite": 4,
        "max_link_distance_km": MAX_LINK_DISTANCE,
        "simulation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "execution_time_seconds": time.time() - start_time
    }

    metadata_file = os.path.join(output_dir, "network_metadata.json")
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    logger.info(f"所有快照数据已保存到: {output_dir}")
    logger.info(f"网络元数据已保存到: {metadata_file}")
    logger.info(f"日志文件已保存到: {log_file}")
    logger.info(f"总运行时间: {time.time() - start_time:.2f} 秒")


if __name__ == "__main__":
    main()
