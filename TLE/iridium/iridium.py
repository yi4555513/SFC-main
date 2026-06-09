from sgp4.api import Satrec, jday
from datetime import datetime
import math

# 读取 TLE 数据文件并按轨道平面分组的 Python 脚本
def read_tle_file(file_path):
    """
    读取 TLE 数据文件，返回卫星列表，每个卫星包含名称、TLE 第一行、TLE 第二行。
    """
    satellites = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        # 每 3 行组成一颗卫星的数据
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):  # 确保有完整的 3 行
                name = lines[i].strip()
                tle_line1 = lines[i + 1].strip()
                tle_line2 = lines[i + 2].strip()
                satellites.append({
                    'name': name,
                    'tle_line1': tle_line1,
                    'tle_line2': tle_line2
                })
    return satellites


def extract_raan(tle_line2):
    """
    从 TLE 第二行提取升交点赤经（RAAN，第 4 字段）。
    """
    fields = tle_line2.split()
    if len(fields) >= 4:
        try:
            return float(fields[3])  # RAAN 在 TLE 第二行的第 4 字段
        except ValueError:
            return None
    return None



def assign_to_orbital_plane(raan, tolerance=10):
    """
    根据 RAAN 分配到轨道平面，超出容差的不分配。
    tolerance: 容许的RAAN偏离度数
    """
    # Iridium NEXT轨道平面中心RAAN
    plane_centers = [78, 110, 142, 173, 205, 236]

    for i, center in enumerate(plane_centers, 1):
        diff = abs(raan - center)
        # 考虑RAAN跨0度的情况
        diff = min(diff, 360 - diff)
        if diff <= tolerance:
            return i
    return None


def group_satellites_by_plane(satellites, tolerance=1):
    """
    根据RAAN和容差将卫星分组到轨道平面。
    """
    orbital_planes = {1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    for sat in satellites:
        raan = extract_raan(sat['tle_line2'])
        if raan is not None:
            plane = assign_to_orbital_plane(raan, tolerance)
            if plane is not None:
                orbital_planes[plane].append(sat)
            else:
                print(f"⚠️ 卫星 {sat['name']} (RAAN: {raan:.4f}) 偏离所有轨道平面，已剔除")
        else:
            print(f"⚠️ 无法解析卫星 {sat['name']} 的 RAAN")

    return orbital_planes

def extract_mean_anomaly(tle_line2):
    """
    从TLE第二行提取Mean Anomaly（平均近点角，第7字段）。
    """
    fields = tle_line2.split()
    if len(fields) >= 7:
        try:
            return float(fields[6])
        except ValueError:
            return None
    return None

def print_orbital_planes(orbital_planes):
    """
    打印每个轨道平面的卫星列表，并根据Mean Anomaly（平均近点角）显示它们在轨道内的相对位置。
    """
    for plane, satellites in orbital_planes.items():
        print(f"\n🛰️ 轨道平面 {plane} ({len(satellites)} 颗卫星):")

        # 提取Mean Anomaly并排序
        satellites_with_anomaly = []
        for sat in satellites:
            mean_anomaly = extract_mean_anomaly(sat['tle_line2'])
            if mean_anomaly is not None:
                satellites_with_anomaly.append((sat, mean_anomaly))
            else:
                print(f"⚠️ 无法解析Mean Anomaly: {sat['name']}")

        satellites_with_anomaly.sort(key=lambda x: x[1])  # 按Mean Anomaly排序

        # 打印排序后的卫星
        for idx, (sat, anomaly) in enumerate(satellites_with_anomaly, start=1):
            print(f"  {idx:02d}. {sat['name']} | RAAN: {extract_raan(sat['tle_line2']):.4f}° | Mean Anomaly: {anomaly:.2f}°")

        print(f"🟢 总计: {len(satellites)} 颗卫星（已按轨道内相对位置排序）")


def export_tle_by_plane(orbital_planes):
    """
    打印每个轨道平面的卫星TLE数据。
    """
    for plane, satellites in orbital_planes.items():
        print(f"\n🚀 === 轨道平面 {plane} === 共 {len(satellites)} 颗卫星 ===")
        for sat in satellites:
            print(sat['name'])
            print(sat['tle_line1'])
            print(sat['tle_line2'])
        print("=======================================")

def main():
    # 替换为你的TLE文件路径
    file_path = r'/TLE/iridium/ideal_iridium_tle.txt'

    satellites = read_tle_file(file_path)
    print(f"总计读取 {len(satellites)} 颗卫星")

    # 设置RAAN容差
    tolerance = 1  # 可以根据需要调整，比如10度或者更严格

    orbital_planes = group_satellites_by_plane(satellites, tolerance)

    # 打印TLE原始数据
    # export_tle_by_plane(orbital_planes)

    # 同时打印按轨道内Mean Anomaly排序的信息
    print_orbital_planes(orbital_planes)

if __name__ == "__main__":
    main()