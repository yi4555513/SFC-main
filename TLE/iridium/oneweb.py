from sgp4.api import Satrec
import math


def read_tle_file(file_path):
    satellites = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for i in range(0, len(lines), 3):
            if i + 2 < len(lines):
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
    try:
        return float(tle_line2[17:25])
    except:
        return None


def extract_mean_anomaly(tle_line2):
    fields = tle_line2.split()
    if len(fields) >= 7:
        try:
            return float(fields[6])
        except ValueError:
            return None
    return None


def assign_to_orbital_plane(raan, tolerance=10):
    """
    根据 RAAN 将卫星分配到 OneWeb 的18个轨道平面
    """
    plane_centers = [i * 20 for i in range(18)]  # 0°, 20°, ..., 340°

    for i, center in enumerate(plane_centers, 1):
        diff = abs(raan - center)
        diff = min(diff, 360 - diff)  # 跨越0度修正
        if diff <= tolerance:
            return i
    return None


def group_satellites_by_plane(satellites, tolerance=10):
    orbital_planes = {i: [] for i in range(1, 19)}  # 18个轨道面

    for sat in satellites:
        raan = extract_raan(sat['tle_line2'])
        if raan is not None:
            plane = assign_to_orbital_plane(raan, tolerance)
            if plane is not None:
                orbital_planes[plane].append(sat)
            else:
                print(f"⚠️ 卫星 {sat['name']} (RAAN: {raan:.4f}) 不符合任何轨道平面")
        else:
            print(f"⚠️ 无法解析卫星 {sat['name']} 的 RAAN")

    return orbital_planes


def print_orbital_planes(orbital_planes):
    for plane, satellites in orbital_planes.items():
        print(f"\n🛰️ 轨道平面 {plane} ({len(satellites)} 颗卫星):")

        satellites_with_anomaly = []
        for sat in satellites:
            mean_anomaly = extract_mean_anomaly(sat['tle_line2'])
            if mean_anomaly is not None:
                satellites_with_anomaly.append((sat, mean_anomaly))
            else:
                print(f"⚠️ 无法解析Mean Anomaly: {sat['name']}")

        satellites_with_anomaly.sort(key=lambda x: x[1])

        for idx, (sat, anomaly) in enumerate(satellites_with_anomaly, start=1):
            print(f"  {idx:02d}. {sat['name']} | RAAN: {extract_raan(sat['tle_line2']):.4f}° | Mean Anomaly: {anomaly:.2f}°")

        print(f"🟢 总计: {len(satellites)} 颗卫星（已按轨道内相对位置排序）")


def export_tle_by_plane(orbital_planes):
    for plane, satellites in orbital_planes.items():
        print(f"\n🚀 === 轨道平面 {plane} === 共 {len(satellites)} 颗卫星 ===")
        for sat in satellites:
            print(sat['name'])
            print(sat['tle_line1'])
            print(sat['tle_line2'])
        print("=======================================")


def main():
    # 替换为你的TLE文件路径
    file_path = r'/TLE/iridium/oneweb.txt'

    satellites = read_tle_file(file_path)
    print(f"✅ 总计读取 {len(satellites)} 颗卫星")

    # 分组
    orbital_planes = group_satellites_by_plane(satellites, tolerance=1)

    # 打印轨道平面信息
    print_orbital_planes(orbital_planes)

    # 导出TLE（可选）
    # export_tle_by_plane(orbital_planes)


if __name__ == "__main__":
    main()
