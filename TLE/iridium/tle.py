import math
import random
def generate_iridium_tle(num_planes=6, sats_per_plane=11, inclination=86.4,
                         altitude=780, epoch_year=25, epoch_day=178.0,
                         start_id=90001, output_file="ideal_iridium_tle.txt"):
    """
    生成合理的理想铱星星座TLE数据并保存到文件

    参数:
    num_planes: 轨道平面数量
    sats_per_plane: 每个平面的卫星数量
    inclination: 轨道倾角(度)
    altitude: 轨道高度(km)
    epoch_year: 历元年份(两位数)
    epoch_day: 历元日(带小数)
    start_id: 起始卫星ID
    output_file: 输出文件名

    返回:
    生成的文件路径
    """

    # 计算平均运动(轨道周期)
    earth_radius = 6378.137  # 地球半径(km)
    earth_mu = 398600.4418  # 地球引力常数(km³/s²)
    semi_major_axis = earth_radius + altitude
    mean_motion = math.sqrt(earth_mu / semi_major_axis ** 3) * 86400 / (2 * math.pi)  # 转换为每日圈数

    # 设置合理的偏心率
    eccentricity = 0.0002  # 接近圆但非完美圆

    tle_data = []
    sat_id = start_id

    for plane in range(num_planes):
        raan = (360.0 / num_planes) * plane  # 升交点赤经

        for sat in range(sats_per_plane):
            mean_anomaly = (360.0 / sats_per_plane) * sat  # 平均近点角

            # 添加随机微扰，使数据更真实
            raan_actual = raan + (random.random() * 0.05 - 0.025)  # ±0.025°扰动
            mean_anomaly_actual = mean_anomaly + (random.random() * 0.1 - 0.05)  # ±0.05°扰动

            # 确保角度在0-360范围内
            raan_actual = raan_actual % 360
            mean_anomaly_actual = mean_anomaly_actual % 360

            # 卫星名称
            name = f"IRIDIUM {sat_id - start_id + 1} (PLANE {plane + 1}, SAT {sat + 1})"

            # 合理的drag项
            bstar = 0.000025 + (random.random() * 0.00001)
            bstar_str = f"{bstar:.8f}".replace("0.", "").ljust(8, '0')

            # TLE第一行
            sat_designator = f"00{plane + 1:02d}{'ABCDEFGHJKLMNPQRSTUVWXYZ'[sat % 24]}"
            line1 = f"1 {sat_id:05d}U {sat_designator}   {epoch_year:02d}{epoch_day:09.5f}  .00000100  00000-0  {bstar_str}-0 0  9990"

            # TLE第二行 - 包含合理的参数
            arg_perigee = random.random() * 360  # 随机近地点辐角
            line2 = f"2 {sat_id:05d}  {inclination:.4f} {raan_actual:8.4f} {eccentricity:.7f}  {arg_perigee:8.4f} {mean_anomaly_actual:8.4f} {mean_motion:.8f}  9990"


            # 计算校验和(简化版)
            line1_sum = sum(ord(c) if c.isdigit() else 0 for c in line1)
            line2_sum = sum(ord(c) if c.isdigit() else 0 for c in line2)
            line1 = line1[:-1] + str(line1_sum % 10)
            line2 = line2[:-1] + str(line2_sum % 10)

            tle_data.append(name)
            tle_data.append(line1)
            tle_data.append(line2)

            sat_id += 1

    # 写入文件
    with open(output_file, 'w') as f:
        f.write("\n".join(tle_data))

    return output_file

# 使用以下代码生成TLE文件:
generate_iridium_tle()
