import requests

# 只获取新一代铱星(Iridium NEXT)的TLE数据
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-next&FORMAT=tle"
response = requests.get(url)

if response.status_code == 200:
    tle_data = response.text
    # 去除空行
    tle_lines = [line for line in tle_data.splitlines() if line.strip() != '']
    tle_clean = '\n'.join(tle_lines)

    # 保存到文件
    with open("iridium_next_tle.txt", "w") as f:
        f.write(tle_clean)

    # 计算卫星数量
    satellite_count = len(tle_lines) // 3
    print(f"获取了 {satellite_count} 颗新一代铱星(Iridium NEXT)卫星的TLE数据")
    print("数据已保存到iridium_next_tle.txt")
else:
    print(f"获取失败，状态码: {response.status_code}")
