import argparse
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from TLE.theory.Make_TLE_data import TLEGenerator
from TLE.theory.multi_snap import generate_multi_snapshot_gml


def load_yaml(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_setting_files(config_name):
    main_cfg = load_yaml(PROJECT_ROOT / "settings" / f"{config_name}.yaml")
    defaults = main_cfg.get("defaults", [])
    p_net_name = None
    v_sim_name = None
    for item in defaults:
        if isinstance(item, dict):
            if "p_net_setting" in item:
                p_net_name = item["p_net_setting"]
            elif "v_sim_setting" in item:
                v_sim_name = item["v_sim_setting"]
        elif isinstance(item, str):
            if item.startswith("p_net_setting/"):
                p_net_name = item.split("/", 1)[1]
            elif item.startswith("v_sim_setting/"):
                v_sim_name = item.split("/", 1)[1]

    if not p_net_name or not v_sim_name:
        raise ValueError(f"Cannot find p_net_setting/v_sim_setting defaults in {config_name}.yaml")

    return (
        PROJECT_ROOT / "settings" / "p_net_setting" / f"{p_net_name}.yaml",
        PROJECT_ROOT / "settings" / "v_sim_setting" / f"{v_sim_name}.yaml",
    )


def main():
    parser = argparse.ArgumentParser(description="Generate TLE and satellite topology snapshots.")
    parser.add_argument("--config-name", default="main_ch3", choices=["main_ch3", "main_ch4"])
    parser.add_argument("--inclination", type=float, default=60.0)
    parser.add_argument("--altitude", type=int, default=500)
    parser.add_argument("--distance-threshold", type=float, default=10000.0)
    parser.add_argument("--max-satellites", type=int, default=None)
    args = parser.parse_args()

    p_net_file, v_sim_file = resolve_setting_files(args.config_name)
    p_net_setting = load_yaml(p_net_file)
    v_sim_setting = load_yaml(v_sim_file)

    topology = p_net_setting["topology"]
    planes = int(topology["planes"])
    nums_per_plane = int(topology["nums_per_plane"])
    num_snapshots = int(topology.get("num_snapshots", 10))

    if v_sim_setting.get("snapshot_duration_ms") is not None:
        interval_seconds = float(v_sim_setting["snapshot_duration_ms"]) / 1000.0
    else:
        interval_seconds = float(v_sim_setting["num_v_nets"]) / (
            1000.0 * num_snapshots * float(v_sim_setting["arrival_rate"]["lam"])
        )

    tle_dir = PROJECT_ROOT / "TLE" / "theory" / "Satellite_Data"
    tle_file = tle_dir / f"{int(args.inclination)}Degree_{args.altitude}_{planes}x{nums_per_plane}_tles_1.txt"
    TLEGenerator(
        tle_file,
        planes,
        nums_per_plane,
        args.inclination,
        args.altitude,
        argpo_init=90,
        nodeo_phase=None,
        argpo_phase=True,
    ).generate_tles()

    output_dir = PROJECT_ROOT / "datasets" / "topology" / "snapshots"
    generate_multi_snapshot_gml(
        tle_file,
        output_dir=output_dir,
        p_net_setting=p_net_setting,
        num_snapshots=num_snapshots,
        time_interval_seconds=interval_seconds,
        max_satellites=args.max_satellites,
        distance_threshold=args.distance_threshold,
    )
    print(f"Topology snapshots are ready in {output_dir}")


if __name__ == "__main__":
    main()
