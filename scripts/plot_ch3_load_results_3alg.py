from pathlib import Path
import re
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(r'D:\any download\virne-main\virne-main')
RESULT_ROOT = ROOT / 'results' / 'virne_ch3'
SAVE_ROOT = Path(r'D:\毕业+就业\毕业大论文\最新图')
TARGET_RATES = [0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021]
FIXED_TOTAL_TIME_MS = 180000
NUM_SNAPSHOTS = 3

# 如果某个算法没有完整 summary.csv，会自动跳过，不会报错。
ALGORITHMS = [
    ('mip', 'ILP'),
    ('aco_meta', 'ACO-Meta'),
    ('ppo_mlp+', 'PPO-Baseline'),
    ('ppo_gat_seq2seq+', 'PPO-MGT'),
]

DISPLAY_NAMES = {
    'ILP': 'MIP',
    'PPO-MGT': '本文算法',
}


def display_name(label):
    return DISPLAY_NAMES.get(label, label)


def lambda_to_arrival_per_second(lam):
    return int(round(float(lam) * 1000))

# 保持和前面负载图一致：ACO绿色方块，PPO-Baseline橙色三角，PPO-MGT蓝色圆点。
COLORS = {
    'ILP': '#d62728',
    'ACO-Meta': '#2ca02c',
    'PPO-Baseline': '#ff7f0e',
    'PPO-MGT': '#1f77b4',
}
MARKERS = {
    'ILP': 'D',
    'ACO-Meta': 's',
    'PPO-Baseline': '^',
    'PPO-MGT': 'o',
}

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def next_save_dir():
    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in SAVE_ROOT.iterdir():
        if p.is_dir() and p.name.isdigit():
            nums.append(int(p.name))
    n = max(nums) + 1 if nums else 1
    d = SAVE_ROOT / f'{n:03d}'
    d.mkdir(parents=True, exist_ok=False)
    return d


def parse_config(run_dir: Path):
    cfg = run_dir / 'config.yaml'
    if not cfg.exists():
        return None
    txt = cfg.read_text(encoding='utf-8', errors='ignore')
    lam_m = re.search(r'arrival_rate:\s*(?:\n\s+.*)*?\n\s+lam:\s*([0-9.]+)', txt) or re.search(r'lam:\s*([0-9.]+)', txt)
    num_m = re.search(r'num_v_nets:\s*([0-9]+)', txt)
    ft_m = re.search(r'fixed_total_time_ms:\s*([0-9.]+)', txt)
    ns_m = re.search(r'num_snapshots:\s*([0-9]+)', txt)
    return {
        'lambda': float(lam_m.group(1)) if lam_m else None,
        'num_v_nets': int(num_m.group(1)) if num_m else None,
        'fixed_total_time_ms': float(ft_m.group(1)) if ft_m else None,
        'num_snapshots': int(ns_m.group(1)) if ns_m else None,
    }


def latest_record(run_dir: Path):
    rec_dir = run_dir / 'records'
    if not rec_dir.exists():
        return None
    files = list(rec_dir.glob('*.csv'))
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def active_resource_metrics(record_file: Path):
    df = pd.read_csv(record_file).sort_values('event_time')
    if df.empty:
        return {}
    for c in ['event_type', 'v_net_id', 'event_time', 'v_net_node_cost', 'v_net_link_cost',
              'inservice_count', 'p_net_node_available_resource', 'p_net_link_available_resource',
              'p_net_node_resource_utilization', 'p_net_link_resource_utilization']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    first = df.iloc[0]
    try:
        total_node = float(first.p_net_node_available_resource) / (1.0 - float(first.p_net_node_resource_utilization))
        total_link = float(first.p_net_link_available_resource) / (1.0 - float(first.p_net_link_resource_utilization))
    except Exception:
        total_node = 18004.0
        total_link = 21150.0

    active = {}
    rows = []
    for _, r in df.iterrows():
        try:
            vid = int(r.v_net_id)
        except Exception:
            continue
        event_type = int(r.event_type) if not pd.isna(r.event_type) else 1
        if event_type == 1:
            if str(r.get('result', '')) == 'True':
                node_cost = float(r.v_net_node_cost) if not pd.isna(r.v_net_node_cost) else 0.0
                link_cost = float(r.v_net_link_cost) if not pd.isna(r.v_net_link_cost) else 0.0
                active[vid] = (node_cost, link_cost)
        else:
            active.pop(vid, None)
        ins = float(r.inservice_count) if not pd.isna(r.inservice_count) else float(len(active))
        rows.append([float(r.event_time), ins, sum(v[0] for v in active.values()), sum(v[1] for v in active.values())])

    res = pd.DataFrame(rows, columns=['time', 'inservice', 'node_cost', 'link_cost']).sort_values('time')
    t = res['time'].to_numpy()
    end = float(t[-1]) if len(t) else 1.0
    if end <= 0:
        end = 1.0
    dt = np.r_[t[1:] - t[:-1], 0.0]
    return {
        'avg_inservice': float((res['inservice'].to_numpy() * dt).sum() / end),
        'peak_inservice': int(res['inservice'].max()),
        'avg_node_util': float((res['node_cost'].to_numpy() * dt).sum() / end / total_node),
        'peak_node_util': float(res['node_cost'].max() / total_node),
        'avg_link_util': float((res['link_cost'].to_numpy() * dt).sum() / end / total_link),
        'peak_link_util': float(res['link_cost'].max() / total_link),
    }


def collect_rows():
    rows = []
    for solver_name, label in ALGORITHMS:
        base = RESULT_ROOT / solver_name
        if not base.exists():
            continue
        latest = {}
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            cfg = parse_config(run_dir)
            if not cfg or cfg['lambda'] is None:
                continue
            lam = round(cfg['lambda'], 3)
            if lam not in [round(x, 3) for x in TARGET_RATES]:
                continue
            if cfg['fixed_total_time_ms'] is None or abs(cfg['fixed_total_time_ms'] - FIXED_TOTAL_TIME_MS) > 1:
                continue
            if cfg['num_snapshots'] != NUM_SNAPSHOTS:
                continue
            summary = run_dir / 'summary.csv'
            record = latest_record(run_dir)
            if not summary.exists() or record is None:
                continue
            if lam not in latest or run_dir.stat().st_mtime > latest[lam].stat().st_mtime:
                latest[lam] = run_dir

        for lam in sorted(latest):
            run_dir = latest[lam]
            cfg = parse_config(run_dir)
            s = pd.read_csv(run_dir / 'summary.csv').iloc[0]
            rec = latest_record(run_dir)
            row = {
                'algorithm': label,
                'solver': solver_name,
                'lambda': lam,
                'num_sfc': cfg['num_v_nets'],
                'run_id': run_dir.name,
                'acceptance_rate': float(s.get('acceptance_rate', math.nan)),
                'avg_latency': float(s.get('avg_latency', math.nan)),
                'r2c': float(s.get('long_term_r2c_ratio', math.nan)),
                'route_failure_count': float(s.get('route_failure_count', math.nan)) if 'route_failure_count' in s else math.nan,
                'place_failure_count': float(s.get('place_failure_count', math.nan)) if 'place_failure_count' in s else math.nan,
            }
            row.update(active_resource_metrics(rec))
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(['algorithm', 'lambda'])


def plot_metric(df, save_dir, y_col, ylabel, filename, percent=False):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for _, label in ALGORITHMS:
        sub = df[df['algorithm'] == label].sort_values('lambda')
        if sub.empty:
            continue
        y = sub[y_col].to_numpy(dtype=float)
        if percent:
            y = y * 100
        ax.plot(sub['lambda'].apply(lambda_to_arrival_per_second), y, label=display_name(label), color=COLORS[label], marker=MARKERS[label],
                linewidth=1.8, markersize=4.5)
    ax.set_xlabel('SFC 到达率（条/s）')
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle='--', alpha=0.35)
    ax.legend(frameon=True)
    ax.set_xticks([lambda_to_arrival_per_second(x) for x in TARGET_RATES])
    fig.tight_layout()
    fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    df = collect_rows()
    save_dir = next_save_dir()
    if df.empty:
        print('没有找到可画的完整结果。')
        print('Saved empty dir:', save_dir)
        return
    df.to_csv(save_dir / 'ch3_load_summary.csv', index=False, encoding='utf-8-sig')

    plot_metric(df, save_dir, 'acceptance_rate', 'SFC 请求接受率（%）', 'ch3_load_acceptance_rate.png', percent=True)
    plot_metric(df, save_dir, 'avg_latency', '成功部署 SFC 的平均端到端时延（ms）', 'ch3_load_avg_latency.png')
    plot_metric(df, save_dir, 'r2c', '成功部署 SFC 的平均收益成本比（Revenue/Cost）', 'ch3_load_r2c.png')
    plot_metric(df, save_dir, 'avg_node_util', '平均节点资源利用率（节点占用资源/节点总资源，%）', 'ch3_load_node_resource_utilization.png', percent=True)
    plot_metric(df, save_dir, 'avg_link_util', '平均链路资源利用率（链路占用带宽/链路总带宽，%）', 'ch3_load_link_resource_utilization.png', percent=True)

    print('Saved to:', save_dir)
    print(df[['algorithm','lambda','num_sfc','acceptance_rate','avg_latency','r2c','avg_node_util','avg_link_util']].to_string(index=False))

if __name__ == '__main__':
    main()
