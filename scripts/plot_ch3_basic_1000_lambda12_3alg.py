from pathlib import Path
import argparse
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path(r'D:\any download\virne-main\virne-main')
RESULT_ROOT = ROOT / 'results' / 'virne_ch3'

ALGORITHMS = [
    ('aco_meta', 'ACO-META'),
    ('ppo_mlp+', 'PPO-Baseline'),
    ('ppo_gat_seq2seq+', 'PPO-MGT'),
]

DISPLAY_NAMES = {
    'PPO-MGT': '本文算法',
}

COLORS = {
    'ACO-META': '#ff7f0e',
    'PPO-Baseline': '#2ca02c',
    'PPO-MGT': '#d62728',
}

MARKERS = {
    'ACO-META': 's',
    'PPO-Baseline': '^',
    'PPO-MGT': 'D',
}

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150


def display_name(name):
    return DISPLAY_NAMES.get(name, name)


def read_yaml(path: Path):
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8', errors='ignore')) or {}
    except Exception:
        return {}


def cfg_get(cfg, keys, default=None):
    cur = cfg
    for key in keys.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def ch3_result_roots():
    roots = [RESULT_ROOT]
    direct_root = ROOT / 'virne_ch3'
    if direct_root.exists():
        roots.append(direct_root)
    outputs_root = ROOT / 'outputs'
    if outputs_root.exists():
        roots.extend(p for p in outputs_root.glob('*/*/virne_ch3') if p.is_dir())
    seen, deduped = set(), []
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def latest_record(run_dir: Path):
    record_dir = run_dir / 'records'
    if not record_dir.exists():
        return None
    csvs = [p for p in record_dir.glob('*.csv') if not p.name.lower().startswith('temp-')]
    if not csvs:
        return None
    return max(csvs, key=lambda p: (p.stat().st_mtime, p.name))


def latest_matching_run(solver_dir_name, lam, num_sfc, snapshot_duration_ms, num_snapshots):
    candidates = []
    for root in ch3_result_roots():
        base = root / solver_dir_name
        if not base.exists():
            continue
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            cfg = read_yaml(run_dir / 'config.yaml')
            run_lam = cfg_get(cfg, 'v_sim_setting.arrival_rate.lam')
            run_num = cfg_get(cfg, 'v_sim_setting.num_v_nets')
            run_auto_num = cfg_get(cfg, 'v_sim_setting.auto_num_v_nets_from_arrival_rate')
            run_snapshot_duration = cfg_get(cfg, 'v_sim_setting.snapshot_duration_ms')
            run_num_snapshots = cfg_get(cfg, 'p_net_setting.topology.num_snapshots')
            try:
                ok = (
                    round(float(run_lam), 6) == round(float(lam), 6)
                    and int(run_num) == int(num_sfc)
                    and run_auto_num is False
                    and int(run_snapshot_duration) == int(snapshot_duration_ms)
                    and int(run_num_snapshots) == int(num_snapshots)
                )
            except Exception:
                ok = False
            if not ok:
                continue
            record = latest_record(run_dir)
            summary = run_dir / 'summary.csv'
            if record is None or not summary.exists():
                continue
            candidates.append((run_dir.stat().st_mtime, run_dir, record, summary))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1:]


def is_success(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes'}


def arrival_records(record_csv: Path):
    df = pd.read_csv(record_csv)
    if 'event_type' in df.columns:
        df = df[df['event_type'] == 1]
    return df.copy()


def acceptance_curve(df):
    values, success = [], 0
    for idx, (_, row) in enumerate(df.iterrows(), start=1):
        if is_success(row.get('result', False)):
            success += 1
        values.append(success / idx)
    return values


def running_average(df, metric, only_success=True):
    values, total, count = [], 0.0, 0
    prev = float('nan')
    if metric not in df.columns:
        return []
    for _, row in df.iterrows():
        success = is_success(row.get('result', False))
        if only_success and not success:
            values.append(prev)
            continue
        value = row.get(metric)
        if pd.notna(value):
            total += float(value)
            count += 1
            prev = total / count
        values.append(prev)
    return values


def sampled_xy(values, step=100, start=100):
    if not values:
        return [], []
    n = len(values)
    xs = list(range(min(start, n), n + 1, step))
    if not xs:
        xs = [n]
    if xs[-1] != n:
        xs.append(n)
    ys = [values[x - 1] for x in xs]
    return xs, ys


def plot_curve(curves, ylabel, save_path):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for alg, values in curves.items():
        x, y = sampled_xy(values)
        if not x:
            continue
        ax.plot(
            x, y,
            linestyle='-',
            marker=MARKERS.get(alg, 'o'),
            markerfacecolor='none',
            markersize=4,
            linewidth=1.4,
            color=COLORS.get(alg, '#333333'),
            label=display_name(alg),
        )
    ax.set_xlabel('到达数量（条）', labelpad=10)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1000)
    ax.set_xticks(list(range(100, 1001, 100)))
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lambda-rate', type=float, default=0.016)
    parser.add_argument('--num-sfc', type=int, default=1000)
    parser.add_argument('--snapshot-duration-ms', type=int, default=60000)
    parser.add_argument('--num-snapshots', type=int, default=20)
    parser.add_argument(
        '--output-dir',
        default=str(ROOT / 'paper_outputs' / 'ch3_basic_1000_lambda16_3alg_figures')
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = []
    curves = {
        'acceptance': {},
        'latency': {},
        'r2c': {},
    }

    for solver_dir, alg in ALGORITHMS:
        match = latest_matching_run(
            solver_dir,
            args.lambda_rate,
            args.num_sfc,
            args.snapshot_duration_ms,
            args.num_snapshots,
        )
        if match is None:
            print(f'跳过 {alg}: 未找到匹配结果')
            continue
        run_dir, record_csv, summary_csv = match
        df = arrival_records(record_csv)
        summary = pd.read_csv(summary_csv).iloc[-1]
        curves['acceptance'][alg] = acceptance_curve(df)
        curves['latency'][alg] = running_average(df, 'total_latency')
        curves['r2c'][alg] = running_average(df, 'v_net_r2c_ratio')
        selected_rows.append({
            'algorithm': alg,
            'run_id': run_dir.name,
            'record_csv': str(record_csv),
            'summary_csv': str(summary_csv),
            'num_sfc': args.num_sfc,
            'arrival_rate_per_s': int(round(args.lambda_rate * 1000)),
            'acceptance_rate': summary.get('acceptance_rate'),
            'avg_latency': summary.get('avg_latency'),
            'r2c': summary.get('long_term_r2c_ratio', summary.get('avg_r2c_ratio')),
            'clock_running_time': summary.get('clock_running_time'),
        })

    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / 'ch3_basic_1000_lambda16_selected_runs.csv', index=False, encoding='utf-8-sig')

    plot_curve(curves['acceptance'], '接受率', output_dir / 'ch3_basic_acceptance_rate_curve.png')
    plot_curve(curves['latency'], '平均端到端时延（ms）', output_dir / 'ch3_basic_latency_curve.png')
    plot_curve(curves['r2c'], '收益成本比', output_dir / 'ch3_basic_r2c_curve.png')

    if not selected.empty:
        print(selected[['algorithm', 'run_id', 'acceptance_rate', 'avg_latency', 'r2c', 'clock_running_time']].to_string(index=False))
    print(f'基础实验图片和结果已保存到: {output_dir}')


if __name__ == '__main__':
    main()
