# -*- coding: utf-8 -*-
from pathlib import Path
import argparse
import math
from functools import lru_cache
import yaml
import pandas as pd
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

ROOT = Path(r'D:\any download\virne-main\virne-main')
RESULT_ROOT = ROOT / 'results' / 'virne_ch4'
FIGURE_ROOT_DIR = Path(r'D:\毕业+就业\毕业大论文\最新图')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

RATIOS = [0.25, 0.40, 0.55, 0.70]
ARRIVAL_RATE = 0.004
SNAPSHOT_DURATION_MS = None
NUM_SNAPSHOTS = None
PREVIEW_MIP = False

ALGORITHMS = {
    'ILP': {
        'dir': 'mip',
        'color': '#1f77b4',
        'marker': 'o',
        'num_requests': 200,
    },
    'ACO-META': {
        'dir': 'aco_meta',
        'color': '#ff7f0e',
        'marker': 's',
        'num_requests': 1000,
    },
    'PPO-Baseline': {
        'dir': 'ppo_mlp+',
        'color': '#2ca02c',
        'marker': '^',
        'num_requests': 1000,
    },
    '本文算法': {
        'dir': 'ppo_gat_seq2seq+',
        'color': '#d62728',
        'marker': 'D',
        'num_requests': 1000,
    },
}

DEFAULT_ALGORITHMS = ['ILP', 'ACO-META', 'PPO-Baseline', '本文算法']
DISPLAY_NAMES = {
    'ILP': 'MIP',
    '本文算法': '本文算法',
}


def display_name(algorithm: str) -> str:
    return DISPLAY_NAMES.get(algorithm, algorithm)

ALGORITHM_ALIASES = {
    'mip': 'ILP',
    'ilp': 'ILP',
    'aco': 'ACO-META',
    'aco_meta': 'ACO-META',
    'aco-meta': 'ACO-META',
    'ppo_mlp+': 'PPO-Baseline',
    'ppo-mlp+': 'PPO-Baseline',
    'ppo_mlp': 'PPO-Baseline',
    'ppo-baseline': 'PPO-Baseline',
    'ppo_baseline': 'PPO-Baseline',
    'ppo_gat_seq2seq+': '本文算法',
    '本文算法': '本文算法',
    'ppo_mgt': '本文算法',
}

SERVICE_LABELS = {
    'overall': '整体',
    'delay_sensitive': '时延敏感型',
    'bandwidth_sensitive': '带宽密集型',
    'reliability_sensitive': '可靠保障型',
    'compute_sensitive': '计算密集型',
}

SERVICE_SETTINGS = {
    'delay_sensitive': {
        'short': 'LSS',
        'filename': 'chapter4_lss_dominant_delay.png',
        'acceptance_filename': 'chapter4_lss_dominant_acceptance_rates.png',
        'metric': 'service_total_latency',
        'fallback_metric': 'total_latency',
        'ylabel': '平均时延（ms）',
        'min_span': 20.0,
        'lower_bound': 0.0,
    },
    'bandwidth_sensitive': {
        'short': 'BSS',
        'filename': 'chapter4_bss_dominant_effective_bandwidth.png',
        'acceptance_filename': 'chapter4_bss_dominant_acceptance_rates.png',
        'metric': 'effective_bandwidth',
        'fallback_metric': None,
        'ylabel': '平均有效带宽',
        'min_span': 12.0,
        'lower_bound': 0.0,
    },
    'reliability_sensitive': {
        'short': 'RSS',
        'filename': 'chapter4_rss_dominant_path_success.png',
        'acceptance_filename': 'chapter4_rss_dominant_acceptance_rates.png',
        'metric': 'path_success_prob',
        'fallback_metric': None,
        'ylabel': '平均路径成功率',
        'min_span': 0.006,
        'lower_bound': 0.0,
        'upper_bound': 1.0,
        'round_to': 0.001,
    },
    'compute_sensitive': {
        'short': 'CSS',
        'filename': 'chapter4_css_dominant_compute_delay.png',
        'acceptance_filename': 'chapter4_css_dominant_acceptance_rates.png',
        'metric': 'compute_delay',
        'fallback_metric': None,
        'ylabel': '平均计算时延（ms）',
        'min_span': 15.0,
        'lower_bound': 0.0,
    },
}

ACCEPTANCE_ITEMS = [
    ('delay_sensitive', '时延敏感型接受率'),
    ('bandwidth_sensitive', '带宽密集型接受率'),
    ('reliability_sensitive', '可靠保障型接受率'),
    ('compute_sensitive', '计算密集型接受率'),
]

SERVICE_ALIASES = {
    'all': 'all',
    'lss': 'delay_sensitive',
    'bss': 'bandwidth_sensitive',
    'rss': 'reliability_sensitive',
    'css': 'compute_sensitive',
}


def next_figure_dir() -> Path:
    FIGURE_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in FIGURE_ROOT_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    save_dir = FIGURE_ROOT_DIR / f'{(max(nums) + 1 if nums else 1):03d}'
    save_dir.mkdir(parents=True, exist_ok=False)
    return save_dir


def read_yaml(path: Path) -> dict:
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


def latest_record(run_dir: Path):
    record_dir = run_dir / 'records'
    if not record_dir.exists():
        return None
    csvs = [p for p in record_dir.glob('*.csv') if not p.name.lower().startswith('temp-')]
    if not csvs:
        return None
    return max(csvs, key=lambda p: (p.stat().st_mtime, p.name))


def is_success(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {'true', '1', 'yes'}


def value_as_bool(value):
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {'true', '1', 'yes'}


def numeric_value(row, column):
    if column not in row:
        return None
    value = pd.to_numeric(row.get(column), errors='coerce')
    if pd.isna(value):
        return None
    return float(value)


def metric_to_qos_flag(metric: str):
    if metric in {'total_latency', 'service_total_latency'}:
        return 'qos_latency_satisfied'
    if metric == 'effective_bandwidth':
        return 'qos_bandwidth_satisfied'
    if metric == 'path_success_prob':
        return 'qos_reliability_satisfied'
    if metric == 'compute_delay':
        return 'qos_computing_satisfied'
    return None


def service_metric_value(row, selected_metric: str, qos_metric: str):
    value = numeric_value(row, selected_metric)
    if value is None:
        return None

    success = is_success(row.get('result')) if 'result' in row else True
    if success:
        return value

    qos_flag = metric_to_qos_flag(qos_metric)
    if qos_flag in row and pd.notna(row.get(qos_flag)) and not value_as_bool(row.get(qos_flag)):
        return value
    return None


def run_info(run_dir: Path):
    cfg = read_yaml(run_dir / 'config.yaml')
    ratios = cfg_get(cfg, 'v_sim_setting.service_qos.service_ratios', {})
    try:
        lam = float(cfg_get(cfg, 'v_sim_setting.arrival_rate.lam'))
    except Exception:
        lam = None
    try:
        num = int(cfg_get(cfg, 'v_sim_setting.num_v_nets'))
    except Exception:
        num = None
    try:
        snapshot_duration_ms = int(cfg_get(cfg, 'v_sim_setting.snapshot_duration_ms'))
    except Exception:
        snapshot_duration_ms = None
    try:
        num_snapshots = int(cfg_get(cfg, 'p_net_setting.topology.num_snapshots'))
    except Exception:
        num_snapshots = None
    return {
        'ratios': ratios if isinstance(ratios, dict) else {},
        'lambda': lam,
        'num_sfc': num,
        'snapshot_duration_ms': snapshot_duration_ms,
        'num_snapshots': num_snapshots,
    }


def ratio_matches(ratios: dict, service: str, ratio: float) -> bool:
    if not ratios:
        return False
    # ratio=0.25 is the uniform setting, so the same run is reused for all
    # four dominant-service figures instead of requiring four duplicate runs.
    target = float(ratios.get(service, -1))
    background = round((1.0 - ratio) / 3.0, 6)
    if abs(target - ratio) > 1e-6:
        return False
    for key in SERVICE_SETTINGS:
        expected = ratio if key == service else background
        if abs(float(ratios.get(key, -1)) - expected) > 1e-6:
            return False
    return True


@lru_cache(maxsize=None)
def find_run(solver_dir_name: str, service: str, ratio: float, num_requests: int):
    base = RESULT_ROOT / solver_dir_name
    if not base.exists():
        return None, None
    candidates = []
    for run_dir in base.iterdir():
        if not run_dir.is_dir():
            continue
        record = latest_record(run_dir)
        if record is None:
            continue
        info = run_info(run_dir)
        if info['num_sfc'] != num_requests:
            continue
        if info['lambda'] is None or abs(info['lambda'] - ARRIVAL_RATE) > 1e-9:
            continue
        if SNAPSHOT_DURATION_MS is not None and info['snapshot_duration_ms'] != SNAPSHOT_DURATION_MS:
            continue
        if NUM_SNAPSHOTS is not None and info['num_snapshots'] != NUM_SNAPSHOTS:
            continue
        if ratio_matches(info['ratios'], service, ratio):
            candidates.append((run_dir.stat().st_mtime, run_dir, record))
    if not candidates:
        return None, None
    # ratio=0.25 是均匀比例，四类主导业务会匹配到同一类实验；取最新完整记录即可。
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[0][1], candidates[0][2]


def read_event_records(record_csv: Path):
    df = pd.read_csv(record_csv)
    if 'event_type' in df.columns:
        df = df[pd.to_numeric(df['event_type'], errors='coerce') == 1]
    return df


def service_metric(record_csv: Path, service: str, metric: str, fallback_metric=None):
    df = read_event_records(record_csv)
    if 'service_type' in df.columns:
        df = df[df['service_type'].astype(str) == service]

    selected_metric = metric
    if selected_metric not in df.columns and fallback_metric and fallback_metric in df.columns:
        selected_metric = fallback_metric
    if selected_metric not in df.columns:
        return float('nan')

    values = []
    for _, row in df.iterrows():
        value = service_metric_value(row, selected_metric, metric)
        if value is not None and pd.notna(value):
            values.append(float(value))
    if not values:
        return float('nan')
    return float(sum(values) / len(values))


def acceptance_rate_from_df(df: pd.DataFrame, service: str = 'overall'):
    if service != 'overall':
        if 'service_type' not in df.columns:
            return float('nan'), 0, 0
        df = df[df['service_type'].astype(str) == service]
    total = len(df)
    if total == 0 or 'result' not in df.columns:
        return float('nan'), 0, 0
    success = int(df['result'].map(is_success).sum())
    return success / total, success, total


def acceptance_rate(record_csv: Path, service: str = 'overall'):
    return acceptance_rate_from_df(read_event_records(record_csv), service)


def make_ylim(values, min_span=20.0, pad_ratio=0.08, lower_bound=0.0, upper_bound=None, round_to=5.0):
    clean = [float(v) for v in values if pd.notna(v)]
    if not clean:
        return None
    low, high = min(clean), max(clean)
    span = max(high - low, min_span)
    center = (high + low) / 2.0
    y0 = center - span / 2.0 - span * pad_ratio
    y1 = center + span / 2.0 + span * pad_ratio
    if round_to and round_to > 0:
        y0 = math.floor(y0 / round_to) * round_to
        y1 = math.ceil(y1 / round_to) * round_to
    y0 = max(lower_bound, y0)
    if upper_bound is not None:
        y1 = min(upper_bound, y1)
    if y1 <= y0:
        y1 = y0 + min_span
    return y0, y1


def parse_algorithms(text: str):
    if not text or text.strip().lower() in {'default', 'complete'}:
        return DEFAULT_ALGORITHMS
    if text.strip().lower() == 'all':
        return list(ALGORITHMS.keys())
    selected = []
    for item in text.split(','):
        key = item.strip()
        if not key:
            continue
        name = ALGORITHM_ALIASES.get(key.lower(), key)
        if name not in ALGORITHMS:
            raise ValueError(f'Unknown algorithm: {key}')
        if name not in selected:
            selected.append(name)
    return selected or DEFAULT_ALGORITHMS


def add_grouped_bars(ax, values_by_algorithm, ylabel, ylim=None):
    x_positions = list(range(len(RATIOS)))
    active_algorithms = list(values_by_algorithm.keys())
    bar_width = min(0.18, 0.72 / max(len(active_algorithms), 1))
    for alg_idx, algorithm in enumerate(active_algorithms):
        alg_setting = ALGORITHMS[algorithm]
        offset = (alg_idx - (len(active_algorithms) - 1) / 2.0) * bar_width
        xs = [x + offset for x in x_positions]
        ys = [r['value'] for r in values_by_algorithm[algorithm]]
        ax.bar(
            xs,
            ys,
            width=bar_width,
            color=alg_setting['color'],
            edgecolor='black',
            linewidth=0.6,
            label=display_name(algorithm),
        )
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'{r:.2f}' for r in RATIOS])
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    return active_algorithms


def preview_mip_qos_value(service: str, ratio: float):
    x = (float(ratio) - 0.25) / 0.45
    x = min(1.0, max(0.0, x))
    if service == 'delay_sensitive':
        return 82.0 + 8.0 * x
    if service == 'bandwidth_sensitive':
        return 29.0 - 1.8 * x
    if service == 'reliability_sensitive':
        return 0.991 - 0.002 * x
    if service == 'compute_sensitive':
        return 25.0 + 5.0 * x
    return float('nan')


def preview_mip_acceptance_value(dominant_service: str, item_key: str, ratio: float):
    x = (float(ratio) - 0.25) / 0.45
    x = min(1.0, max(0.0, x))
    if item_key == dominant_service:
        return 0.72 - 0.09 * x
    return 0.66 - 0.07 * x


def collect_service_qos_rows(service: str, setting: dict, selected_algorithms):
    all_values = []
    printed_rows = []
    values_by_algorithm = {}

    for algorithm in selected_algorithms:
        alg_setting = ALGORITHMS[algorithm]
        rows = []
        for ratio in RATIOS:
            run_dir, record = find_run(
                alg_setting['dir'],
                service,
                ratio,
                int(alg_setting.get('num_requests', 1000)),
            )
            if run_dir is None:
                if PREVIEW_MIP and algorithm == 'ILP':
                    value = preview_mip_qos_value(service, ratio)
                    rows.append({'ratio': ratio, 'value': value, 'run_id': 'mip_preview_only'})
                    printed_rows.append({
                        'algorithm': algorithm,
                        'ratio': ratio,
                        'value': value,
                        'run_id': 'mip_preview_only',
                    })
                    continue
                print(f'Missing QoS result: {algorithm}, {setting["short"]}, ratio={ratio:.2f}')
                rows.append({'ratio': ratio, 'value': float('nan'), 'run_id': ''})
                continue
            value = service_metric(record, service, setting['metric'], setting.get('fallback_metric'))
            rows.append({'ratio': ratio, 'value': value, 'run_id': run_dir.name})
            printed_rows.append({
                'algorithm': algorithm,
                'ratio': ratio,
                'value': value,
                'run_id': run_dir.name,
            })

        if not any(pd.notna(r['value']) for r in rows):
            print(f'Missing all QoS results: {algorithm}, {setting["short"]}')
            continue
        values_by_algorithm[algorithm] = rows
        all_values.extend([r['value'] for r in rows])

    return values_by_algorithm, all_values, printed_rows


def plot_service_qos(save_dir: Path, service: str, setting: dict, selected_algorithms):
    fig, ax = plt.subplots(figsize=(7, 5))
    values_by_algorithm, all_values, printed_rows = collect_service_qos_rows(service, setting, selected_algorithms)

    if not printed_rows:
        plt.close(fig)
        return []

    if 'ylim' in setting:
        ylim = setting['ylim']
    else:
        ylim = make_ylim(
            all_values,
            min_span=setting.get('min_span', 20.0),
            lower_bound=setting.get('lower_bound', 0.0),
            upper_bound=setting.get('upper_bound', None),
            round_to=setting.get('round_to', 5.0),
        )
    add_grouped_bars(ax, values_by_algorithm, setting['ylabel'], ylim=ylim)
    ax.set_xlabel('主导业务占比', labelpad=10)
    ax.legend(loc='best')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(save_dir / setting['filename'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    return printed_rows


def plot_service_qos_2x2(save_dir: Path, selected_algorithms):
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))
    flat_axes = axes.flatten()
    printed_rows = []
    legend_handles = None
    legend_labels = None

    for ax, (service, setting) in zip(flat_axes, SERVICE_SETTINGS.items()):
        values_by_algorithm, all_values, rows = collect_service_qos_rows(service, setting, selected_algorithms)
        printed_rows.extend([{**row, 'service': service} for row in rows])

        if not rows:
            ax.axis('off')
            ax.set_title(f'{SERVICE_LABELS[service]}：数据缺失')
            continue

        if 'ylim' in setting:
            ylim = setting['ylim']
        else:
            ylim = make_ylim(
                all_values,
                min_span=setting.get('min_span', 20.0),
                lower_bound=setting.get('lower_bound', 0.0),
                upper_bound=setting.get('upper_bound', None),
                round_to=setting.get('round_to', 5.0),
            )
        add_grouped_bars(ax, values_by_algorithm, setting['ylabel'], ylim=ylim)
        ax.set_title(f'{SERVICE_LABELS[service]}', fontsize=12, fontweight='bold')
        ax.set_xlabel('主导业务占比', labelpad=10)

        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(len(legend_labels), 4), frameon=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(save_dir / 'chapter4_dominant_qos_2x2.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return printed_rows


def collect_acceptance_rows(dominant_service: str, selected_algorithms):
    rows_by_item = {item_key: {} for item_key, _ in ACCEPTANCE_ITEMS}
    printed_rows = []
    for algorithm in selected_algorithms:
        alg_setting = ALGORITHMS[algorithm]
        rows_by_current_algorithm = {item_key: [] for item_key, _ in ACCEPTANCE_ITEMS}
        for ratio in RATIOS:
            run_dir, record = find_run(
                alg_setting['dir'],
                dominant_service,
                ratio,
                int(alg_setting.get('num_requests', 1000)),
            )
            if run_dir is None:
                if PREVIEW_MIP and algorithm == 'ILP':
                    total = int(alg_setting.get('num_requests', 200))
                    for item_key, _ in ACCEPTANCE_ITEMS:
                        value = preview_mip_acceptance_value(dominant_service, item_key, ratio)
                        success = int(round(value * total))
                        rows_by_current_algorithm[item_key].append({
                            'ratio': ratio,
                            'value': value,
                            'success': success,
                            'total': total,
                            'run_id': 'mip_preview_only',
                        })
                        printed_rows.append({
                            'dominant_service': dominant_service,
                            'acceptance_type': item_key,
                            'algorithm': algorithm,
                            'ratio': ratio,
                            'value': value,
                            'success': success,
                            'total': total,
                            'run_id': 'mip_preview_only',
                        })
                    continue
                print(f'Missing acceptance result: {algorithm}, {SERVICE_SETTINGS[dominant_service]["short"]}, ratio={ratio:.2f}')
                for item_key, _ in ACCEPTANCE_ITEMS:
                    rows_by_current_algorithm[item_key].append({'ratio': ratio, 'value': float('nan'), 'success': 0, 'total': 0, 'run_id': ''})
                continue

            df = read_event_records(record)
            for item_key, _ in ACCEPTANCE_ITEMS:
                value, success, total = acceptance_rate_from_df(df, item_key)
                rows_by_current_algorithm[item_key].append({'ratio': ratio, 'value': value, 'success': success, 'total': total, 'run_id': run_dir.name})
                printed_rows.append({
                    'dominant_service': dominant_service,
                    'acceptance_type': item_key,
                    'algorithm': algorithm,
                    'ratio': ratio,
                    'value': value,
                    'success': success,
                    'total': total,
                    'run_id': run_dir.name,
                })

        for item_key, rows in rows_by_current_algorithm.items():
            if any(pd.notna(r['value']) for r in rows):
                rows_by_item[item_key][algorithm] = rows
    return rows_by_item, printed_rows


def plot_acceptance_rates(save_dir: Path, dominant_service: str, setting: dict, selected_algorithms):
    rows_by_item, printed_rows = collect_acceptance_rows(dominant_service, selected_algorithms)
    if not printed_rows:
        return []

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharex=True, sharey=True)
    flat_axes = axes.flatten()
    legend_handles = None
    legend_labels = None

    for idx, (item_key, title) in enumerate(ACCEPTANCE_ITEMS):
        ax = flat_axes[idx]
        active = add_grouped_bars(ax, rows_by_item[item_key], title, ylim=(0.0, 1.05))
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('主导业务占比', labelpad=10)
        ax.set_ylabel('接受率')
        if active and legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles:
        fig.legend(legend_handles, legend_labels, loc='lower center', ncol=min(len(legend_labels), 4), frameon=True)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(save_dir / setting['acceptance_filename'], dpi=300, bbox_inches='tight')
    plt.close(fig)
    return printed_rows


def main():
    global PREVIEW_MIP, ARRIVAL_RATE, SNAPSHOT_DURATION_MS, NUM_SNAPSHOTS, RATIOS
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--service',
        default='all',
        choices=['all', 'lss', 'bss', 'rss', 'css', *SERVICE_SETTINGS.keys()],
        help='选择单独绘制的主导业务比例图；all 为一次生成四类业务图。',
    )
    parser.add_argument(
        '--algorithms',
        default='default',
        help='逗号分隔算法名。默认绘制 ILP、ACO-META、PPO-Baseline、本文算法；all 表示绘制全部算法。',
    )
    parser.add_argument(
        '--figure-set',
        default='all',
        choices=['all', 'qos', 'acceptance'],
        help='选择输出图类型：qos=只画主导业务QoS指标，acceptance=只画接受率，all=两类都画。',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='可选，直接保存到指定目录；不填写时仍保存到最新图的新编号目录。',
    )
    parser.add_argument(
        '--preview-mip',
        action='store_true',
        help='Preview MIP values when matching MIP results are missing. For layout checking only.',
    )
    parser.add_argument('--arrival-rate', type=float, default=ARRIVAL_RATE, help='筛选实验到达率，例如 0.016。')
    parser.add_argument('--num-v-nets', type=int, default=None, help='筛选每轮 SFC 请求数量，并统一应用到所有算法。')
    parser.add_argument('--snapshot-duration-ms', type=int, default=None, help='筛选快照时长，例如 60000。')
    parser.add_argument('--num-snapshots', type=int, default=None, help='筛选快照数量，例如 20。')
    parser.add_argument('--ratios', default=None, help='逗号分隔主导占比，例如 0.25,0.40,0.55,0.70。')
    args = parser.parse_args()
    PREVIEW_MIP = bool(args.preview_mip)
    ARRIVAL_RATE = float(args.arrival_rate)
    SNAPSHOT_DURATION_MS = args.snapshot_duration_ms
    NUM_SNAPSHOTS = args.num_snapshots
    if args.num_v_nets is not None:
        for setting in ALGORITHMS.values():
            setting['num_requests'] = int(args.num_v_nets)
    if args.ratios:
        RATIOS = [float(x.strip()) for x in args.ratios.split(',') if x.strip()]
    selected = SERVICE_ALIASES.get(args.service, args.service)
    selected_algorithms = parse_algorithms(args.algorithms)

    save_dir = Path(args.output_dir) if args.output_dir else next_figure_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'本次图片保存目录：{save_dir}')
    print(f'当前绘制算法：{", ".join(selected_algorithms)}')
    service_items = SERVICE_SETTINGS.items() if selected == 'all' else [(selected, SERVICE_SETTINGS[selected])]

    for service, setting in service_items:
        if args.figure_set in {'all', 'qos'}:
            rows = plot_service_qos(save_dir, service, setting, selected_algorithms)
            if rows:
                print(f'\n{setting["short"]} 主导业务QoS指标图：')
                for row in rows:
                    print(
                        f'  {row["algorithm"]}: ratio={row["ratio"]:.2f}, '
                        f'value={row["value"]:.6f}, run_id={row["run_id"]}'
                    )

        if args.figure_set in {'all', 'acceptance'}:
            rows = plot_acceptance_rates(save_dir, service, setting, selected_algorithms)
            if rows:
                print(f'\n{setting["short"]} 主导业务接受率图：')
                for row in rows:
                    value_text = 'nan' if not pd.notna(row['value']) else f'{row["value"]:.6f}'
                    print(
                        f'  {row["algorithm"]}: type={SERVICE_LABELS[row["acceptance_type"]]}, '
                        f'ratio={row["ratio"]:.2f}, value={value_text}, '
                        f'success={row["success"]}/{row["total"]}, run_id={row["run_id"]}'
                    )


if __name__ == '__main__':
    main()
