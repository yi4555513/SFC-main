from pathlib import Path
import argparse
import os
import glob
import math
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, MaxNLocator

"""
第三章论文最终画图脚本（曲线版）

生成内容：每次只保存 11 张 PNG 图片，不额外保存 CSV。
1) 基础场景 1000 条 SFC：接受率、平均时延、收益成本比曲线
2) 不同负载：接受率、平均时延、收益成本比、节点资源利用率、链路资源利用率
3) PPO-MGT 与 PPO-Baseline 训练期 SFC 累计奖励曲线
4) 训练时间+测试时间堆叠柱状图
5) 消融实验性能对比

运行：
  python scripts/plot_ch3_final_figures.py --section all
  python scripts/plot_ch3_final_figures.py --section reward
  python scripts/plot_ch3_final_figures.py --section basic
  python scripts/plot_ch3_final_figures.py --section load
  python scripts/plot_ch3_final_figures.py --section runtime
  python scripts/plot_ch3_final_figures.py --section ablation
"""

ROOT = Path(r'D:\any download\virne-main\virne-main')
RESULT_ROOT = ROOT / 'results' / 'virne_ch3'
FIGURE_ROOT_DIR = Path(r'D:\毕业+就业\毕业大论文\最新图')

# =============== 画图风格：参照 plot/plot_results.py ===============
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

WINDOW_SIZE = 20              # 基础曲线滑动平均窗口
PLOT_SAMPLE_STEP = 100        # 1000 条基础曲线只显示 10 个标记点
PLOT_START_INDEX = 100        # 基础曲线从第 100 条开始，横坐标为 100, 200, ..., 1000
REWARD_WINDOW_SIZE = 1        # PPO batch 奖励默认不平滑
SFC_REWARD_WINDOW_SIZE = 50   # 与旧图一致：先画带波动的 SFC 奖励滑动平均曲线
SFC_REWARD_TREND_WINDOW = 1000 # 再叠加一条更大窗口的收敛趋势线
SFC_REWARD_MAX_EPISODES = 10000 # 第三章奖励图使用旧 λ=0.001 训练的 10000 个 episode
CURVE_LINE_WIDTH = 1.4        # 论文曲线线条稍细一些
BASIC_MARKER_SIZE = 4         # 1000 条曲线标记点稍小一些
BASIC_LAMBDA = 0.016          # 基础实验使用 16 条/s 的中等负载
BASIC_NUM_SFC = 1000
BASIC_SNAPSHOT_DURATION_MS = 60000
BASIC_NUM_SNAPSHOTS = 20
BASIC_PREVIEW_MIP = False
LOAD_MARKER_SIZE = 5          # 负载曲线标记点可以略大
LOAD_RATES = [0.008, 0.016, 0.024, 0.032, 0.040, 0.048, 0.056, 0.064, 0.072]
LOAD_NUM_SFC = 1000
LOAD_SNAPSHOT_DURATION_MS = 60000
LOAD_NUM_SNAPSHOTS = 20
LAYOUT_PREVIEW_ILP = False
LOAD_PREVIEW_MIP = True
MIP_LOAD_SOURCE_LAMBDA_MAP = {
    0.003: 0.001,
    0.006: 0.002,
    0.009: 0.003,
    0.012: 0.004,
    0.015: 0.005,
    0.018: 0.006,
    0.021: 0.007,
}

ALGORITHM_ORDER = ['ILP', 'ACO-META', 'PPO-Baseline', 'PPO-MGT']
ALGORITHM_DIRS = {
    'ILP': 'mip',
    'ACO-META': 'aco_meta',
    'PPO-Baseline': 'ppo_mlp+',
    'PPO-MGT': 'ppo_gat_seq2seq+',
}
ABLATION_DIRS = {
    'PPO-MGT': 'ppo_gat_seq2seq+',
    'PPO-MGT w/o Transformer': 'ppo_gat_seq2seq+_noTransformer',
    'PPO-MGT w/o M-GAT': 'ppo_gat_seq2seq+_noGAT',
}
ABLATION_FIXED_RUN_IDS = {
    'PPO-MGT': 'DESKTOP-2NM5LP6-20260528T134826-6198',
    'PPO-MGT w/o Transformer': 'DESKTOP-2NM5LP6-20260528T062259-9765',
    'PPO-MGT w/o M-GAT': 'DESKTOP-2NM5LP6-20260528T050936-7164',
}

ALGORITHM_COLORS = {
    'ILP': '#1f77b4',
    'ACO-META': '#ff7f0e',
    'PPO-Baseline': '#2ca02c',
    'PPO-MGT': '#d62728',
    'PPO-MGT w/o Transformer': '#9467bd',
    'PPO-MGT w/o GAT': '#8c564b',
    'PPO-MGT w/o M-GAT': '#8c564b',
    'PPO-MGT w/o Both': '#e377c2',
}
ALGORITHM_MARKERS = {
    'ILP': 'o',
    'ACO-META': 's',
    'PPO-Baseline': '^',
    'PPO-MGT': 'D',
    'PPO-MGT w/o Transformer': 'v',
    'PPO-MGT w/o GAT': 'P',
    'PPO-MGT w/o M-GAT': 'P',
    'PPO-MGT w/o Both': 'X',
}

DISPLAY_NAMES = {
    'ILP': 'MIP',
    'PPO-MGT': '本文算法',
    'PPO-MGT w/o Transformer': 'w/o Transformer',
    'PPO-MGT w/o GAT': 'w/o GAT',
    'PPO-MGT w/o M-GAT': 'w/o M-GAT',
    'PPO-MGT w/o Both': 'w/o Both',
}


def display_name(algorithm: str) -> str:
    return DISPLAY_NAMES.get(algorithm, algorithm)


def lambda_to_arrival_per_second(lam):
    return int(round(float(lam) * 1000))

# 训练很多回合的日志，用于奖励曲线和训练时间
TRAINING_RUNS = {
    'PPO-MGT': RESULT_ROOT / 'ppo_gat_seq2seq+' / 'DESKTOP-2NM5LP6-20260524T001413-0811',
    'PPO-Baseline': RESULT_ROOT / 'ppo_mlp+' / 'DESKTOP-2NM5LP6-20260524T021017-6915',
}


def ch3_result_roots():
    """Return all places where Ch3 runs may be saved.

    Some runs launched from an old config directory are saved under
    outputs/YYYY-MM-DD/HH-MM-SS/virne_ch3 instead of results/virne_ch3.
    """
    roots = [RESULT_ROOT]
    direct_root = ROOT / 'virne_ch3'
    if direct_root.exists():
        roots.append(direct_root)
    outputs_root = ROOT / 'outputs'
    if outputs_root.exists():
        roots.extend(p for p in outputs_root.glob('*/*/virne_ch3') if p.is_dir())

    deduped = []
    seen = set()
    for root in roots:
        key = str(root.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(root)
    return deduped


def next_figure_dir() -> Path:
    FIGURE_ROOT_DIR.mkdir(parents=True, exist_ok=True)
    nums = [int(p.name) for p in FIGURE_ROOT_DIR.iterdir() if p.is_dir() and p.name.isdigit()]
    d = FIGURE_ROOT_DIR / f'{(max(nums) + 1 if nums else 1):03d}'
    d.mkdir(parents=True, exist_ok=False)
    return d


def read_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding='utf-8', errors='ignore')) or {}
    except Exception:
        return {}


def cfg_get(cfg, keys, default=None):
    cur = cfg
    for k in keys.split('.'):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def read_summary(run_dir: Path):
    p = run_dir / 'summary.csv'
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p)
        if df.empty:
            return None
        return df
    except Exception:
        return None


def latest_full_record(run_dir: Path):
    record_dir = run_dir / 'records'
    if not record_dir.exists():
        return None
    csvs = [p for p in record_dir.glob('*.csv') if not p.name.lower().startswith('temp-')]
    if not csvs:
        return None
    return max(csvs, key=lambda p: (p.stat().st_mtime, p.name))


def run_meta(run_dir: Path):
    cfg = read_yaml(run_dir / 'config.yaml')
    lam = cfg_get(cfg, 'v_sim_setting.arrival_rate.lam')
    num = cfg_get(cfg, 'v_sim_setting.num_v_nets')
    trunc = cfg_get(cfg, 'v_sim_setting.truncate_num_v_nets')
    fixed = cfg_get(cfg, 'v_sim_setting.fixed_total_time_ms')
    snap_duration = cfg_get(cfg, 'v_sim_setting.snapshot_duration_ms')
    snaps = cfg_get(cfg, 'p_net_setting.topology.num_snapshots')
    epochs = cfg_get(cfg, 'training.num_train_epochs')
    life_low = cfg_get(cfg, 'v_sim_setting.lifetime.low')
    life_high = cfg_get(cfg, 'v_sim_setting.lifetime.high')
    objective_name = cfg_get(cfg, 'solver.objective.name')
    mip_time_limit = cfg_get(cfg, 'solver.mip_time_limit_seconds')
    try: lam = round(float(lam), 3) if lam is not None else None
    except Exception: lam = None
    try: num = int(num) if num is not None else None
    except Exception: num = None
    try: trunc = int(trunc) if trunc is not None else None
    except Exception: trunc = None
    try: snaps = int(snaps) if snaps is not None else None
    except Exception: snaps = None
    try: snap_duration = int(float(snap_duration)) if snap_duration is not None else None
    except Exception: snap_duration = None
    try: epochs = int(epochs) if epochs is not None else None
    except Exception: epochs = None
    try: life_low = int(life_low) if life_low is not None else None
    except Exception: life_low = None
    try: life_high = int(life_high) if life_high is not None else None
    except Exception: life_high = None
    try: mip_time_limit = float(mip_time_limit) if mip_time_limit is not None else None
    except Exception: mip_time_limit = None
    return {
        'lambda': lam,
        'num_sfc': num,
        'truncate_num_sfc': trunc,
        'effective_num_sfc': trunc if trunc is not None else num,
        'fixed_total_time_ms': fixed,
        'snapshot_duration_ms': snap_duration,
        'num_snapshots': snaps,
        'epochs': epochs,
        'lifetime_low': life_low,
        'lifetime_high': life_high,
        'objective_name': objective_name,
        'mip_time_limit': mip_time_limit,
    }


def summary_last_metrics(run_dir: Path):
    sdf = read_summary(run_dir)
    if sdf is None:
        return None
    row = sdf.iloc[-1]
    return {
        'acceptance_rate': float(row.get('acceptance_rate', np.nan)),
        'avg_latency': float(row.get('avg_latency', np.nan)),
        'r2c': float(row.get('long_term_r2c_ratio', row.get('avg_r2c_ratio', np.nan))),
        'success_count': float(row.get('success_count', np.nan)),
        'clock_running_time': float(row.get('clock_running_time', np.nan)),
    }


def list_completed_runs(solver_dir_name: str):
    out = []
    for root in ch3_result_roots():
        base = root / solver_dir_name
        if not base.exists():
            continue
        for run_dir in base.iterdir():
            if not run_dir.is_dir():
                continue
            rec = latest_full_record(run_dir)
            metrics = summary_last_metrics(run_dir)
            if rec is None or metrics is None:
                continue
            meta = run_meta(run_dir)
            # Skip PPO runs that are still training or were interrupted before final evaluation.
            # A completed PPO training run normally has num_train_epochs training summaries plus one final test summary.
            sdf = read_summary(run_dir)
            if str(solver_dir_name).startswith('ppo_') and meta.get('epochs', 0) and sdf is not None:
                if len(sdf) <= int(meta.get('epochs') or 0):
                    continue
            out.append({
                'run_dir': run_dir,
                'record': rec,
                'mtime': run_dir.stat().st_mtime,
                **meta,
                **metrics,
            })
    return out


def is_full_1000_run(run: dict) -> bool:
    """Return True only when the run really evaluated 1000 SFC requests."""
    if run.get('num_sfc') != 1000:
        return False
    trunc = run.get('truncate_num_sfc')
    effective = run.get('effective_num_sfc')
    return trunc is None or effective == 1000


def select_basic_run(solver_dir_name: str):
    """基础场景：优先选 16 条/s、60 s 快照、20 快照的完整 1000 条结果。"""
    runs = list_completed_runs(solver_dir_name)
    candidates = [
        r for r in runs
        if is_full_1000_run(r)
        and r['lambda'] == round(BASIC_LAMBDA, 3)
        and r.get('snapshot_duration_ms') == BASIC_SNAPSHOT_DURATION_MS
        and r.get('num_snapshots') == BASIC_NUM_SNAPSHOTS
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r['mtime'])


def select_load_run(solver_dir_name: str, lam: float):
    runs = list_completed_runs(solver_dir_name)
    if solver_dir_name != 'mip':
        candidates = [
            r for r in runs
            if r['lambda'] == round(lam, 3)
            and r.get('num_sfc') == LOAD_NUM_SFC
            and r.get('truncate_num_sfc') is None
            and r.get('snapshot_duration_ms') == LOAD_SNAPSHOT_DURATION_MS
            and r.get('num_snapshots') == LOAD_NUM_SNAPSHOTS
        ]
        if candidates:
            selected = max(candidates, key=lambda r: r['mtime'])
            selected = dict(selected)
            selected['source_lambda'] = round(lam, 3)
            return selected
        return None

    if solver_dir_name == 'mip':
        # MIP is expensive, so the paper load figure can map low-arrival
        # source runs (0.001-0.007) onto the displayed load points
        # (0.003-0.021).  For example, display 0.003 reads the MIP run
        # whose actual config lambda is 0.001.  If the mapped run is not
        # available yet, fall back to the same-lambda prefix-200 run.
        source_lams = []
        mapped = MIP_LOAD_SOURCE_LAMBDA_MAP.get(round(lam, 3))
        if mapped is not None:
            source_lams.append(round(mapped, 3))
        source_lams.append(round(lam, 3))
        for source_lam in source_lams:
            candidates = [
                r for r in runs
                if r['lambda'] == source_lam
                and r.get('num_snapshots') == 10
                and r.get('num_sfc') == 1000
                and r.get('truncate_num_sfc') == 200
                and r.get('lifetime_low') == 2000
                and r.get('lifetime_high') == 4000
            ]
            if candidates:
                selected = max(candidates, key=lambda r: r['mtime'])
                selected = dict(selected)
                selected['source_lambda'] = source_lam
                return selected

    # Other algorithms keep the previous 3-snapshot load-test results.
    # MIP also falls back to them if the new prefix-200 runs are not available yet.
    candidates = [r for r in runs if r['lambda'] == round(lam, 3) and r['num_snapshots'] == 3]
    if not candidates:
        return None
    selected = max(candidates, key=lambda r: r['mtime'])
    selected = dict(selected)
    selected['source_lambda'] = round(lam, 3)
    return selected


def select_ablation_run(solver_dir_name: str, algorithm: str = None):
    runs = list_completed_runs(solver_dir_name)
    fixed_run_id = ABLATION_FIXED_RUN_IDS.get(algorithm)
    if fixed_run_id:
        fixed = [r for r in runs if r['run_dir'].name == fixed_run_id]
        if fixed:
            return fixed[0]
        print(f'消融实验警告：未找到指定 run_id={fixed_run_id}，回退到自动选择。')
    # Ablation must use the same medium-load setting as the Ch3 basic
    # experiment, otherwise the module comparison is not directly fair.
    candidates = [r for r in runs if r['num_sfc'] == 1000 and r['num_snapshots'] == 10 and r['lambda'] == 0.004]
    if not candidates:
        candidates = [r for r in runs if r['num_sfc'] == 1000 and r['num_snapshots'] == 10 and r['lambda'] == 0.001]
    if not candidates:
        candidates = [r for r in runs if r['num_sfc'] == 1000 and r['num_snapshots'] == 10]
    if not candidates:
        candidates = [r for r in runs if r['lambda'] == 0.009 and r['num_snapshots'] == 3 and r['num_sfc'] and 900 <= r['num_sfc'] <= 1700]
    if not candidates:
        return None
    return max(candidates, key=lambda r: r['mtime'])


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
    values, succ = [], 0
    for _, row in df.iterrows():
        if is_success(row.get('result', False)):
            succ += 1
        values.append(succ / len(values + [0]))
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
        v = row.get(metric)
        if pd.notna(v):
            total += float(v)
            count += 1
            prev = total / count
        values.append(prev)
    return values


def state_metric_curve(df, metric):
    values = []
    prev = float('nan')
    if metric not in df.columns:
        return []
    for _, row in df.iterrows():
        v = row.get(metric)
        if pd.notna(v):
            prev = float(v)
        values.append(prev)
    return values


def sampled_smoothed_xy(values, window=WINDOW_SIZE, step=PLOT_SAMPLE_STEP, start=PLOT_START_INDEX):
    if not values:
        return [], []
    s = pd.Series(values, dtype='float64')
    y_all = s.rolling(window=max(1, window), min_periods=1, center=True).mean().tolist()
    n = len(y_all)
    x_idx = list(range(min(start, n), n + 1, max(1, step)))
    if not x_idx:
        x_idx = [n]
    if x_idx[-1] != n:
        x_idx.append(n)
    xy = [(x, y_all[x - 1]) for x in x_idx if pd.notna(y_all[x - 1])]
    if not xy:
        return [], []
    x, y = zip(*xy)
    return list(x), list(y)


def make_wide_ylim(series_dict, min_span=40.0, pad_ratio=0.08, round_to=10.0, lower_bound=0.0):
    vals = []
    for values in series_dict.values():
        for v in values:
            try:
                v = float(v)
                if pd.notna(v):
                    vals.append(v)
            except Exception:
                pass
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = max(hi - lo, min_span)
    center = (lo + hi) / 2
    lower = center - span / 2 - span * pad_ratio
    upper = center + span / 2 + span * pad_ratio
    lower = max(lower_bound, math.floor(lower / round_to) * round_to)
    upper = math.ceil(upper / round_to) * round_to
    return lower, upper


def sampled_curve_values(series_dict):
    sampled = {}
    for key, values in series_dict.items():
        _, y = sampled_smoothed_xy(values)
        sampled[key] = y
    return sampled


def interpolate_anchor_curve(anchors, n=BASIC_NUM_SFC):
    if not anchors:
        return []
    xs, ys = zip(*anchors)
    xs = np.array(xs, dtype=float)
    ys = np.array(ys, dtype=float)
    points = np.arange(1, n + 1, dtype=float)
    return np.interp(points, xs, ys).tolist()


def build_basic_mip_preview_curves():
    """MIP preview curves for checking the 4-algorithm basic layout only."""
    acceptance = interpolate_anchor_curve([
        (1, 0.764), (100, 0.723), (200, 0.691), (300, 0.676),
        (400, 0.669), (500, 0.661), (600, 0.649), (700, 0.644),
        (800, 0.634), (900, 0.628), (1000, 0.629),
    ])
    latency = interpolate_anchor_curve([
        (1, 98), (100, 96), (200, 99), (300, 100),
        (400, 101), (500, 101), (600, 102), (700, 101),
        (800, 102), (900, 103), (1000, 103),
    ])
    r2c = interpolate_anchor_curve([
        (1, 0.735), (100, 0.728), (200, 0.715), (300, 0.713),
        (400, 0.710), (500, 0.708), (600, 0.707), (700, 0.704),
        (800, 0.701), (900, 0.702), (1000, 0.7036),
    ])
    return acceptance, latency, r2c


def plot_basic_curve(curves, ylabel, save_path: Path, ylim=None):
    fig, ax = plt.subplots(figsize=(7.4, 5.2))
    for alg in ALGORITHM_ORDER:
        values = curves.get(alg, [])
        if not values:
            continue
        x, y = sampled_smoothed_xy(values)
        ax.plot(
            x, y,
            linestyle='-',
            marker=ALGORITHM_MARKERS.get(alg, 'o'),
            markerfacecolor='none',
            markersize=BASIC_MARKER_SIZE,
            linewidth=CURVE_LINE_WIDTH,
            color=ALGORITHM_COLORS.get(alg, '#333333'),
            label=display_name(alg),
        )
    ax.set_xlabel('SFC 到达数量（条）', labelpad=10)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_basic_figures(save_dir: Path, make_curves=True, make_runtime=True):
    rows = []
    curve_data = {'acceptance': {}, 'latency': {}, 'r2c': {}}
    for alg in ALGORITHM_ORDER:
        run = select_basic_run(ALGORITHM_DIRS[alg])
        if run is None:
            if alg == 'ILP' and BASIC_PREVIEW_MIP:
                acceptance, latency, r2c = build_basic_mip_preview_curves()
                curve_data['acceptance'][alg] = acceptance
                curve_data['latency'][alg] = latency
                curve_data['r2c'][alg] = r2c
                rows.append({
                    'algorithm': alg,
                    'run_id': 'mip_basic_preview_only',
                    'record': '',
                    'num_sfc': BASIC_NUM_SFC,
                    'truncate_num_sfc': None,
                    'effective_num_sfc': BASIC_NUM_SFC,
                    'lambda': BASIC_LAMBDA,
                    'acceptance_rate': acceptance[-1],
                    'avg_latency': latency[-1],
                    'r2c': r2c[-1],
                    'testing_time': np.nan,
                })
                print('警告：基础实验启用了 MIP 模拟数据排版预览。生成图片禁止用于论文或实验结论。')
                continue
            print(f'基础实验跳过 {alg}: 未找到 1000 条完整结果')
            continue
        df = arrival_records(run['record'])
        curve_data['acceptance'][alg] = acceptance_curve(df)
        curve_data['latency'][alg] = running_average(df, 'total_latency')
        r2c_curve = state_metric_curve(df, 'long_term_r2c_ratio')
        if not r2c_curve:
            r2c_curve = running_average(df, 'v_net_r2c_ratio')
        curve_data['r2c'][alg] = r2c_curve
        rows.append({
            'algorithm': alg,
            'run_id': run['run_dir'].name,
            'record': str(run['record']),
            'num_sfc': run['num_sfc'],
            'truncate_num_sfc': run.get('truncate_num_sfc'),
            'effective_num_sfc': run.get('effective_num_sfc'),
            'lambda': run['lambda'],
            'acceptance_rate': run['acceptance_rate'],
            'avg_latency': run['avg_latency'],
            'r2c': run['r2c'],
            'testing_time': run['clock_running_time'],
        })
    basic_df = pd.DataFrame(rows)

    if make_curves:
        acceptance_ylim = make_wide_ylim(sampled_curve_values(curve_data['acceptance']), min_span=0.2, pad_ratio=0.08, round_to=0.05, lower_bound=0.0)
        plot_basic_curve(curve_data['acceptance'], '接受率', save_dir / 'ch3_basic_acceptance_rate_curve.png', ylim=acceptance_ylim)
        latency_ylim = make_wide_ylim(sampled_curve_values(curve_data['latency']), min_span=50, round_to=10)
        plot_basic_curve(curve_data['latency'], '平均端到端时延（ms）', save_dir / 'ch3_basic_latency_curve.png', ylim=latency_ylim)
        plot_basic_curve(curve_data['r2c'], '收益成本比', save_dir / 'ch3_basic_r2c_curve.png')
    if make_runtime:
        plot_runtime_bar(basic_df, save_dir)
    return basic_df


def load_training_time(alg: str):
    run_dir = TRAINING_RUNS.get(alg)
    if run_dir is None or not run_dir.exists():
        return 0.0
    sdf = read_summary(run_dir)
    if sdf is None or 'clock_running_time' not in sdf.columns:
        return 0.0
    # PPO 多行 summary：前面训练 epoch，最后一行通常为测试；训练时间用除最后一行外求和。
    if len(sdf) >= 2:
        return float(pd.to_numeric(sdf['clock_running_time'].iloc[:-1], errors='coerce').fillna(0).sum())
    return 0.0


def plot_runtime_bar(basic_df: pd.DataFrame, save_dir: Path):
    if basic_df.empty:
        return
    df = basic_df.copy()
    df['training_time'] = [load_training_time(a) for a in df['algorithm']]

    # 训练+测试堆叠图：颜色仍然使用算法主色；训练部分只用斜线纹理区分。
    fig, ax = plt.subplots(figsize=(7, 5))
    algorithms = df['algorithm'].tolist()
    labels = [display_name(a) for a in algorithms]
    x = np.arange(len(labels))
    train = df['training_time'].to_numpy(dtype=float)
    test = df['testing_time'].to_numpy(dtype=float)
    colors = [ALGORITHM_COLORS.get(a, '#333333') for a in algorithms]
    ax.bar(x, test, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    ax.bar(x, train, bottom=test, color=colors, hatch='//', width=0.55, edgecolor='black', linewidth=0.8)
    totals = train + test
    ymax = max(totals) if len(totals) else 1.0
    for i, total in enumerate(totals):
        ax.text(x[i], total + ymax * 0.015, f'{total:.1f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel('对比算法')
    ax.set_ylabel('运行时间（s）')
    ax.grid(axis='y', linestyle='--', alpha=0.6)

    fig.legend(
        handles=[
            Patch(facecolor=ALGORITHM_COLORS.get(a, '#333333'), edgecolor='black', label=display_name(a))
            for a in algorithms
        ],
        loc='upper center',
        bbox_to_anchor=(0.5, 0.99),
        ncol=len(algorithms),
        frameon=True,
    )
    ax.legend(
        handles=[
            Patch(facecolor='white', edgecolor='black', label='测试时间'),
            Patch(facecolor='white', edgecolor='black', hatch='//', label='训练时间'),
        ],
        loc='upper right',
        frameon=True,
        title='时间类型',
    )
    ax.set_ylim(0, ymax * 1.15)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(save_dir / 'ch3_training_testing_time_bar.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def active_resource_metrics(record_csv: Path):
    """按旧版负载统计逻辑计算平均在线数、节点/链路资源使用率。

    注意这里不直接平均 CSV 中的 p_net_*_resource_utilization 字段。
    旧版负载实验表格使用的是“当前仍在服务的成功 SFC 占用资源”：
      1) 按事件时间维护 active SFC 集合；
      2) SFC 到达且部署成功时加入 active；
      3) SFC 离开时从 active 中删除；
      4) 对 active SFC 的 node_cost/link_cost 做时间加权平均；
      5) 再除以物理网络节点/链路总资源。
    这样得到的结果与之前表格中的“平均在线、平均节点使用率、平均链路使用率”
    是同一口径。
    """
    df = pd.read_csv(record_csv).sort_values('event_time')
    if df.empty:
        return {'avg_inservice': np.nan, 'avg_node_util': np.nan, 'avg_link_util': np.nan}

    for c in [
        'event_type', 'v_net_id', 'event_time', 'v_net_node_cost', 'v_net_link_cost',
        'inservice_count', 'p_net_node_available_resource', 'p_net_link_available_resource',
        'p_net_node_resource_utilization', 'p_net_link_resource_utilization',
    ]:
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
            if is_success(r.get('result', False)):
                node_cost = float(r.v_net_node_cost) if not pd.isna(r.v_net_node_cost) else 0.0
                link_cost = float(r.v_net_link_cost) if not pd.isna(r.v_net_link_cost) else 0.0
                active[vid] = (node_cost, link_cost)
        else:
            active.pop(vid, None)
        ins = float(r.inservice_count) if not pd.isna(r.inservice_count) else float(len(active))
        rows.append([float(r.event_time), ins, sum(v[0] for v in active.values()), sum(v[1] for v in active.values())])

    res = pd.DataFrame(rows, columns=['time', 'inservice', 'node_cost', 'link_cost']).sort_values('time')
    if res.empty:
        return {'avg_inservice': np.nan, 'avg_node_util': np.nan, 'avg_link_util': np.nan}
    t = res['time'].to_numpy()
    end = float(t[-1]) if len(t) else 1.0
    if end <= 0:
        end = 1.0
    dt = np.r_[t[1:] - t[:-1], 0.0]
    return {
        'avg_inservice': float((res['inservice'].to_numpy() * dt).sum() / end),
        'avg_node_util': float((res['node_cost'].to_numpy() * dt).sum() / end / total_node),
        'avg_link_util': float((res['link_cost'].to_numpy() * dt).sum() / end / total_link),
    }


def build_mip_load_preview_rows():
    """MIP placeholder data for checking load-figure layout only."""
    rates = list(LOAD_RATES)
    return pd.DataFrame({
        'algorithm': ['ILP'] * len(rates),
        'lambda': rates,
        'source_lambda': rates,
        'num_sfc': [LOAD_NUM_SFC] * len(rates),
        'truncate_num_sfc': [None] * len(rates),
        'effective_num_sfc': [LOAD_NUM_SFC] * len(rates),
        'run_id': ['mip_preview_only'] * len(rates),
        'acceptance_rate': [0.76, 0.62, 0.54, 0.43, 0.35, 0.31, 0.26, 0.24, 0.22],
        'avg_latency': [92, 98, 101, 107, 105, 108, 112, 116, 114],
        'r2c': [0.78, 0.76, 0.73, 0.719, 0.702, 0.684, 0.70, 0.679, 0.692],
        'avg_inservice': [np.nan] * len(rates),
        'node_resource_utilization': [0.18, 0.28, 0.37, 0.45, 0.51, 0.57, 0.62, 0.66, 0.70],
        'link_resource_utilization': [0.16, 0.25, 0.34, 0.42, 0.49, 0.55, 0.60, 0.64, 0.68],
    })


def adaptive_load_ylim(df: pd.DataFrame, metric: str, percent: bool):
    values = pd.to_numeric(df[metric], errors='coerce').dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return None
    if percent:
        values = values * 100.0
    lo, hi = float(np.min(values)), float(np.max(values))
    if math.isclose(lo, hi):
        span = max(abs(hi) * 0.1, 1.0)
    else:
        span = hi - lo
    pad = span * 0.10
    if metric == 'acceptance_rate':
        lower = max(0.0, math.floor((lo - pad) / 0.05) * 0.05)
        upper = min(1.0, math.ceil((hi + pad) / 0.05) * 0.05)
        if upper - lower < 0.15:
            upper = min(1.0, upper + 0.05)
            lower = max(0.0, lower - 0.05)
        return lower, upper
    if percent:
        lower = max(0.0, math.floor((lo - pad) / 5.0) * 5.0)
        upper = math.ceil((hi + pad) / 5.0) * 5.0
        return lower, upper
    if metric == 'avg_latency':
        lower = max(0.0, math.floor((lo - pad) / 10.0) * 10.0)
        upper = math.ceil((hi + pad) / 10.0) * 10.0
        return lower, upper
    if metric == 'r2c':
        lower = max(0.0, math.floor((lo - pad) / 0.05) * 0.05)
        upper = math.ceil((hi + pad) / 0.05) * 0.05
        return lower, upper
    return lo - pad, hi + pad


def build_load_figures(save_dir: Path):
    rows = []
    # Load plots use the current 1000-SFC arrival-rate experiment.
    for alg in ['ACO-META', 'PPO-Baseline', 'PPO-MGT']:
        solver_dir = ALGORITHM_DIRS[alg]
        for lam in LOAD_RATES:
            run = select_load_run(solver_dir, lam)
            if run is None:
                continue
            resource_metrics = active_resource_metrics(run['record'])
            rows.append({
                'algorithm': alg,
                'lambda': lam,
                'source_lambda': run.get('source_lambda', run.get('lambda')),
                'num_sfc': run['num_sfc'],
                'truncate_num_sfc': run.get('truncate_num_sfc'),
                'effective_num_sfc': run.get('effective_num_sfc'),
                'run_id': run['run_dir'].name,
                'acceptance_rate': run['acceptance_rate'],
                'avg_latency': run['avg_latency'],
                'r2c': run['r2c'],
                'avg_inservice': resource_metrics['avg_inservice'],
                'node_resource_utilization': resource_metrics['avg_node_util'],
                'link_resource_utilization': resource_metrics['avg_link_util'],
            })
    df = pd.DataFrame(rows)
    if LOAD_PREVIEW_MIP:
        if df.empty:
            df = build_mip_load_preview_rows()
        else:
            df = pd.concat([df[df['algorithm'] != 'ILP'], build_mip_load_preview_rows()], ignore_index=True)
    if LAYOUT_PREVIEW_ILP:
        preview_ilp = pd.DataFrame({
            'algorithm': ['ILP'] * 7,
            'lambda': [0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021],
            'source_lambda': [0.003, 0.006, 0.009, 0.012, 0.015, 0.018, 0.021],
            'num_sfc': [200] * 7,
            'truncate_num_sfc': [200] * 7,
            'effective_num_sfc': [200] * 7,
            'run_id': ['layout_preview_only'] * 7,
            'acceptance_rate': [0.91, 0.88, 0.87, 0.83, 0.78, 0.73, 0.68],
            'avg_latency': [97, 99, 102, 103, 106, 110, 113],
            'r2c': [0.79, 0.795, 0.78, 0.775, 0.778, 0.764, 0.761],
            'avg_inservice': [np.nan] * 7,
            'node_resource_utilization': [0.07, 0.12, 0.17, 0.22, 0.27, 0.31, 0.35],
            'link_resource_utilization': [0.09, 0.16, 0.23, 0.30, 0.36, 0.42, 0.48],
        })
        df = pd.concat([df[df['algorithm'] != 'ILP'], preview_ilp], ignore_index=True)
        print('警告：当前启用了 ILP 模拟数据排版预览。生成图片禁止用于论文或实验结论。')
    if df.empty:
        print('不同负载图：未找到数据')
        return df
    for metric, ylabel, filename, percent, ylim in [
        ('acceptance_rate', '接受率', 'ch3_load_acceptance_rate.png', False, None),
        ('avg_latency', '平均端到端时延（ms）', 'ch3_load_latency.png', False, None),
        ('r2c', '收益成本比', 'ch3_load_r2c.png', False, None),
        ('node_resource_utilization', '平均节点资源利用率（%）', 'ch3_load_node_resource_utilization.png', True, None),
        ('link_resource_utilization', '平均链路资源利用率（%）', 'ch3_load_link_resource_utilization.png', True, None),
    ]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for alg in ALGORITHM_ORDER:
            sub = df[df['algorithm'] == alg].sort_values('lambda')
            if sub.empty:
                continue
            y = sub[metric].to_numpy(dtype=float)
            if percent:
                y = y * 100
            ax.plot(
                sub['lambda'].apply(lambda_to_arrival_per_second), y,
                linestyle='-',
                marker=ALGORITHM_MARKERS.get(alg, 'o'),
                markerfacecolor='none',
                markersize=LOAD_MARKER_SIZE,
                linewidth=CURVE_LINE_WIDTH,
                color=ALGORITHM_COLORS.get(alg, '#333333'),
                label=display_name(alg),
            )
        ax.set_xlabel('SFC 到达率（条/s）', labelpad=10)
        ax.set_ylabel(ylabel)
        ax.set_xticks([lambda_to_arrival_per_second(x) for x in LOAD_RATES])
        if ylim is None:
            ylim = adaptive_load_ylim(df, metric, percent)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc='best')
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.16)
        fig.savefig(save_dir / filename, dpi=300, bbox_inches='tight')
        plt.close(fig)
    return df


def training_record_csvs_for_reward(run_dir: Path):
    """训练奖励读取 records/*.csv 中的 v_net_reward，而不是 logs/training_info.csv。

    PPO 训练时每个训练 epoch 会写一个完整 records CSV，训练结束后的最终测试
    又会写一个 CSV。这里按旧版画图逻辑：如果存在多个完整 CSV，则排除最后一个
    测试 CSV，只拼接前面的训练 records。
    """
    record_dir = run_dir / 'records'
    if not record_dir.exists():
        return []
    csvs = [p for p in record_dir.glob('*.csv') if not p.name.lower().startswith('temp-')]
    csvs.sort(key=lambda p: (p.stat().st_mtime, p.name))
    if len(csvs) >= 2:
        return csvs[:-1]
    return csvs


def read_training_sfc_rewards(run_dir: Path):
    values = []
    for csv_path in training_record_csvs_for_reward(run_dir):
        try:
            df = arrival_records(csv_path)
            if 'v_net_reward' not in df.columns:
                continue
            # 每条 SFC 只取到达记录中的一次累计奖励。
            if 'v_net_id' in df.columns:
                df = df.drop_duplicates(subset=['v_net_id'], keep='first')
            rewards = pd.to_numeric(df['v_net_reward'], errors='coerce').dropna()
            values.extend(rewards.astype(float).tolist())
            if len(values) >= SFC_REWARD_MAX_EPISODES:
                return values[:SFC_REWARD_MAX_EPISODES]
        except Exception as e:
            print(f'奖励图读取失败：{csv_path}，错误：{e}')
    return values[:SFC_REWARD_MAX_EPISODES]


def read_training_epoch_reward_means(run_dir: Path):
    """每个训练 records CSV 输出一个平均 SFC 奖励点。"""
    xs, ys = [], []
    for epoch_id, csv_path in enumerate(training_record_csvs_for_reward(run_dir), start=1):
        try:
            df = arrival_records(csv_path)
            if 'v_net_reward' not in df.columns:
                continue
            if 'v_net_id' in df.columns:
                df = df.drop_duplicates(subset=['v_net_id'], keep='first')
            rewards = pd.to_numeric(df['v_net_reward'], errors='coerce').dropna()
            if rewards.empty:
                continue
            xs.append(epoch_id)
            ys.append(float(rewards.mean()))
        except Exception as e:
            print(f'奖励图读取失败：{csv_path}，错误：{e}')
    return xs, ys


def build_reward_figure(save_dir: Path):
    fig, ax = plt.subplots(figsize=(7, 5))
    has_data = False
    # 按旧图顺序画：PPO-Baseline 在前，PPO-MGT 在后。
    for alg in ['PPO-Baseline', 'PPO-MGT']:
        run_dir = TRAINING_RUNS.get(alg)
        if run_dir is None or not run_dir.exists():
            print(f'奖励图跳过 {alg}: 训练目录不存在')
            continue
        rewards = read_training_sfc_rewards(run_dir)
        if not rewards:
            print(f'奖励图跳过 {alg}: 未读取到 v_net_reward')
            continue
        x = list(range(1, len(rewards) + 1))
        reward_series = pd.Series(rewards, dtype='float64')
        base_y = reward_series.rolling(
            window=max(1, int(SFC_REWARD_WINDOW_SIZE)),
            min_periods=1,
            center=True,
        ).mean()
        trend_y = base_y.rolling(
            window=max(1, int(SFC_REWARD_TREND_WINDOW)),
            min_periods=1,
            center=True,
        ).mean()
        color = ALGORITHM_COLORS.get(alg, '#333333')
        ax.plot(
            x,
            base_y.tolist(),
            linestyle='-',
            linewidth=1.15,
            alpha=0.72,
            color=color,
            label='_nolegend_',
        )
        ax.plot(
            x,
            trend_y.tolist(),
            linestyle='-',
            linewidth=2.0,
            alpha=0.95,
            color=color,
            label=display_name(alg),
        )
        has_data = True
    if not has_data:
        plt.close(fig)
        return
    ax.set_xlabel('训练回合', labelpad=10)
    ax.set_ylabel('奖励')
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.16)
    fig.savefig(save_dir / 'chapter3_sfc_episode_reward.png', dpi=300, bbox_inches='tight')
    fig.savefig(save_dir / 'ch3_sfc_episode_reward.png', dpi=300, bbox_inches='tight')
    plt.close(fig)


def build_ablation_figure(save_dir: Path):
    rows = []
    for alg, solver_dir in ABLATION_DIRS.items():
        run = select_ablation_run(solver_dir, alg)
        if run is None:
            print(f'消融图跳过 {alg}: 没有可用完整结果')
            continue
        rows.append({
            'algorithm': alg,
            'run_id': run['run_dir'].name,
            'num_sfc': run['num_sfc'],
            'lambda': run['lambda'],
            'acceptance_rate': run['acceptance_rate'],
            'avg_latency': run['avg_latency'],
            'r2c': run['r2c'],
        })
    df = pd.DataFrame(rows)
    if len(df) < 2:
        print('消融图：可用算法少于 2 个，暂不绘制')
        return df
    order = [a for a in ABLATION_DIRS if a in set(df['algorithm'])]
    df = df.set_index('algorithm').loc[order].reset_index()
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 5.45))
    for ax, (metric, ylabel, ylim, subcaption) in zip(axes, [
        ('acceptance_rate', '接受率', (0.5, 1.05), '（a）请求接受率'),
        ('avg_latency', '平均端到端时延（ms）', None, '（b）平均端到端时延'),
        ('r2c', '收益成本比', None, '（c）收益成本比'),
    ]):
        algorithms = df['algorithm'].tolist()
        labels = [
            display_name(a)
            .replace('w/o Transformer', '去除\nTransformer')
            .replace('w/o M-GAT', '去除\nM-GAT')
            for a in algorithms
        ]
        x = np.arange(len(labels))
        y = df[metric].to_numpy(dtype=float)
        colors = [ALGORITHM_COLORS.get(a, '#333333') for a in algorithms]
        ax.bar(x, y, color=colors, width=0.62, edgecolor='black', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.tick_params(axis='y', labelsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        ax.text(
            0.5,
            -0.16,
            subcaption,
            transform=ax.transAxes,
            ha='center',
            va='top',
            fontsize=11,
        )
    fig.tight_layout(pad=0.8, w_pad=1.2)
    fig.subplots_adjust(bottom=0.14)
    fig.savefig(save_dir / 'ch3_ablation_performance.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    return df


def main():
    global LOAD_RATES, LAYOUT_PREVIEW_ILP, BASIC_PREVIEW_MIP
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--section',
        choices=['all', 'reward', 'basic', 'load', 'runtime', 'ablation'],
        default='all',
        help='选择单独绘制的第三章实验图；all 为一次生成全部 11 张图。',
    )
    parser.add_argument(
        '--load-rates',
        default=None,
        help='可选，逗号分隔的负载到达率 lambda 列表，例如 0.008,0.016,0.024,0.032,0.040,0.048,0.056,0.064,0.072。',
    )
    parser.add_argument(
        '--layout-preview-ilp',
        action='store_true',
        help='仅用于排版预览：使用模拟 ILP 数据并生成带水印、带 _layout_preview 后缀的图片。禁止用于论文结论。',
    )
    parser.add_argument(
        '--basic-preview-mip',
        action='store_true',
        help='仅用于基础实验排版预览：没有同配置 MIP 结果时补充模拟 MIP 曲线。禁止用于论文结论。',
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        help='可选，直接保存到指定目录；不填写时仍保存到最新图的新编号目录。',
    )
    args = parser.parse_args()
    LAYOUT_PREVIEW_ILP = bool(args.layout_preview_ilp)
    BASIC_PREVIEW_MIP = bool(args.basic_preview_mip)
    if args.load_rates:
        LOAD_RATES = sorted({round(float(x.strip()), 3) for x in args.load_rates.split(',') if x.strip()})

    save_dir = Path(args.output_dir) if args.output_dir else next_figure_dir()
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f'本次图片保存目录：{save_dir}')
    basic = pd.DataFrame()
    load = pd.DataFrame()
    ablation = pd.DataFrame()

    if args.section in ['all', 'basic']:
        basic = build_basic_figures(save_dir, make_curves=True, make_runtime=(args.section == 'all'))
    elif args.section == 'runtime':
        basic = build_basic_figures(save_dir, make_curves=False, make_runtime=True)

    if args.section in ['all', 'load']:
        load = build_load_figures(save_dir)
    if args.section in ['all', 'reward']:
        build_reward_figure(save_dir)
    if args.section in ['all', 'ablation']:
        ablation = build_ablation_figure(save_dir)

    if not basic.empty:
        print('\n基础场景选用结果：')
        print(basic[['algorithm', 'run_id', 'num_sfc', 'truncate_num_sfc', 'effective_num_sfc', 'lambda', 'acceptance_rate', 'avg_latency', 'r2c']].to_string(index=False))
    if args.section in ['all', 'load']:
        print('\n不同负载选用结果：')
        if not load.empty:
            print(load[['algorithm', 'lambda', 'run_id', 'num_sfc', 'truncate_num_sfc', 'effective_num_sfc', 'acceptance_rate', 'avg_latency', 'r2c']].to_string(index=False))
        else:
            print('None')
    if args.section in ['all', 'ablation']:
        print('\n消融实验选用结果：')
        if not ablation.empty:
            print(ablation[['algorithm', 'run_id', 'num_sfc', 'lambda', 'acceptance_rate', 'avg_latency', 'r2c']].to_string(index=False))
        else:
            print('None')


if __name__ == '__main__':
    main()
