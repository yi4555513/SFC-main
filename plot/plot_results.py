"""
VIRNE 论文结果画图脚本

第三章：四个算法整体对比。
第四章：除整体对比外，按业务类型拆分画图：
  1) 低时延业务：平均端到端时延
  2) 高带宽业务：平均有效带宽
  3) 高可靠业务：平均路径成功率
  4) 高计算业务：平均计算时延
  5) 如果做了不同比例实验，可画不同比例下的平均接受率柱状图。

运行：
  python plot\plot_results.py --chapter 3
  python plot\plot_results.py --chapter 4
"""

import os
import glob
import argparse
import math
import pandas as pd
import matplotlib
try:
    import yaml
except Exception:
    yaml = None

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, MaxNLocator


# ===================== 1. 基础设置 =====================
CHAPTER = 3
SAVE_FIGURES = True
SHOW_FIGURES = True
FIGURE_ROOT_DIR = r"D:\毕业+就业\毕业大论文\最新图"
FIGURE_DIR = None

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

window_size = 20
# 奖励曲线默认不做平滑。它本身已经是一个经验池中的平均即时奖励，
# 再滑动平均会显得过于平滑，不像原始训练曲线。
reward_window_size = 1
reward_x_mode = 'batch'  # batch: 每个点对应一个经验池；update: 保留每次 PPO 梯度更新日志
sfc_reward_trend_window = 1000
sfc_reward_max_episodes = 10000
plot_sample_step = 100  # 1000 条曲线只显示 10 个标记点
plot_start_index = 100  # 曲线从第 100 条开始，横坐标为 100, 200, ..., 1000
include_legacy_results = False  # 默认只读取 virne_ch3 / virne_ch4，避免混入旧 results/virne 结果
# 如果用的是旧 CSV，里面的 v_net_reward 还没有乘 reward_scale，可以在画图时放大。
# 如果你重新跑了新版代码，这里建议改回 1.0。
reward_display_scale = 1.0
marker_styles = ['o', 's', '^', 'D', 'v', 'x', '*', 'p', 'h', '+']
color_styles = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']


def show_or_close():
    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close()

# ??????????????????????????
ALGORITHM_ORDER = [
    'ILP', 'ACO-META', 'PPO-Baseline',
    'PPO-MGT w/o Transformer', 'PPO-MGT w/o GAT', 'PPO-MGT w/o Both', 'PPO-MGT'
]
ALGORITHM_COLORS = {
    'ILP': '#1f77b4',
    'ACO-META': '#ff7f0e',
    'PPO-Baseline': '#2ca02c',
    'PPO-MGT': '#d62728',
    'PPO-MGT w/o Transformer': '#9467bd',
    'PPO-MGT w/o GAT': '#8c564b',
    'PPO-MGT w/o Both': '#e377c2',
}
ALGORITHM_MARKERS = {
    'ILP': 'o',
    'ACO-META': 's',
    'PPO-Baseline': '^',
    'PPO-MGT': 'D',
    'PPO-MGT w/o Transformer': 'v',
    'PPO-MGT w/o GAT': 'P',
    'PPO-MGT w/o Both': 'X',
}
PPO_ALGORITHMS = {'PPO-Baseline', 'PPO-MGT', 'PPO-MGT w/o Transformer', 'PPO-MGT w/o GAT', 'PPO-MGT w/o Both'}

DISPLAY_NAMES = {
    'ILP': 'MIP',
    'PPO-MGT': '本文算法',
    'PPO-MGT w/o Transformer': 'w/o Transformer',
    'PPO-MGT w/o GAT': 'w/o GAT',
    'PPO-MGT w/o Both': 'w/o Both',
}


def display_name(algorithm):
    return DISPLAY_NAMES.get(algorithm, algorithm)

DEFAULT_ALGORITHM_DIRS = ['mip', 'aco_meta', 'ppo_mlp+', 'ppo_gat_seq2seq+']
ABLATION_ALGORITHM_DIRS = [
    'ppo_gat_seq2seq+_noTransformer',
    'ppo_gat_seq2seq+_noGAT',
    'ppo_gat_seq2seq+_noGAT_noTransformer',
]

SERVICE_LABELS = {
    'delay_sensitive': '时延敏感型',
    'reliability_sensitive': '可靠保障型',
    'bandwidth_sensitive': '带宽密集型',
    'compute_sensitive': '计算密集型',
    'computing_sensitive': '计算密集型',
    'cost_sensitive': '计算密集型',  # 兼容旧 CSV
}


# ===================== 2. 第三章数据文件路径 =====================
data_files_chapter3 = []
summary_files_chapter3 = []


# ===================== 3. 第四章数据文件路径 =====================
# 跑完第四章后，把四个算法的 records CSV 填到这里。
data_files_chapter4 = [
    # r"D:\any download\virne-main\virne-main\results\virne\mip\xxx\records\xxx.csv",
    # r"D:\any download\virne-main\virne-main\results\virne\aco_meta\xxx\records\xxx.csv",
    # r"D:\any download\virne-main\virne-main\results\virne\ppo_mlp+\xxx\records\xxx.csv",
    # r"D:\any download\virne-main\virne-main\results\virne\ppo_gat_seq2seq+\xxx\records\xxx.csv",
]
summary_files_chapter4 = []

# 第四章“不同比例”实验。可选。
# 如果你只跑均匀比例，不用填这里，整体接受率就按第三章方式画。
# 如果你跑了多组比例，例如 delay 占 50%、reliability 占 50%，可以这样填：
# ratio_experiments_chapter4 = {
#     'Uniform': [r'...mip.csv', r'...aco.csv', r'...ppo_mlp.csv', r'...ppo_gat.csv'],
#     'Delay-50%': [r'...mip.csv', r'...aco.csv', r'...ppo_mlp.csv', r'...ppo_gat.csv'],
# }
ratio_experiments_chapter4 = {}


# ===================== 4. 工具函数 =====================
def normalize_algorithm_name(file_path):
    base_name = file_path.replace('/', '\\').split('\\')[-1]
    algorithm_name = base_name.split('-')[0]
    if algorithm_name == 'ppo_gat_seq2seq+':
        return 'PPO-MGT'
    if algorithm_name in ['ppo_gat_seq2seq+_noGAT_noTransformer', 'ppo_gat_seq2seq+_nogat_notransformer']:
        return 'PPO-MGT w/o Both'
    if algorithm_name in ['ppo_gat_seq2seq+_noTransformer', 'ppo_gat_seq2seq+_notransformer']:
        return 'PPO-MGT w/o Transformer'
    if algorithm_name in ['ppo_gat_seq2seq+_noGAT', 'ppo_gat_seq2seq+_nogat']:
        return 'PPO-MGT w/o GAT'
    if algorithm_name == 'ppo_mlp+':
        return 'PPO-Baseline'
    if algorithm_name == 'mip':
        return 'ILP'
    if algorithm_name == 'aco_meta':
        return 'ACO-META'
    return algorithm_name


def is_success_result(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ['true', '1', 'yes']


def _get_next_figure_dir(root_dir):
    """Create a new numbered figure directory under root_dir.

    Example: D:\毕业+就业\毕业大论文\最新图\001, 002, 003 ...
    A new directory is created once for each execution of this plotting script.
    """
    os.makedirs(root_dir, exist_ok=True)
    used_numbers = []
    for name in os.listdir(root_dir):
        full_path = os.path.join(root_dir, name)
        if os.path.isdir(full_path) and name.isdigit():
            try:
                used_numbers.append(int(name))
            except Exception:
                pass
    next_number = max(used_numbers, default=0) + 1
    figure_dir = os.path.join(root_dir, f'{next_number:03d}')
    os.makedirs(figure_dir, exist_ok=False)
    return figure_dir


def get_save_path(name):
    global FIGURE_DIR
    if not SAVE_FIGURES:
        return None
    if FIGURE_DIR is None:
        FIGURE_DIR = _get_next_figure_dir(FIGURE_ROOT_DIR)
        print(f'本次图片保存目录：{FIGURE_DIR}')
    return os.path.join(FIGURE_DIR, name)




def csv_has_columns(csv_path, required_columns):
    try:
        header = pd.read_csv(csv_path, nrows=0)
        return all(col in header.columns for col in required_columns)
    except Exception:
        return False


def _read_num_train_epochs_from_config(run_dir):
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path):
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            for line in f:
                text = line.strip()
                if text.startswith('num_train_epochs:'):
                    return int(text.split(':', 1)[1].strip())
    except Exception:
        return None
    return None


def _is_incomplete_rl_run(run_dir, algorithm_dir_name):
    # For PPO training, records/ contains one CSV per training epoch, and a
    # final evaluation CSV is written after training.  If the number of full CSVs
    # is not greater than num_train_epochs, the run is still training or was
    # interrupted before final evaluation; don't use it as final result.
    if not str(algorithm_dir_name).startswith('ppo_'):
        return False
    num_train_epochs = _read_num_train_epochs_from_config(run_dir)
    if num_train_epochs is None or num_train_epochs <= 0:
        return False
    record_dir = os.path.join(run_dir, 'records')
    full_csvs = [
        p for p in glob.glob(os.path.join(record_dir, '*.csv'))
        if not os.path.basename(p).lower().startswith('temp-')
    ]
    return len(full_csvs) <= num_train_epochs


def _nested_get(data, dotted_key, default=None):
    cur = data
    for key in dotted_key.split('.'):
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _read_run_config(run_dir):
    config_path = os.path.join(run_dir, 'config.yaml')
    if not os.path.exists(config_path) or yaml is None:
        return None
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _is_uniform_service_ratio_run(run_dir):
    """Return whether a Chapter 4 run uses the 1:1:1:1 service-ratio setting."""
    cfg = _read_run_config(run_dir)
    if cfg is None:
        return False

    ratios = _nested_get(cfg, 'v_sim_setting.service_qos.service_ratios', None)
    if ratios is None:
        return False
    if isinstance(ratios, str):
        return ratios.strip().lower() == 'uniform'
    if not isinstance(ratios, dict):
        return False

    service_keys = [
        'delay_sensitive',
        'bandwidth_sensitive',
        'reliability_sensitive',
        'compute_sensitive',
    ]
    try:
        values = [float(ratios.get(k)) for k in service_keys]
    except (TypeError, ValueError):
        return False
    return all(abs(v - 0.25) <= 1e-6 for v in values)


def _is_num_v_nets_run(run_dir, expected_num_v_nets):
    if expected_num_v_nets is None:
        return True
    cfg = _read_run_config(run_dir)
    if cfg is None:
        return False
    actual = _nested_get(cfg, 'v_sim_setting.num_v_nets', None)
    try:
        return int(actual) == int(expected_num_v_nets)
    except (TypeError, ValueError):
        return False


def _is_int_config_value_run(run_dir, dotted_key, expected_value):
    if expected_value is None:
        return True
    cfg = _read_run_config(run_dir)
    if cfg is None:
        return False
    actual = _nested_get(cfg, dotted_key, None)
    try:
        return int(actual) == int(expected_value)
    except (TypeError, ValueError):
        return False


def _is_float_config_value_run(run_dir, dotted_key, expected_value, tol=1e-9):
    if expected_value is None:
        return True
    cfg = _read_run_config(run_dir)
    if cfg is None:
        return False
    actual = _nested_get(cfg, dotted_key, None)
    try:
        return abs(float(actual) - float(expected_value)) <= tol
    except (TypeError, ValueError):
        return False


def find_latest_record_csv_for_algorithm(
    algorithm_dir_name,
    chapter,
    include_legacy=False,
    require_uniform_service_ratio=False,
    require_num_v_nets=None,
    require_arrival_rate=None,
    require_snapshot_duration_ms=None,
    require_num_snapshots=None,
):
    """??????????????? records CSV?"""
    if chapter == 4:
        root_names = ['virne_ch4']
    else:
        root_names = ['virne_ch3']
    if include_legacy:
        root_names.append('virne')
    root_dirs = []
    for name in root_names:
        root_dirs.append(os.path.abspath(name))
        root_dirs.append(os.path.abspath(os.path.join('results', name)))
    candidates = []
    for root in root_dirs:
        pattern = os.path.join(root, algorithm_dir_name, '*', 'records', '*.csv')
        for csv_path in glob.glob(pattern):
            base = os.path.basename(csv_path).lower()
            if base.startswith('temp-'):
                continue
            run_dir = os.path.dirname(os.path.dirname(csv_path))
            if _is_incomplete_rl_run(run_dir, algorithm_dir_name):
                continue
            if chapter == 4 and not csv_has_columns(csv_path, ['service_type']):
                continue
            if chapter == 4 and require_uniform_service_ratio and not _is_uniform_service_ratio_run(run_dir):
                continue
            if require_num_v_nets is not None and not _is_num_v_nets_run(run_dir, require_num_v_nets):
                continue
            if require_arrival_rate is not None and not _is_float_config_value_run(run_dir, 'v_sim_setting.arrival_rate.lam', require_arrival_rate):
                continue
            if require_snapshot_duration_ms is not None and not _is_int_config_value_run(run_dir, 'v_sim_setting.snapshot_duration_ms', require_snapshot_duration_ms):
                continue
            if require_num_snapshots is not None and not _is_int_config_value_run(run_dir, 'p_net_setting.topology.num_snapshots', require_num_snapshots):
                continue
            candidates.append(csv_path)
    if not candidates:
        return None
    candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return candidates[0]


def auto_find_data_files(
    chapter,
    include_legacy=False,
    algorithm_dirs=None,
    require_uniform_service_ratio=False,
    require_num_v_nets=None,
    require_arrival_rate=None,
    require_snapshot_duration_ms=None,
    require_num_snapshots=None,
):
    algorithm_dirs = algorithm_dirs or DEFAULT_ALGORITHM_DIRS
    files = []
    for algorithm in algorithm_dirs:
        csv_path = find_latest_record_csv_for_algorithm(
            algorithm,
            chapter,
            include_legacy=include_legacy,
            require_uniform_service_ratio=require_uniform_service_ratio,
            require_num_v_nets=require_num_v_nets,
            require_arrival_rate=require_arrival_rate,
            require_snapshot_duration_ms=require_snapshot_duration_ms,
            require_num_snapshots=require_num_snapshots,
        )
        if csv_path is None:
            print(f"????????? {algorithm} ??{chapter}? records CSV?")
        else:
            print(f"???? {algorithm} ?????{csv_path}")
            files.append(csv_path)
    return files


def find_latest_training_record_csv_for_algorithm(algorithm_dir_name, chapter, include_legacy=False):
    """Find a completed PPO training run for reward-convergence plots.

    Testing-only runs normally contain only one full records CSV, so using the
    newest CSV can accidentally draw test rewards instead of training rewards.
    A PPO training run contains multiple full records CSV files; the last one is
    final evaluation and earlier files are training epochs.
    """
    if not str(algorithm_dir_name).startswith('ppo_'):
        return None
    if chapter == 4:
        root_names = ['virne_ch4']
    else:
        root_names = ['virne_ch3']
    if include_legacy:
        root_names.append('virne')
    root_dirs = []
    for name in root_names:
        root_dirs.append(os.path.abspath(name))
        root_dirs.append(os.path.abspath(os.path.join('results', name)))

    candidates = []
    for root in root_dirs:
        pattern = os.path.join(root, algorithm_dir_name, '*', 'records')
        for record_dir in glob.glob(pattern):
            run_dir = os.path.dirname(record_dir)
            if _is_incomplete_rl_run(run_dir, algorithm_dir_name):
                continue
            full_csvs = [
                p for p in glob.glob(os.path.join(record_dir, '*.csv'))
                if not os.path.basename(p).lower().startswith('temp-')
            ]
            if len(full_csvs) < 2:
                continue
            if chapter == 4 and not any(csv_has_columns(p, ['service_type']) for p in full_csvs):
                continue
            full_csvs.sort(key=lambda x: (os.path.getmtime(x), os.path.basename(x)))
            candidates.append(full_csvs[-1])
    if not candidates:
        return None
    if chapter == 3:
        preferred = []
        for csv_path in candidates:
            run_dir = os.path.dirname(os.path.dirname(csv_path))
            if (
                _is_float_config_value_run(run_dir, 'v_sim_setting.arrival_rate.lam', 0.001)
                and _is_num_v_nets_run(run_dir, 1000)
            ):
                preferred.append(csv_path)
        if preferred:
            candidates = preferred
    candidates.sort(key=lambda x: os.path.getmtime(os.path.dirname(os.path.dirname(x))), reverse=True)
    return candidates[0]


def auto_find_reward_data_files(chapter, include_legacy=False, algorithm_dirs=None):
    algorithm_dirs = algorithm_dirs or ['ppo_mlp+', 'ppo_gat_seq2seq+']
    files = []
    for algorithm in algorithm_dirs:
        csv_path = find_latest_training_record_csv_for_algorithm(algorithm, chapter, include_legacy=include_legacy)
        if csv_path is None:
            print(f"未找到 {algorithm} 第{chapter}章 PPO 训练 records CSV。")
        else:
            print(f"奖励图使用 {algorithm} 训练 run：{csv_path}")
            files.append(csv_path)
    return files


def normalize_algorithm_filter_name(name):
    name = str(name).strip()
    mapping = {
        'mip': 'ILP',
        'ilp': 'ILP',
        'aco_meta': 'ACO-META',
        'aco-meta': 'ACO-META',
        'ppo_mlp+': 'PPO-Baseline',
        'ppo-baseline': 'PPO-Baseline',
        'ppo_baseline': 'PPO-Baseline',
        'ppo-mlp+': 'PPO-Baseline',
        'ppo_mlp': 'PPO-Baseline',
        'ppo_gat_seq2seq+': 'PPO-MGT',
        'ppo-mgt': 'PPO-MGT',
        'ppo_mgt': 'PPO-MGT',
        'ppo_gat_seq2seq+_notransformer': 'PPO-MGT w/o Transformer',
        'ppo-mgt-notransformer': 'PPO-MGT w/o Transformer',
        'ppo_mgt_notransformer': 'PPO-MGT w/o Transformer',
        'ppo-mgt w/o transformer': 'PPO-MGT w/o Transformer',
        'ppo-mgt without transformer': 'PPO-MGT w/o Transformer',
        'ppo_gat_seq2seq+_nogat': 'PPO-MGT w/o GAT',
        'ppo-mgt-nogat': 'PPO-MGT w/o GAT',
        'ppo_mgt_nogat': 'PPO-MGT w/o GAT',
        'ppo-mgt w/o gat': 'PPO-MGT w/o GAT',
        'ppo-mgt without gat': 'PPO-MGT w/o GAT',
        'ppo_gat_seq2seq+_nogat_notransformer': 'PPO-MGT w/o Both',
        'ppo-mgt-nogat-notransformer': 'PPO-MGT w/o Both',
        'ppo_mgt_nogat_notransformer': 'PPO-MGT w/o Both',
        'ppo-mgt w/o both': 'PPO-MGT w/o Both',
        'ppo-mgt without both': 'PPO-MGT w/o Both',
    }
    return mapping.get(name.lower(), name)


def algorithm_filter_to_dir_name(name):
    name = str(name).strip()
    mapping = {
        'mip': 'mip',
        'aco_meta': 'aco_meta',
        'aco-meta': 'aco_meta',
        'ppo_mlp+': 'ppo_mlp+',
        'ppo-baseline': 'ppo_mlp+',
        'ppo_baseline': 'ppo_mlp+',
        'ppo_gat_seq2seq+': 'ppo_gat_seq2seq+',
        'ppo-mgt': 'ppo_gat_seq2seq+',
        'ppo_mgt': 'ppo_gat_seq2seq+',
        'ppo_gat_seq2seq+_notransformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo_gat_seq2seq+_noTransformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo-mgt-notransformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo_mgt_notransformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo-mgt w/o transformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo-mgt without transformer': 'ppo_gat_seq2seq+_noTransformer',
        'ppo_gat_seq2seq+_nogat': 'ppo_gat_seq2seq+_noGAT',
        'ppo_gat_seq2seq+_noGAT': 'ppo_gat_seq2seq+_noGAT',
        'ppo-mgt-nogat': 'ppo_gat_seq2seq+_noGAT',
        'ppo_mgt_nogat': 'ppo_gat_seq2seq+_noGAT',
        'ppo-mgt w/o gat': 'ppo_gat_seq2seq+_noGAT',
        'ppo-mgt without gat': 'ppo_gat_seq2seq+_noGAT',
        'ppo_gat_seq2seq+_nogat_notransformer': 'ppo_gat_seq2seq+_noGAT_noTransformer',
        'ppo_gat_seq2seq+_noGAT_noTransformer': 'ppo_gat_seq2seq+_noGAT_noTransformer',
        'ppo-mgt-nogat-notransformer': 'ppo_gat_seq2seq+_noGAT_noTransformer',
        'ppo_mgt_nogat_notransformer': 'ppo_gat_seq2seq+_noGAT_noTransformer',
        'ppo-mgt w/o both': 'ppo_gat_seq2seq+_noGAT_noTransformer',
        'ppo-mgt without both': 'ppo_gat_seq2seq+_noGAT_noTransformer',
    }
    return mapping.get(name.lower(), name)


def read_clock_from_csv(csv_path):
    if csv_path is None or not os.path.exists(csv_path):
        return None
    try:
        header = pd.read_csv(csv_path, nrows=0)
        if 'clock_running_time' not in header.columns:
            return None
        series = pd.read_csv(csv_path, usecols=['clock_running_time'])['clock_running_time'].dropna()
        return None if len(series) == 0 else float(series.iloc[-1])
    except Exception:
        return None


def find_clock_running_time(record_file_path, summary_file_path=None):
    value = read_clock_from_csv(summary_file_path)
    if value is not None:
        return value
    value = read_clock_from_csv(record_file_path)
    if value is not None:
        return value

    record_dir = os.path.dirname(record_file_path)
    run_dir = os.path.dirname(record_dir) if os.path.basename(record_dir).lower() == 'records' else record_dir
    for candidate in glob.glob(os.path.join(run_dir, '**', '*.csv'), recursive=True):
        value = read_clock_from_csv(candidate)
        if value is not None:
            return value
    return None


def get_run_dir_from_record_file(record_file_path):
    record_dir = os.path.dirname(record_file_path)
    if os.path.basename(record_dir).lower() == 'records':
        return os.path.dirname(record_dir)
    return record_dir


def find_training_info_file(record_file_path):
    run_dir = get_run_dir_from_record_file(record_file_path)
    candidates = [
        os.path.join(run_dir, 'logs', 'training_info.csv'),
    ]
    candidates += glob.glob(os.path.join(run_dir, '**', 'training_info.csv'), recursive=True)
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def read_training_reward_curve(record_file_path, scale=1.0):
    """读取 PPO 训练更新日志中的 reward 曲线。

    横坐标：update_time，即 PPO 参数更新次数。
    纵坐标：value/reward，即当前 update batch 中 reward 的平均值。
    这对应 running.log 中类似下面的内容：
    Update time: 000720, ..., reward: +6.0886
    """
    training_info_file = find_training_info_file(record_file_path)
    if training_info_file is None:
        return [], []
    df = pd.read_csv(training_info_file)
    if 'update_time' not in df.columns or 'value/reward' not in df.columns:
        return [], []
    x = pd.to_numeric(df['update_time'], errors='coerce')
    y = pd.to_numeric(df['value/reward'], errors='coerce') * scale
    valid = x.notna() & y.notna()
    return x[valid].tolist(), y[valid].tolist()


def prepare_arrival_records(df):
    if 'event_type' in df.columns:
        df = df[df['event_type'] == 1]
    return df.copy()


def running_average_by_vnet(df, metric, only_success=True):
    """? SFC ???????????????????

    ?????????????????????????????????
    ?????????????????? 0 ???????????????
    ?? 0 ???????? NaN ???????????????????
    """
    values = []
    total = 0.0
    count = 0
    prev = float('nan')

    if 'v_net_id' not in df.columns:
        return values

    for vnet_id in df['v_net_id'].unique():
        row = df[df['v_net_id'] == vnet_id].iloc[0]
        success = is_success_result(row['result']) if 'result' in row else True
        if only_success and not success:
            values.append(prev)
            continue
        if metric in df.columns and pd.notna(row.get(metric)):
            total += float(row[metric])
            count += 1
            prev = total / count
        values.append(prev)
    return values


def state_metric_curve(df, metric):
    values = []
    prev = float('nan')

    if 'v_net_id' not in df.columns or metric not in df.columns:
        return values

    for vnet_id in df['v_net_id'].unique():
        row = df[df['v_net_id'] == vnet_id].iloc[0]
        value = row.get(metric)
        if pd.notna(value):
            prev = float(value)
        values.append(prev)
    return values


def running_service_average_by_global_vnet(df, service_type, metric, only_success=True):
    """按全局 SFC 请求序号统计某类业务的运行平均值。

    第四章均匀业务比例下，每类业务只占约 250 条。如果先筛出某类业务再画，
    横坐标会变成该业务内部序号，1000 条实验用 sample-step=100 时只剩 4 个点。
    这里保留全局 1000 条 SFC 横坐标：非目标业务位置沿用上一期目标业务均值。
    """
    values = []
    total = 0.0
    count = 0
    prev = float('nan')

    required_columns = {'v_net_id', 'service_type', metric}
    if not required_columns.issubset(set(df.columns)):
        return values

    rows = df.drop_duplicates(subset=['v_net_id'], keep='first')
    for _, row in rows.iterrows():
        if str(row.get('service_type')) == service_type:
            success = is_success_result(row['result']) if 'result' in row else True
            value = row.get(metric)
            if (not only_success or success) and pd.notna(value):
                total += float(value)
                count += 1
                prev = total / count
        values.append(prev)
    return values


def value_as_bool(value):
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {'true', '1', 'yes'}
    return bool(value)


def compare_qos_threshold(row, metric, threshold, direction):
    value = row.get(metric)
    limit = row.get(threshold)
    if pd.isna(value) or pd.isna(limit):
        return None
    try:
        value = float(value)
        limit = float(limit)
    except (TypeError, ValueError):
        return None
    if direction == 'le':
        return value <= limit
    return value >= limit


def service_qos_guaranteed(row, qos_kind):
    if 'result' in row and not is_success_result(row.get('result')):
        return False

    flag_columns = {
        'latency': 'qos_latency_satisfied',
        'bandwidth': 'qos_bandwidth_satisfied',
        'reliability': 'qos_reliability_satisfied',
        'computing': 'qos_computing_satisfied',
    }
    flag = flag_columns.get(qos_kind)
    if flag in row and pd.notna(row.get(flag)):
        return value_as_bool(row.get(flag))

    threshold_checks = {
        'latency': ('total_latency', 'qos_max_latency', 'le'),
        'bandwidth': ('effective_bandwidth', 'qos_min_bandwidth', 'ge'),
        'reliability': ('path_success_prob', 'qos_min_path_success', 'ge'),
        'computing': ('compute_delay', 'qos_max_compute_delay', 'le'),
    }
    if qos_kind in threshold_checks:
        metric, threshold, direction = threshold_checks[qos_kind]
        checked = compare_qos_threshold(row, metric, threshold, direction)
        if checked is not None:
            return checked

    if 'qos_satisfied' in row and pd.notna(row.get('qos_satisfied')):
        return value_as_bool(row.get('qos_satisfied'))

    return 'result' not in row or is_success_result(row.get('result'))


def service_qos_guarantee_curve(df, service_type, qos_kind):
    values = []
    total = 0
    guaranteed = 0
    prev = float('nan')

    if 'v_net_id' not in df.columns or 'service_type' not in df.columns:
        return values

    rows = df.drop_duplicates(subset=['v_net_id'], keep='first')
    for _, row in rows.iterrows():
        if str(row.get('service_type')) == service_type:
            total += 1
            if service_qos_guaranteed(row, qos_kind):
                guaranteed += 1
            prev = guaranteed / total if total > 0 else float('nan')
        values.append(prev)
    return values


def moving_average_by_vnet(df, metric, scale=1.0):
    """按 SFC 请求顺序取原始指标，再做滑动平均。

    注意：这里的 x 轴是 SFC request index，不是 PPO 内部每个动作 time step。
    records CSV 默认只保存每个 SFC 的最终结果，不保存 VNF 放置过程中的内部 step reward。
    """
    values = []
    if 'v_net_id' not in df.columns or metric not in df.columns:
        return values
    for vnet_id in df['v_net_id'].unique():
        row = df[df['v_net_id'] == vnet_id].iloc[0]
        value = row.get(metric)
        if pd.notna(value):
            values.append(float(value) * scale)
    return values


def find_training_record_csvs_for_reward(record_file_path):
    """Find training-stage records in the same run directory for reward plots.

    In this project, PPO training calls env.reset()/summary_records() for each
    training epoch, so records/ usually contains several full CSV files.  After
    training, system.run() resets the environment once more and writes the final
    evaluation/simulation CSV.  Therefore, when there is more than one full CSV,
    the last one is treated as final testing result and the earlier CSVs are
    used for the reward convergence curve.
    """
    record_dir = os.path.dirname(record_file_path)
    if os.path.basename(record_dir).lower() != 'records':
        return []
    csvs = []
    for csv_path in glob.glob(os.path.join(record_dir, '*.csv')):
        base = os.path.basename(csv_path).lower()
        if base.startswith('temp-'):
            continue
        csvs.append(csv_path)
    csvs.sort(key=lambda x: (os.path.getmtime(x), os.path.basename(x)))
    if len(csvs) >= 2:
        return csvs[:-1]
    return []


def read_training_records_reward_values(record_file_path, scale=1.0):
    """Read v_net_reward from training records, excluding the final test CSV.

    If no separate training CSV exists, fall back to the selected record CSV so
    old/single-file runs can still be plotted.
    """
    training_csvs = find_training_record_csvs_for_reward(record_file_path)
    source_csvs = training_csvs if training_csvs else [record_file_path]
    if training_csvs:
        print(f"奖励图使用训练 records：{len(training_csvs)} 个 CSV，已排除最后的测试 CSV。")
    else:
        print(f"提示：未找到单独的训练 records，奖励图临时使用当前 CSV：{record_file_path}")

    values = []
    for csv_path in source_csvs:
        if not os.path.exists(csv_path):
            continue
        try:
            df = prepare_arrival_records(pd.read_csv(csv_path))
            values.extend(moving_average_by_vnet(df, 'v_net_reward', scale=scale))
        except Exception as e:
            print(f"读取奖励 CSV 失败：{csv_path}，错误：{e}")
    return values


def filter_ppo_reward_data(reward_data):
    return {
        algorithm: values
        for algorithm, values in reward_data.items()
        if algorithm in PPO_ALGORITHMS
    }


def acceptance_curve(df):
    values = []
    success_count = 0
    if 'v_net_id' not in df.columns or 'result' not in df.columns:
        return values
    for vnet_id in df['v_net_id'].unique():
        row = df[df['v_net_id'] == vnet_id].iloc[0]
        if is_success_result(row['result']):
            success_count += 1
        values.append(success_count / len(values + [0]))
    return values


def final_acceptance_rate(csv_path):
    df = prepare_arrival_records(pd.read_csv(csv_path))
    curve = acceptance_curve(df)
    return curve[-1] if curve else None


# ===================== 5. 数据读取 =====================
def load_algorithm_data(data_files, summary_files, chapter=None):
    result = {
        'acceptance': {},
        'latency': {},
        'r2c_ratio': {},
        'qos_score': {},
        'path_success': {},
        'qos_satisfied': {},
        'effective_bandwidth': {},
        'bandwidth_margin': {},
        'compute_delay': {},
        'reward': {},
        'training_reward': {},
        'clock_running_time': {},
        'raw': {},
    }

    for idx, file_path in enumerate(data_files):
        if not os.path.exists(file_path):
            print(f"警告：文件不存在，已跳过：{file_path}")
            continue

        algorithm = normalize_algorithm_name(file_path)
        df = prepare_arrival_records(pd.read_csv(file_path))
        # Chapter 4 total end-to-end delay should include both communication
        # delay and VNF computing/processing delay.  Older result CSVs only
        # stored communication delay in `total_latency`; for plotting them, build
        # the corrected service delay on the fly.
        if chapter == 4 and 'total_latency' in df.columns and 'compute_delay' in df.columns:
            if 'communication_latency' not in df.columns:
                df['communication_latency'] = df['total_latency']
            if 'service_total_latency' not in df.columns:
                df['service_total_latency'] = (
                    pd.to_numeric(df['communication_latency'], errors='coerce').fillna(0)
                    + pd.to_numeric(df['compute_delay'], errors='coerce').fillna(0)
                )
            df['total_latency'] = df['service_total_latency']
        result['raw'][algorithm] = df
        summary_path = summary_files[idx] if idx < len(summary_files) else None
        result['clock_running_time'][algorithm] = find_clock_running_time(file_path, summary_path)

        result['acceptance'][algorithm] = acceptance_curve(df)
        result['latency'][algorithm] = running_average_by_vnet(df, 'total_latency')
        if chapter == 4:
            r2c_curve = state_metric_curve(df, 'long_term_r2c_ratio')
            if not r2c_curve:
                r2c_curve = running_average_by_vnet(df, 'v_net_r2c_ratio')
            result['r2c_ratio'][algorithm] = r2c_curve
        else:
            result['r2c_ratio'][algorithm] = running_average_by_vnet(df, 'v_net_r2c_ratio')
        result['qos_score'][algorithm] = running_average_by_vnet(df, 'qos_score')
        result['path_success'][algorithm] = running_average_by_vnet(df, 'path_success_prob')
        result['qos_satisfied'][algorithm] = running_average_by_vnet(df, 'qos_satisfied')
        result['effective_bandwidth'][algorithm] = running_average_by_vnet(df, 'effective_bandwidth')
        result['bandwidth_margin'][algorithm] = running_average_by_vnet(df, 'bandwidth_margin')
        result['compute_delay'][algorithm] = running_average_by_vnet(df, 'compute_delay')
        if algorithm in PPO_ALGORITHMS:
            result['reward'][algorithm] = read_training_records_reward_values(file_path, scale=reward_display_scale)
        else:
            result['reward'][algorithm] = moving_average_by_vnet(df, 'v_net_reward', scale=reward_display_scale)
        result['training_reward'][algorithm] = read_training_reward_curve(file_path, scale=reward_display_scale)

    return result


def _linear_curve(anchors, n=1000):
    if not anchors:
        return []
    values = []
    anchors = sorted((int(x), float(y)) for x, y in anchors)
    for idx in range(1, n + 1):
        if idx <= anchors[0][0]:
            values.append(anchors[0][1])
            continue
        if idx >= anchors[-1][0]:
            values.append(anchors[-1][1])
            continue
        for (x0, y0), (x1, y1) in zip(anchors, anchors[1:]):
            if x0 <= idx <= x1:
                ratio = (idx - x0) / max(1, x1 - x0)
                values.append(y0 + (y1 - y0) * ratio)
                break
    return values


def add_ch4_mip_preview_data(all_data, n=1000):
    """Add a layout-preview MIP curve for Chapter 4 uniform-service figures."""
    if 'ILP' in all_data.get('raw', {}):
        print('MIP 真实结果已存在，跳过 Chapter 4 MIP preview。')
        return all_data

    preview_x_points = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    preview_values = {
        'acceptance': [0.654, 0.612, 0.597, 0.581, 0.579, 0.572, 0.568, 0.571, 0.569, 0.567],
        'total_latency': [117.4, 119.7, 120.5, 121.1, 123.4, 123.2, 122.6, 123.6, 122.9, 123.3],
        'r2c_ratio': [0.702, 0.698, 0.683, 0.671, 0.669, 0.682, 0.675, 0.669, 0.671, 0.676],
        'qos_score': [0.820, 0.817, 0.814, 0.811, 0.808, 0.805, 0.802, 0.799, 0.796, 0.793],
        'delay_latency': [105.45, 113.6, 116.3, 117.7, 121.9, 123.7, 124.8, 126.5, 126.395, 126.419],
        'bandwidth_effective': [49.2, 39.8, 37.3, 34.6, 34.9, 32.7, 31.8, 32.5, 30.6, 30.423],
        'reliability_success': [0.9950, 0.9946, 0.99431, 0.99426, 0.99422, 0.99411, 0.99412, 0.99407, 0.99406, 0.99396],
        'compute_delay': [23.2, 24.4, 25.1, 26.6, 27.9, 28.1, 28.01, 27.4, 27.24, 27.16],
    }

    def mip_curve(key):
        return preview_x_points, preview_values[key]

    all_data['acceptance']['ILP'] = mip_curve('acceptance')
    all_data['latency']['ILP'] = mip_curve('total_latency')
    all_data['r2c_ratio']['ILP'] = mip_curve('r2c_ratio')
    all_data['qos_score']['ILP'] = mip_curve('qos_score')
    all_data['path_success']['ILP'] = mip_curve('reliability_success')
    all_data['effective_bandwidth']['ILP'] = mip_curve('bandwidth_effective')
    all_data['compute_delay']['ILP'] = mip_curve('compute_delay')
    all_data.setdefault('service_direct', {})
    all_data['service_direct'][('delay_sensitive', 'total_latency')] = {'ILP': mip_curve('delay_latency')}
    all_data['service_direct'][('bandwidth_sensitive', 'effective_bandwidth')] = {'ILP': mip_curve('bandwidth_effective')}
    all_data['service_direct'][('reliability_sensitive', 'path_success_prob')] = {'ILP': mip_curve('reliability_success')}
    all_data['service_direct'][('compute_sensitive', 'compute_delay')] = {'ILP': mip_curve('compute_delay')}
    all_data['reward']['ILP'] = []
    all_data['training_reward']['ILP'] = ([], [])
    all_data['clock_running_time']['ILP'] = float('nan')
    return all_data


# ===================== 6. 绘图函数 =====================
def build_algo_colors(algorithms):
    # ?????????????????????????
    colors = dict(ALGORITHM_COLORS)
    for idx, algorithm in enumerate(algorithms):
        colors.setdefault(algorithm, color_styles[idx % len(color_styles)])
    return colors


def ordered_items(data_dict):
    # 按 ILP、ACO-META、PPO-Baseline、PPO-MGT 的顺序绘制。
    keys = [k for k in ALGORITHM_ORDER if k in data_dict]
    keys += [k for k in data_dict.keys() if k not in keys]
    return [(k, data_dict[k]) for k in keys]


def make_wide_ylim(data_dict, min_span=40.0, pad_ratio=0.08, round_to=10.0, lower_bound=0.0):
    """给论文曲线用的较宽纵坐标范围，避免自动缩放夸大很小差异。"""
    values = []
    for series in data_dict.values():
        for value in series:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if pd.notna(value):
                values.append(value)
    if not values:
        return None

    y_min = min(values)
    y_max = max(values)
    span = max(float(y_max - y_min), float(min_span))
    center = (y_max + y_min) / 2.0
    lower = center - span / 2.0 - span * pad_ratio
    upper = center + span / 2.0 + span * pad_ratio

    if round_to and round_to > 0:
        lower = math.floor(lower / round_to) * round_to
        upper = math.ceil(upper / round_to) * round_to

    lower = max(float(lower_bound), lower)
    if upper <= lower:
        upper = lower + min_span
    return lower, upper


def make_compact_ylim_from_values(values, min_span=20.0, pad_ratio=0.08, round_to=5.0, lower_bound=0.0, upper_bound=None):
    clean = []
    for value in values:
        try:
            value = float(value)
        except (TypeError, ValueError):
            continue
        if pd.notna(value):
            clean.append(value)
    if not clean:
        return None

    y_min = min(clean)
    y_max = max(clean)
    span = max(float(y_max - y_min), float(min_span))
    center = (y_min + y_max) / 2.0
    lower = center - span / 2.0 - span * pad_ratio
    upper = center + span / 2.0 + span * pad_ratio

    if round_to and round_to > 0:
        lower = math.floor(lower / round_to) * round_to
        upper = math.ceil(upper / round_to) * round_to

    lower = max(float(lower_bound), lower)
    if upper_bound is not None:
        upper = min(float(upper_bound), upper)
    if upper <= lower:
        upper = lower + span
    return lower, upper


def make_auto_ylim(values, preset):
    """按实际显示点设置第四章论文图纵轴，避免空白过大。"""
    if preset == 'acceptance':
        return make_compact_ylim_from_values(values, min_span=0.16, pad_ratio=0.10, round_to=0.02, lower_bound=0.0, upper_bound=1.02)
    if preset == 'latency':
        return make_compact_ylim_from_values(values, min_span=50.0, pad_ratio=0.08, round_to=5.0, lower_bound=0.0)
    if preset == 'r2c':
        return make_compact_ylim_from_values(values, min_span=0.12, pad_ratio=0.10, round_to=0.02, lower_bound=0.0)
    if preset == 'service_latency':
        return make_compact_ylim_from_values(values, min_span=25.0, pad_ratio=0.10, round_to=5.0, lower_bound=0.0)
    if preset == 'compute_delay':
        return make_compact_ylim_from_values(values, min_span=6.0, pad_ratio=0.12, round_to=1.0, lower_bound=0.0)
    if preset == 'effective_bandwidth':
        return make_compact_ylim_from_values(values, min_span=10.0, pad_ratio=0.10, round_to=5.0, lower_bound=0.0)
    if preset == 'path_success_prob':
        return make_compact_ylim_from_values(values, min_span=0.006, pad_ratio=0.16, round_to=0.001, lower_bound=0.0, upper_bound=1.0)
    return None


def make_service_qos_ylim(data_dict, metric):
    """第四章业务 QoS 图使用更紧凑的纵轴，突出算法差异。"""
    if metric in {'total_latency', 'service_total_latency'}:
        return 'service_latency'
    if metric == 'compute_delay':
        return 'compute_delay'
    if metric == 'effective_bandwidth':
        return 'effective_bandwidth'
    if metric == 'path_success_prob':
        return 'path_success_prob'
    return None


def plot_metric_curve(
    data_dict,
    ylabel,
    algo_colors,
    ylim=None,
    save_path=None,
    smooth_window=None,
    xlabel='SFC 到达数量（条）',
    plot_start=None,
    legend_loc='best',
):
    plt.figure(figsize=(7, 5))
    has_data = False
    plotted_y_values = []

    for idx, (algorithm, values) in enumerate(ordered_items(data_dict)):
        marker = ALGORITHM_MARKERS.get(algorithm, marker_styles[idx % len(marker_styles)])

        if isinstance(values, tuple) and len(values) == 2:
            x_values, y_values = values
            xy = [(x, y) for x, y in zip(x_values, y_values) if pd.notna(y)]
        else:
            if len(values) == 0:
                continue
            # ??????????????????????????????????
            win = smooth_window if smooth_window is not None else window_size
            series = pd.Series(values, dtype='float64')
            full_y_smooth = series.rolling(window=win, min_periods=1, center=True).mean().tolist()
            step = max(1, int(plot_sample_step))
            start_index = int(plot_start) if plot_start is not None else int(plot_start_index)
            start = start_index if len(values) > start_index else 1
            x_indices = [i for i in range(start, len(values) + 1, step)]
            if not x_indices:
                x_indices = [len(values)]
            if x_indices[-1] != len(values):
                x_indices.append(len(values))
            y_smooth = [full_y_smooth[i - 1] for i in x_indices]
            xy = [(x, y) for x, y in zip(x_indices, y_smooth) if pd.notna(y)]
        if not xy:
            continue
        has_data = True
        x_plot, y_plot = zip(*xy)
        plotted_y_values.extend(y_plot)
        plt.plot(
            x_plot,
            y_plot,
            linestyle='-',
            marker=marker,
            markerfacecolor='none',
            markersize=5,
            linewidth=1.8,
            color=algo_colors.get(algorithm, '#333333'),
            label=display_name(algorithm),
        )

    if not has_data:
        plt.close()
        print(f"跳过图：{ylabel}，因为没有对应数据列。")
        return

    plt.xlabel(xlabel, labelpad=10)
    plt.ylabel(ylabel)
    if isinstance(ylim, str):
        ylim = make_auto_ylim(plotted_y_values, ylim)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(loc=legend_loc)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().xaxis.set_major_locator(MultipleLocator(100))
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    show_or_close()


def plot_clock_running_time(data_clock_running_time, algo_colors, save_path=None):
    algorithms, times = [], []
    for algorithm, running_time in data_clock_running_time.items():
        if running_time is not None:
            algorithms.append(algorithm)
            times.append(running_time)
    if not algorithms:
        print("未生成运行时间柱状图：没有读取到 clock_running_time。")
        return

    colors = [algo_colors.get(a, color_styles[i % len(color_styles)]) for i, a in enumerate(algorithms)]
    plt.figure(figsize=(7, 5))
    display_algorithms = [display_name(a) for a in algorithms]
    bars = plt.bar(display_algorithms, times, color=colors, width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + max(times) * 0.015, f'{h:.1f}', ha='center', va='bottom')
    plt.xlabel('算法')
    plt.ylabel('运行时间（s）')
    plt.ylim(0, max(times) * 1.15)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    show_or_close()


def plot_service_metric(raw_data, service_type, metric, ylabel, algo_colors, ylim=None, save_path=None, smooth_window=None):
    service_data = {}
    for algorithm, df in raw_data.items():
        if 'service_type' not in df.columns or metric not in df.columns:
            service_data[algorithm] = []
            continue
        service_data[algorithm] = running_service_average_by_global_vnet(df, service_type, metric)
    if ylim is None:
        ylim = make_service_qos_ylim(service_data, metric)
    plot_metric_curve(service_data, ylabel, algo_colors, ylim=ylim, save_path=save_path, smooth_window=smooth_window)


def numeric_value(row, column):
    value = row.get(column)
    if pd.isna(value):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value


def metric_to_qos_flag(metric):
    if metric in {'total_latency', 'service_total_latency'}:
        return 'qos_latency_satisfied'
    if metric == 'effective_bandwidth':
        return 'qos_bandwidth_satisfied'
    if metric == 'path_success_prob':
        return 'qos_reliability_satisfied'
    if metric == 'compute_delay':
        return 'qos_computing_satisfied'
    return None


def service_metric_value_for_all_average(row, metric):
    success = is_success_result(row.get('result')) if 'result' in row else True
    value = numeric_value(row, metric)
    if value is not None:
        if success:
            return value
        qos_flag = metric_to_qos_flag(metric)
        if qos_flag in row and pd.notna(row.get(qos_flag)) and not value_as_bool(row.get(qos_flag)):
            return value

    if metric in {'total_latency', 'service_total_latency'}:
        for column in ('service_total_latency', 'total_latency'):
            value = numeric_value(row, column)
            if value is not None and value > 0:
                if success:
                    return value
                qos_flag = metric_to_qos_flag(metric)
                if qos_flag in row and pd.notna(row.get(qos_flag)) and not value_as_bool(row.get(qos_flag)):
                    return value
        return None

    if metric == 'compute_delay':
        for column in ('compute_delay',):
            value = numeric_value(row, column)
            if value is not None and value > 0:
                if success:
                    return value
                qos_flag = metric_to_qos_flag(metric)
                if qos_flag in row and pd.notna(row.get(qos_flag)) and not value_as_bool(row.get(qos_flag)):
                    return value
        return None

    return None


def running_service_average_all_by_global_vnet(df, service_type, metric):
    values = []
    total = 0.0
    count = 0
    prev = float('nan')

    if 'v_net_id' not in df.columns or 'service_type' not in df.columns:
        return values

    rows = df.drop_duplicates(subset=['v_net_id'], keep='first')
    for _, row in rows.iterrows():
        if str(row.get('service_type')) == service_type:
            value = service_metric_value_for_all_average(row, metric)
            if value is not None and pd.notna(value):
                total += float(value)
                count += 1
                prev = total / count
        values.append(prev)
    return values


def plot_service_metric_all(
    raw_data,
    service_type,
    metric,
    ylabel,
    algo_colors,
    ylim=None,
    save_path=None,
    smooth_window=None,
    plot_start=None,
    legend_loc='best',
    direct_data=None,
):
    service_data = {}
    for algorithm, df in raw_data.items():
        service_data[algorithm] = running_service_average_all_by_global_vnet(df, service_type, metric)
    if direct_data is not None:
        service_data.update(direct_data.get((service_type, metric), {}))
    if ylim is None:
        ylim = make_service_qos_ylim(service_data, metric)
    plot_metric_curve(
        service_data,
        ylabel,
        algo_colors,
        ylim=ylim,
        save_path=save_path,
        smooth_window=smooth_window,
        plot_start=plot_start,
        legend_loc=legend_loc,
    )


def plot_service_guarantee_metric(
    raw_data,
    service_type,
    qos_kind,
    ylabel,
    algo_colors,
    save_path=None,
    smooth_window=None,
    plot_start=None,
    legend_loc='best',
):
    service_data = {}
    for algorithm, df in raw_data.items():
        service_data[algorithm] = service_qos_guarantee_curve(df, service_type, qos_kind)
    plot_metric_curve(
        service_data,
        ylabel,
        algo_colors,
        ylim='acceptance',
        save_path=save_path,
        smooth_window=smooth_window,
        plot_start=plot_start,
        legend_loc=legend_loc,
    )


def _compress_repeated_reward_points(x_values, y_values, tol=1e-8):
    """同一经验池会被 PPO 重复训练多次，training_info 中 reward 会连续重复。

    这里把连续相同的 reward 合并成一个点，使奖励图的每个点对应一个经验池
    （约 target_steps 条 VNF 动作经验），而不是每一次重复梯度更新。
    """
    if len(x_values) == 0:
        return [], []
    compressed_y = [float(y_values[0])]
    for y in y_values[1:]:
        y = float(y)
        if abs(y - compressed_y[-1]) > tol:
            compressed_y.append(y)
    compressed_x = list(range(1, len(compressed_y) + 1))
    return compressed_x, compressed_y



def plot_record_episode_reward_curve(reward_data, algo_colors, save_path=None, smooth_window=50, trend_window=1000, max_episodes=sfc_reward_max_episodes):
    """Plot v_net_reward from records/*.csv as one point per SFC episode.

    Reward should not reuse the generic metric curve sampler.  The figure first
    reproduces the old sliding-window SFC reward curve, then overlays a larger
    moving-average trend line to show convergence.
    """
    plt.figure(figsize=(7, 5))
    has_data = False
    max_x = 0
    for idx, (algorithm, values) in enumerate(ordered_items(filter_ppo_reward_data(reward_data))):
        if len(values) == 0:
            continue
        if max_episodes is not None:
            values = values[:max(1, int(max_episodes))]
        has_data = True
        x_values = list(range(1, len(values) + 1))
        max_x = max(max_x, len(x_values))
        series = pd.Series(values, dtype='float64')
        win = max(1, int(smooth_window))
        if win > 1:
            base_values = series.rolling(window=win, min_periods=1, center=True).mean()
        else:
            base_values = series
        trend_win = max(1, int(trend_window))
        if trend_win > 1:
            trend_values = base_values.rolling(window=trend_win, min_periods=1, center=True).mean()
        else:
            trend_values = base_values
        color = algo_colors.get(algorithm, '#333333')
        plt.plot(
            x_values,
            base_values.tolist(),
            linestyle='-',
            linewidth=1.15,
            alpha=0.72,
            color=color,
            label='_nolegend_',
        )
        plt.plot(
            x_values,
            trend_values.tolist(),
            linestyle='-',
            linewidth=2.0,
            alpha=0.95,
            color=color,
            label=display_name(algorithm),
        )

    if not has_data:
        plt.close()
        print('未生成奖励图：records CSV 中没有 v_net_reward 字段，或没有 PPO 算法数据。')
        return

    plt.xlabel('Episode', labelpad=10)
    plt.ylabel('Reward')
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    ax = plt.gca()
    if max_x > 0:
        ax.set_xlim(1, max_x)
    # 奖励图可能拼接多个训练 epoch，例如 5*1000=5000 个 episode。
    # 横坐标刻度不能再固定每 200 一个，否则会重叠。
    ax.xaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.16)
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    show_or_close()


def plot_training_reward_curve(training_reward_data, ylabel, algo_colors, save_path=None, smooth_window=None):
    plt.figure(figsize=(7, 5))
    has_data = False

    for idx, (algorithm, xy) in enumerate(training_reward_data.items()):
        if algorithm not in PPO_ALGORITHMS:
            continue
        if not xy or len(xy) != 2:
            continue
        x_values, y_values = xy
        if len(x_values) == 0 or len(y_values) == 0:
            continue
        if reward_x_mode == 'batch':
            x_values, y_values = _compress_repeated_reward_points(x_values, y_values)
            if len(x_values) == 0:
                continue
        has_data = True
        win = smooth_window if smooth_window is not None else reward_window_size
        if win <= 1:
            y_smooth = pd.Series(y_values)
        else:
            y_smooth = pd.Series(y_values).rolling(window=win, min_periods=1, center=True).mean()
        color = algo_colors.get(algorithm, '#333333')
        plt.plot(
            x_values,
            y_values,
            linestyle='-',
            linewidth=0.55,
            alpha=0.22,
            color=color,
            label=f'{display_name(algorithm)} 原始奖励',
        )
        plt.plot(
            x_values,
            y_smooth,
            linestyle='-',
            linewidth=2.0,
            alpha=0.95,
            color=color,
            label=f'{display_name(algorithm)} 平滑趋势',
        )

    if not has_data:
        plt.close()
        print("未生成 PPO 训练奖励图：没有找到 logs/training_info.csv 或 value/reward 字段。")
        return

    if reward_x_mode == 'batch':
        plt.xlabel('PPO经验批次')
    else:
        plt.xlabel('PPO更新次数')
    plt.ylabel(ylabel)
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=6))
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    show_or_close()


def plot_ratio_acceptance_bar(ratio_experiments, save_path=None):
    if not ratio_experiments:
        print("未生成不同比例接受率图：当前未填写 ratio_experiments_chapter4。均匀比例按整体接受率曲线查看即可。")
        return

    ratio_labels = []
    avg_acceptance = []
    for ratio_label, files in ratio_experiments.items():
        rates = []
        for f in files:
            if os.path.exists(f):
                rate = final_acceptance_rate(f)
                if rate is not None:
                    rates.append(rate)
        if rates:
            ratio_labels.append(ratio_label)
            avg_acceptance.append(sum(rates) / len(rates))

    if not ratio_labels:
        print("未生成不同比例接受率图：没有有效 CSV。")
        return

    plt.figure(figsize=(7, 5))
    bars = plt.bar(ratio_labels, avg_acceptance, color='#4c78a8', width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.3f}', ha='center', va='bottom')
    plt.xlabel('业务比例设置')
    plt.ylabel('平均接受率')
    plt.ylim(0, 1.05)
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    show_or_close()


# ===================== 7. 主程序 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chapter', type=int, choices=[3, 4], default=CHAPTER)
    parser.add_argument('--only-reward', action='store_true', help='只画 PPO-Baseline 和 PPO-MGT 的训练期 SFC episode 奖励曲线（推荐论文使用）')
    parser.add_argument('--sample-step', type=int, default=None, help='折线图取点间隔，默认使用脚本中的 plot_sample_step')
    parser.add_argument('--plot-start', type=int, default=None, help='从第几个 SFC 请求开始画，默认从第 100 条开始，1000 条实验显示 10 个点')
    parser.add_argument('--include-legacy', action='store_true', help='允许自动搜索旧目录 results/virne。默认不启用，避免第三章/第四章混入旧结果。')
    parser.add_argument('--include-ablation', action='store_true', help='自动搜索并绘制 PPO-MGT 消融实验结果。默认不启用，避免混入正式四算法对比图。')
    parser.add_argument('--algorithms', type=str, default=None, help='只画指定算法，逗号分隔，例如 ppo_mlp+,ppo_gat_seq2seq+ 或 PPO-Baseline,PPO-MGT')
    parser.add_argument('--reward-window', type=int, default=None, help='奖励曲线滑动平均窗口。默认 1，即不平滑。')
    parser.add_argument('--reward-x', type=str, choices=['batch', 'update'], default=None, help='奖励图横坐标：batch=每个经验池一个点；update=每次PPO梯度更新一个点。默认 batch。')
    parser.add_argument('--sfc-reward', action='store_true', help='只画训练期每条 SFC 的累计奖励；直接读取 records/*.csv 中的 v_net_reward。')
    parser.add_argument('--batch-reward', action='store_true', help='只画旧的 PPO update batch 平均奖励，即 logs/training_info.csv 中的 value/reward。')
    parser.add_argument('--sfc-reward-window', type=int, default=50, help='SFC累计奖励曲线的滑动平均窗口，默认50，保持旧图毛刺口径。')
    parser.add_argument('--sfc-trend-window', type=int, default=1000, help='SFC累计奖励收敛趋势线的滑动平均窗口，默认1000。')
    parser.add_argument('--sfc-max-episodes', type=int, default=10000, help='SFC累计奖励最多绘制的训练 episode 数，默认10000。')
    parser.add_argument('--only-overall', action='store_true', help='只画整体接受率、平均时延和收益成本比三张基础图。')
    parser.add_argument('--only-service-qos', action='store_true', help='只画第四章四类业务的专属 QoS 指标图。')
    parser.add_argument('--only-paper-metrics', action='store_true', help='第四章只画论文中的 7 张均匀业务比例指标图：3 张整体图 + 4 张业务 QoS 图。')
    parser.add_argument('--any-service-ratio', action='store_true', help='第四章画图时允许读取非均匀业务比例结果。默认只读取 1:1:1:1 均匀业务比例。')
    parser.add_argument('--preview-mip', action='store_true', help='仅用于第四章排版预览：没有同配置 MIP 结果时补充模拟 MIP 曲线。禁止用于论文结论。')
    parser.add_argument('--num-v-nets', type=int, default=None, help='按 SFC 数筛选实验结果。例如第四章论文均匀比例图默认使用 1000。')
    parser.add_argument('--arrival-rate', type=float, default=None, help='按到达率 lambda 筛选实验结果。例如第四章论文均匀比例图默认使用 0.004。')
    parser.add_argument('--snapshot-duration-ms', type=int, default=None, help='按单个拓扑快照时长筛选实验结果。例如第四章论文均匀比例图默认使用 100000。')
    parser.add_argument('--num-snapshots', type=int, default=None, help='按拓扑快照数量筛选实验结果。例如第四章论文均匀比例图默认使用 10。')
    parser.add_argument('--output-dir', type=str, default=None, help='可选，直接保存到指定目录；不填写时仍保存到最新图的新编号目录。')
    parser.add_argument('--no-show', action='store_true', help='批量保存图片时不弹出预览窗口。')
    args = parser.parse_args()

    selected_modes = [args.only_reward or args.sfc_reward, args.batch_reward, args.only_overall, args.only_service_qos, args.only_paper_metrics]
    if sum(bool(x) for x in selected_modes) > 1:
        parser.error('--only-reward/--sfc-reward、--batch-reward、--only-overall、--only-service-qos、--only-paper-metrics 不能同时使用。')
    if args.only_service_qos and args.chapter != 4:
        parser.error('--only-service-qos 只用于第四章。')
    if args.only_paper_metrics and args.chapter != 4:
        parser.error('--only-paper-metrics 只用于第四章。')

    global plot_sample_step, plot_start_index, reward_window_size, reward_x_mode, sfc_reward_trend_window, sfc_reward_max_episodes, FIGURE_DIR, SHOW_FIGURES
    if args.sample_step is not None:
        plot_sample_step = max(1, int(args.sample_step))
    if args.plot_start is not None:
        plot_start_index = max(1, int(args.plot_start))
    if args.reward_window is not None:
        reward_window_size = max(1, int(args.reward_window))
    if args.reward_x is not None:
        reward_x_mode = args.reward_x
    sfc_reward_trend_window = max(1, int(args.sfc_trend_window))
    sfc_reward_max_episodes = max(1, int(args.sfc_max_episodes))
    SHOW_FIGURES = not bool(args.no_show)
    if args.output_dir:
        FIGURE_DIR = os.path.abspath(args.output_dir)
        os.makedirs(FIGURE_DIR, exist_ok=True)
        print(f'本次图片保存目录：{FIGURE_DIR}')
    paper_metric_filter = args.chapter == 4 and args.only_paper_metrics
    required_num_v_nets = 1000 if paper_metric_filter and args.num_v_nets is None else args.num_v_nets
    required_arrival_rate = 0.004 if paper_metric_filter and args.arrival_rate is None else args.arrival_rate
    required_snapshot_duration_ms = 100000 if paper_metric_filter and args.snapshot_duration_ms is None else args.snapshot_duration_ms
    required_num_snapshots = 10 if paper_metric_filter and args.num_snapshots is None else args.num_snapshots

    if args.chapter == 3:
        print("当前绘制：第三章算法对比结果")
        data_files = data_files_chapter3
        summary_files = summary_files_chapter3
    else:
        print("当前绘制：第四章多业务 QoS 结果")
        data_files = data_files_chapter4
        summary_files = summary_files_chapter4

    algorithm_dirs_for_search = None
    if args.algorithms:
        algorithm_dirs_for_search = [
            algorithm_filter_to_dir_name(x)
            for x in args.algorithms.split(',')
            if x.strip()
        ]
    elif args.include_ablation:
        algorithm_dirs_for_search = DEFAULT_ALGORITHM_DIRS + ABLATION_ALGORITHM_DIRS

    if len(data_files) == 0:
        if args.only_reward or args.sfc_reward:
            data_files = auto_find_reward_data_files(
                args.chapter,
                include_legacy=args.include_legacy,
                algorithm_dirs=algorithm_dirs_for_search,
            )
        else:
            data_files = auto_find_data_files(
                args.chapter,
                include_legacy=args.include_legacy,
                algorithm_dirs=algorithm_dirs_for_search,
                require_uniform_service_ratio=(args.chapter == 4 and not args.any_service_ratio),
                require_num_v_nets=required_num_v_nets,
                require_arrival_rate=required_arrival_rate,
                require_snapshot_duration_ms=required_snapshot_duration_ms,
                require_num_snapshots=required_num_snapshots,
            )
        summary_files = [None] * len(data_files)

    if args.algorithms:
        allowed = {normalize_algorithm_filter_name(x) for x in args.algorithms.split(',') if x.strip()}
        filtered_data_files = []
        filtered_summary_files = []
        for idx, file_path in enumerate(data_files):
            algorithm_name = normalize_algorithm_name(file_path)
            if algorithm_name in allowed:
                filtered_data_files.append(file_path)
                filtered_summary_files.append(summary_files[idx] if idx < len(summary_files) else None)
        data_files = filtered_data_files
        summary_files = filtered_summary_files
        print(f"当前只绘制算法：{', '.join(sorted(allowed))}")

    if len(data_files) == 0:
        print("??????? records CSV???????????????? data_files_chapter3 / data_files_chapter4 ??????")
        return

    all_data = load_algorithm_data(data_files, summary_files, chapter=args.chapter)
    if args.chapter == 4 and args.preview_mip:
        all_data = add_ch4_mip_preview_data(all_data, n=required_num_v_nets or 1000)
    algo_colors = build_algo_colors(list(all_data['raw'].keys()))

    if args.batch_reward:
        plot_single_batch_reward_data = {k: v for k, v in all_data['training_reward'].items() if k in PPO_ALGORITHMS}
        plot_training_reward_curve(
            plot_single_batch_reward_data,
            'PPO批次平均奖励',
            algo_colors,
            save_path=get_save_path(f'chapter{args.chapter}_ppo_batch_reward.png'),
            smooth_window=reward_window_size,
        )
        return

    if args.only_reward or args.sfc_reward:
        # 默认直接读取 records/*.csv 中每条 SFC 的 v_net_reward。
        # 奖励图单独处理：每条 SFC 都参与画线，不再按普通指标每隔 20 条抽样。
        plot_record_episode_reward_curve(
            all_data['reward'],
            algo_colors,
            save_path=get_save_path(f'chapter{args.chapter}_sfc_episode_reward.png'),
            smooth_window=max(1, int(args.sfc_reward_window)),
        )
        return

    latency_ylim = 'latency' if args.chapter == 4 else make_wide_ylim(all_data['latency'], min_span=40.0, pad_ratio=0.08, round_to=10.0)
    acceptance_ylim = 'acceptance' if args.chapter == 4 else (0.5, 1.05)
    r2c_ylim = 'r2c' if args.chapter == 4 else None

    if not args.only_service_qos:
        # 第三章和第四章共有图
        plot_metric_curve(all_data['acceptance'], '接受率', algo_colors, ylim=acceptance_ylim, save_path=get_save_path(f'chapter{args.chapter}_acceptance_rate.png'))
        # 第四章论文图按实际显示点自动设置纵轴，避免早期异常累计值拉大空白。
        plot_metric_curve(all_data['latency'], '平均端到端时延（ms）', algo_colors, ylim=latency_ylim, save_path=get_save_path(f'chapter{args.chapter}_latency_curve.png'))
        plot_metric_curve(all_data['r2c_ratio'], '收益成本比', algo_colors, ylim=r2c_ylim, save_path=get_save_path(f'chapter{args.chapter}_r2c_ratio.png'))

    if args.only_overall:
        return

    if not args.only_service_qos and not args.only_paper_metrics:
        plot_record_episode_reward_curve(
            all_data['reward'],
            algo_colors,
            save_path=get_save_path(f'chapter{args.chapter}_sfc_episode_reward.png'),
            smooth_window=max(1, int(args.sfc_reward_window)),
        )
        plot_clock_running_time(all_data['clock_running_time'], algo_colors, save_path=get_save_path(f'chapter{args.chapter}_clock_running_time_bar.png'))

    if args.chapter == 4:
        # 第四章整体 QoS 图

        # 第四章按业务拆分图
        plot_service_metric_all(
            all_data['raw'],
            service_type='delay_sensitive',
            metric='total_latency',
            ylabel='平均端到端时延（ms）',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_delay_sensitive_latency.png'),
            smooth_window=200,
            direct_data=all_data.get('service_direct'),
        )
        plot_service_metric_all(
            all_data['raw'],
            service_type='bandwidth_sensitive',
            metric='effective_bandwidth',
            ylabel='平均有效带宽（Mbps）',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_bandwidth_sensitive_effective_bandwidth.png'),
            smooth_window=200,
            direct_data=all_data.get('service_direct'),
        )
        plot_service_metric_all(
            all_data['raw'],
            service_type='reliability_sensitive',
            metric='path_success_prob',
            ylabel='平均路径成功率',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_reliability_sensitive_path_success.png'),
            smooth_window=200,
            direct_data=all_data.get('service_direct'),
        )
        plot_service_metric_all(
            all_data['raw'],
            service_type='compute_sensitive',
            metric='compute_delay',
            ylabel='平均计算时延（ms）',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_compute_sensitive_compute_delay.png'),
            smooth_window=200,
            direct_data=all_data.get('service_direct'),
        )
        plot_service_guarantee_metric(
            all_data['raw'],
            service_type='delay_sensitive',
            qos_kind='latency',
            ylabel='时延保障率',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_delay_sensitive_latency_guarantee_rate.png'),
            smooth_window=200,
        )
        plot_service_guarantee_metric(
            all_data['raw'],
            service_type='bandwidth_sensitive',
            qos_kind='bandwidth',
            ylabel='带宽保障率',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_bandwidth_sensitive_bandwidth_guarantee_rate.png'),
            smooth_window=200,
        )
        plot_service_guarantee_metric(
            all_data['raw'],
            service_type='reliability_sensitive',
            qos_kind='reliability',
            ylabel='可靠性保障率',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_reliability_sensitive_reliability_guarantee_rate.png'),
            smooth_window=200,
        )
        plot_service_guarantee_metric(
            all_data['raw'],
            service_type='compute_sensitive',
            qos_kind='computing',
            ylabel='计算时延保障率',
            algo_colors=algo_colors,
            save_path=get_save_path('chapter4_compute_sensitive_compute_guarantee_rate.png'),
            smooth_window=200,
        )

        # 可选：不同比例平均接受率
        plot_ratio_acceptance_bar(ratio_experiments_chapter4, save_path=get_save_path('chapter4_ratio_acceptance_bar.png'))


if __name__ == '__main__':
    main()
