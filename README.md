# 低轨卫星网络中服务功能链部署与路由联合优化

本仓库为硕士论文实验代码，主要用于低轨卫星网络（LEO Satellite Network）场景下的服务功能链（Service Function Chain, SFC）部署与路由联合优化仿真。代码在 Virne 仿真框架基础上进行扩展，支持动态卫星拓扑快照、节点资源约束、链路带宽约束、端到端时延约束以及多业务 QoS 需求。

## 主要内容

本项目包含两类实验场景：

1. **单业务 SFC 部署与路由联合优化**  
   面向不区分业务类型的 SFC 请求，比较 MIP、ACO-META、PPO-Baseline 和 PPO-MGT 等算法在请求接受率、平均端到端时延、收益成本比和运行时间等指标上的表现。

2. **多业务 SFC 部署与路由联合优化**  
   面向时延敏感型、带宽密集型、可靠保障型和计算密集型四类业务，引入差异化业务需求，用于验证算法在多业务场景下的部署适应能力。

## 主要算法

- **MIP**：混合整数规划方法；
- **ACO-META**：蚁群元启发式算法；
- **PPO-Baseline / PPO-MLP+**：基于 PPO 的基础强化学习算法；
- **PPO-MGT**：融合 Transformer 与多跳图注意力网络的 PPO 方法。

## 目录结构

```text
.
├── main.py              # 仿真主入口
├── settings/            # 实验配置文件
├── virne/               # 核心环境、控制器、求解器和强化学习算法
├── scripts/             # 实验运行脚本
├── TLE/                 # 卫星拓扑与快照生成相关代码
└── datasets/            # 拓扑数据目录
```

说明：实验结果、模型文件、日志文件和图片文件没有上传到仓库，相关目录如 `results/`、`outputs/`、`paper_figures/`、`overnight_logs/` 等已在 `.gitignore` 中忽略。

## 环境配置

建议使用 Python 3.10。可使用 Conda 创建环境：

```bash
conda create -n sfc-leo python=3.10
conda activate sfc-leo
pip install -r requirements.txt
```

## 基本运行方式

项目使用 Hydra 配置系统。第三章实验使用 `main_ch3`，第四章实验使用 `main_ch4`。

### 运行第三章单业务实验

PPO-MGT 示例：

```bash
python main.py --config-name main_ch3 solver.solver_name=ppo_gat_seq2seq+
```

PPO-Baseline 示例：

```bash
python main.py --config-name main_ch3 solver.solver_name=ppo_mlp+
```

ACO-META 示例：

```bash
python main.py --config-name main_ch3 solver.solver_name=aco_meta
```

MIP 示例：

```bash
python main.py --config-name main_ch3 solver.solver_name=mip
```

### 运行第四章多业务实验

PPO-MGT 示例：

```bash
python main.py --config-name main_ch4 solver.solver_name=ppo_gat_seq2seq+
```

PPO-Baseline 示例：

```bash
python main.py --config-name main_ch4 solver.solver_name=ppo_mlp+
```

ACO-META 示例：

```bash
python main.py --config-name main_ch4 solver.solver_name=aco_meta
```

MIP 示例：

```bash
python main.py --config-name main_ch4 solver.solver_name=mip
```

## 常用实验脚本

仓库中的 `scripts/` 目录提供了部分批量实验脚本，例如：

```bash
powershell -ExecutionPolicy Bypass -File scripts/run_ch3_load_8_to_72_3alg.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ch4_basic_lambda16_3alg.ps1
powershell -ExecutionPolicy Bypass -File scripts/run_ch4_dominant_ratio_lambda16_200_4alg.ps1
```

运行前请根据本地路径和模型位置检查脚本中的参数，尤其是 Python 环境路径、预训练模型路径、SFC 数量、到达率、快照间隔和快照数量等设置。

## 注意事项

- 本仓库仅保留运行实验所需代码和配置，不包含实验结果和训练模型。
- 若使用预训练模型测试 PPO 类算法，需要在命令或脚本中指定 `solver.pretrained_model_path`。
- 若重新训练 PPO 类算法，可调整 `training.num_train_epochs` 等训练参数。
- 运行 MIP 时可根据机器性能设置求解时间限制，例如 `solver.mip_time_limit_seconds`。

## License

本项目基于 Virne 框架进行二次开发，原始框架版权与许可证请参考 Virne 项目说明。本仓库新增代码仅用于学术研究与论文实验复现。
