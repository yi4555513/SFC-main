# 低轨卫星网络中服务功能链部署与路由联合优化

本仓库为硕士论文实验代码，主要用于低轨卫星网络（LEO Satellite Network）场景下的服务功能链（Service Function Chain, SFC）部署与路由联合优化仿真。代码围绕论文实验需求进行整理，支持动态卫星拓扑快照、节点资源约束、链路带宽约束、端到端时延约束以及多业务 QoS 需求。

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
├── main/                # 核心环境、控制器、求解器和强化学习算法
├── scripts/             # 拓扑快照生成脚本
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

## 拓扑快照生成

`scripts/generate_topology_snapshots.py` 可用于生成卫星网络拓扑快照。具体参数可根据实验星座规模、快照数量和快照间隔进行调整。

## 注意事项

- 本仓库仅保留运行实验所需核心代码和配置，不包含实验结果和训练模型。
- 若使用预训练模型测试 PPO 类算法，需要在命令或配置中指定 `solver.pretrained_model_path`。
- 若重新训练 PPO 类算法，可调整 `training.num_train_epochs` 等训练参数。
- 运行 MIP 时可根据机器性能设置求解时间限制，例如 `solver.mip_time_limit_seconds`。

## License

本仓库代码仅用于学术研究与论文实验复现。
