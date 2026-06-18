# 低轨卫星网络中服务功能链部署与路由联合优化

本仓库为硕士论文实验代码，主要用于低轨卫星网络（LEO Satellite Network）场景下的服务功能链（Service Function Chain, SFC）部署与路由联合优化仿真。代码在 Virne 仿真框架基础上进行扩展，面向动态卫星拓扑、节点资源约束、链路带宽约束、端到端时延约束以及多业务 QoS 需求，构建 SFC 在线部署实验环境，并实现多种对比算法和绘图脚本。

## 研究内容

本项目主要包含两部分实验：

1. **不区分业务类型的 SFC 部署与路由联合优化**  
   面向单业务 SFC 请求，综合考虑卫星节点 CPU/RAM 资源、星间链路带宽、链路时延和动态拓扑快照，比较不同算法在请求接受率、平均端到端时延、收益成本比和运行时间等指标上的性能。

2. **区分业务类型的多业务 SFC 部署与路由联合优化**  
   面向时延敏感型、带宽密集型、可靠保障型和计算密集型四类业务，引入差异化 QoS 指标，包括端到端时延、有效带宽、路径传输成功率和计算时延，用于分析算法在多业务场景下的适应能力。

## 主要算法

仓库中主要涉及以下算法：

- **MIP**：混合整数规划方法，用作精确优化基线；
- **ACO-META**：基于蚁群思想的元启发式算法；
- **PPO-Baseline / PPO-MLP+**：基于 PPO 的基础强化学习算法；
- **PPO-MGT**：本文提出的融合 Transformer 与多跳图注意力网络的 PPO 方法。

## 目录说明

```text
.
├── main.py                         # 仿真主入口
├── settings/                       # 第三章、第四章实验配置文件
├── virne/                          # 核心仿真环境、求解器和算法实现
├── scripts/                        # 实验运行脚本与绘图脚本
├── plot/                           # 论文图像绘制脚本
├── TLE/                            # 卫星拓扑与快照生成相关代码
└── datasets/                       # 拓扑数据目录
```

说明：实验结果、模型文件、日志文件和论文图片未上传到 GitHub，避免仓库体积过大。相关目录如 `results/`、`outputs/`、`paper_figures/` 等已在 `.gitignore` 中忽略。

## 运行环境

建议使用 Python 3.10，并根据本地环境安装依赖。可参考：

```bash
pip install -r requirements.txt
```

如果使用 Conda，可先创建独立环境：

```bash
conda create -n sfc-leo python=3.10
conda activate sfc-leo
pip install -r requirements.txt
```

## 运行示例

### 第三章实验

运行第三章基础实验或负载实验时，可使用 `settings/main_ch3.yaml` 作为配置入口，例如：

```bash
python main.py --config-name main_ch3 solver.solver_name=ppo_gat_seq2seq+
```

常用绘图命令示例：

```bash
python scripts/plot_ch3_final_figures.py --section basic --basic-preview-mip
python scripts/plot_ch3_final_figures.py --section reward
python scripts/plot_ch3_final_figures.py --section load
```

### 第四章实验

运行第四章多业务实验时，可使用 `settings/main_ch4.yaml` 作为配置入口，例如：

```bash
python main.py --config-name main_ch4 solver.solver_name=ppo_gat_seq2seq+
```

第四章基础指标绘图示例：

```bash
python plot/plot_results.py --chapter 4 --only-paper-metrics --algorithms aco_meta,ppo_mlp+,ppo_gat_seq2seq+ --num-v-nets 1000 --arrival-rate 0.016 --snapshot-duration-ms 60000 --num-snapshots 20 --preview-mip --no-show
```

第四章不同业务比例柱状图绘制示例：

```bash
python scripts/plot_ch4_dominant_ratio_figures.py --service all --figure-set all --algorithms mip,aco_meta,ppo_mlp+,ppo_gat_seq2seq+ --arrival-rate 0.004 --snapshot-duration-ms 100000 --num-snapshots 10 --ratios 0.25,0.40,0.55,0.70
```

## 说明

本仓库主要用于论文实验复现与代码备份。由于训练模型、仿真结果和绘图结果文件较大，仓库仅保留核心代码、配置文件和运行脚本。运行不同实验前，请根据本地路径、预训练模型路径和实验参数修改对应配置或脚本。

## License

本项目基于 Virne 框架进行二次开发，原始框架版权与许可证请参考 Virne 项目说明。本仓库新增代码仅用于学术研究与论文实验复现。
