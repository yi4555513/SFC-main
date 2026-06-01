import numpy as np
from gym import spaces
from omegaconf import DictConfig
from typing import Any, Dict, Optional, Type

from virne.network import VirtualNetwork, PhysicalNetwork
from virne.network import AttributeBenchmarkManager, AttributeBenchmarks, TopologicalMetrics, \
    TopologicalMetricCalculator
from virne.solver.learning.obs_handler import ObservationHandler
from virne.core import Solution, Controller


class BaseFeatureConstructor:
    """
    Abstract base class for feature construction. Extend this for custom observation logic.
    All required state should be passed as arguments to methods, not stored as instance attributes.
    """

    def __init__(
            self,
            p_net: PhysicalNetwork,
            v_net: VirtualNetwork,
            config: Optional[DictConfig] = None
    ):
        self.obs_handler = ObservationHandler()
        self.config = config
        self.p_net = p_net
        self.v_net = v_net
        p_net_attribute_benchmarks = AttributeBenchmarkManager.get_from_cache('p_net')
        p_net_topological_metrics = TopologicalMetricCalculator.get_from_cache('p_net')
        v_net_topological_metrics = TopologicalMetricCalculator.get_from_cache('v_net')
        if p_net_attribute_benchmarks is None:
            p_net_attribute_benchmarks = AttributeBenchmarkManager.get_benchmarks(p_net, node_attrs=True,
                                                                                  link_attrs=True, link_sum_attrs=True)
        if p_net_topological_metrics is None:
            p_net_topological_metrics = TopologicalMetricCalculator.calculate(p_net, degree=True, closeness=True,
                                                                              eigenvector=True, betweenness=True)
        if v_net_topological_metrics is None:
            v_net_topological_metrics = TopologicalMetricCalculator.calculate(v_net, degree=True, closeness=True,
                                                                              eigenvector=True, betweenness=True)
        self.p_net_attribute_benchmarks = p_net_attribute_benchmarks
        self.node_attr_benchmarks = p_net_attribute_benchmarks.node_attr_benchmarks
        self.link_attr_benchmarks = p_net_attribute_benchmarks.link_attr_benchmarks
        self.link_sum_attr_benchmarks = p_net_attribute_benchmarks.link_sum_attr_benchmarks

        self.p_net_topological_metrics = p_net_topological_metrics
        self.v_net_topological_metrics = v_net_topological_metrics
        self.extracted_attr_types = config.rl.feature_constructor.extracted_attr_types
        # Chapter 4 adds the link information attribute `loss`.  Older cached
        # benchmarks were built only for resource/extrema/latency attributes,
        # so make sure all attributes requested by extracted_attr_types have a
        # normalization benchmark before constructing observations.
        self.node_attr_benchmarks = self._merge_missing_benchmarks(
            self.node_attr_benchmarks,
            AttributeBenchmarkManager.get_node_attr_benchmarks(p_net, self.extracted_attr_types)
        )
        self.link_attr_benchmarks = self._merge_missing_benchmarks(
            self.link_attr_benchmarks,
            AttributeBenchmarkManager.get_link_attr_benchmarks(p_net, self.extracted_attr_types)
        )
        self.link_sum_attr_benchmarks = self._merge_missing_benchmarks(
            self.link_sum_attr_benchmarks,
            AttributeBenchmarkManager.get_link_sum_attr_benchmarks(p_net, self.extracted_attr_types)
        )
        self.if_use_degree_metric = config.rl.feature_constructor.if_use_degree_metric
        self.if_use_more_topological_metrics = config.rl.feature_constructor.if_use_more_topological_metrics
        self.if_use_aggregated_link_attrs = config.rl.feature_constructor.if_use_aggregated_link_attrs
        self.if_use_node_status_flags = config.rl.feature_constructor.if_use_node_status_flags

    @staticmethod
    def _merge_missing_benchmarks(base: Optional[Dict[str, float]], extra: Optional[Dict[str, float]]) -> Dict[str, float]:
        merged = dict(base or {})
        for key, value in (extra or {}).items():
            value = float(value or 0.0)
            if value <= 0.0:
                value = 1.0
            merged.setdefault(key, value)
        return merged

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[
        str, Any]:
        raise NotImplementedError

    def guess_observation_space(self, p_net: PhysicalNetwork, v_net: VirtualNetwork,
                                solution: Optional[Solution] = None, v_node_id: int = 0) -> Dict[str, Any]:
        """
        This method is used to guess the observation space based on the physical and virtual networks.
        It should return a dictionary with keys as observation names and values as their respective spaces.
        """
        num_p_net_node_attrs = len(p_net.get_node_attrs(self.extracted_attr_types))
        num_p_net_link_attrs = len(p_net.get_link_attrs(self.extracted_attr_types))
        num_obs_attrs = num_p_net_node_attrs + num_p_net_link_attrs + 2
        spaces.Dict({
            'p_net_x': spaces.Box(low=0, high=1, shape=(p_net.num_nodes, num_p_net_node_attrs), dtype=np.float32),
            'p_net_edge_index': spaces.Box(low=0, high=p_net.num_nodes, shape=(2, p_net.num_links), dtype=np.int32),
            'p_net_edge_attr': spaces.Box(low=0, high=p_net.num_nodes, shape=(p_net.num_links, 2), dtype=np.int32),
            'v_node': spaces.Box(low=0, high=100,
                                 shape=(int(num_p_net_node_attrs / 2) + int(num_p_net_link_attrs / 2) + 2,),
                                 dtype=np.float32)
        })
        raise NotImplementedError

    def _get_service_phi(self, v_net: VirtualNetwork) -> np.ndarray:
        dims = ['latency', 'reliability', 'bandwidth', 'computing']
        values = []
        for dim in dims:
            value = v_net.graph.get(f'phi_{dim}', 0.0)
            values.append(float(value or 0.0))
        if sum(values) <= 0:
            return np.zeros((0,), dtype=np.float32)
        return np.array(values, dtype=np.float32)

    def _get_compute_sensitive_flag(self, v_net: VirtualNetwork, v_node_id: int) -> np.ndarray:
        if v_node_id not in v_net.nodes:
            return np.zeros((0,), dtype=np.float32)
        if 'compute_sensitive' not in v_net.nodes[v_node_id]:
            return np.zeros((0,), dtype=np.float32)
        return np.array([float(v_net.nodes[v_node_id].get('compute_sensitive', 0.0) or 0.0)], dtype=np.float32)

    def _construct_p_net_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution,
                                  curr_v_node_id: int) -> Dict[str, Any]:
        """
        node_resource, average_distance, p_net_degreees, p_net_nodes_states, v_node_features
        """
        # ====== Node Data of Physical Network ======
        # Node Data 1: Node Attributes
        p_node_attrs_data = self.obs_handler.get_node_attrs_obs(p_net, node_attr_types=self.extracted_attr_types,
                                                                node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            p_nodes_status = self.obs_handler.get_p_net_nodes_status(p_net, v_net, solution['node_slots'],
                                                                     curr_v_node_id)
        else:
            p_nodes_status = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            p_node_link_min_data = self.obs_handler.get_link_aggr_attrs_obs(p_net,
                                                                            link_attr_types=self.extracted_attr_types,
                                                                            aggr='min',
                                                                            link_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_mean_data = self.obs_handler.get_link_aggr_attrs_obs(p_net,
                                                                             link_attr_types=self.extracted_attr_types,
                                                                             aggr='mean',
                                                                             link_sum_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_max_data = self.obs_handler.get_link_aggr_attrs_obs(p_net,
                                                                            link_attr_types=self.extracted_attr_types,
                                                                            aggr='max',
                                                                            link_attr_benchmarks=self.link_attr_benchmarks)
            p_node_link_sum_data = self.obs_handler.get_link_aggr_attrs_obs(p_net,
                                                                            link_attr_types=self.extracted_attr_types,
                                                                            aggr='sum',
                                                                            link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
            node_link_aggr_attrs_data = np.concatenate(
                (p_node_link_min_data, p_node_link_mean_data, p_node_link_max_data, p_node_link_sum_data), axis=-1)
        else:
            node_link_aggr_attrs_data = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        # Node Data 4: Topological Metrics
        avg_distance = self.obs_handler.get_average_distance(p_net, solution['node_slots'], normalization=True)
        if self.config.rl.feature_constructor.if_use_degree_metric:
            p_net_degree_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=True, closeness=False, eigenvector=False,
                betweenness=False)
        else:
            p_net_degree_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        if self.config.rl.feature_constructor.if_use_more_topological_metrics:
            p_net_more_topological_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=False, closeness=True, eigenvector=True, betweenness=True)
        else:
            p_net_more_topological_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)
        p_net_topological_metrics = np.concatenate((avg_distance, p_net_degree_metrics, p_net_more_topological_metrics),
                                                   axis=-1)
        # Node Data Merging
        node_data = np.concatenate(
            (p_node_attrs_data, p_nodes_status, node_link_aggr_attrs_data, p_net_topological_metrics), axis=-1)
        # ====== Node Data of Virtual Node ======
        # num_node_attrs = len(p_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(p_net.get_link_attrs(self.extracted_attr_types))
        # num_p_net_x = num_node_attrs + 1  # avg distance
        # num_p_net_x += 2 if self.if_use_node_status_flags else 0
        # num_p_net_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_p_net_x += 1 if self.if_use_degree_metric else 0
        # num_p_net_x += 3 if self.if_use_more_topological_metrics else 0

        # Edge Index
        edge_index = self.obs_handler.get_link_index_obs(p_net)
        # Edge Attributes
        link_data = self.obs_handler.get_link_attrs_obs(p_net, link_attr_types=self.extracted_attr_types,
                                                        link_attr_benchmarks=self.link_attr_benchmarks)
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data
        }
        return p_net_obs

    def _get_cached_two_hop_pairs(self, p_net: PhysicalNetwork, edge_index: np.ndarray):
        """Return 2-hop edge index and middle-node lists for the current topology.

        The previous implementation rebuilt the dense adjacency matrix and
        searched all middle nodes every time an observation was constructed.
        During PPO-MGT training this happens once per VNF placement step, so it
        becomes a major CPU bottleneck.  The topology of one snapshot is fixed
        while residual resources change, therefore the 2-hop topology and the
        middle-node candidates can be cached; only the 2-hop edge attributes
        need to be refreshed from the current residual link attributes.
        """
        edge_index = np.asarray(edge_index, dtype=np.int64)
        cache_key = (int(p_net.num_nodes), edge_index.shape, edge_index.tobytes())
        cache = getattr(self, '_two_hop_topology_cache', {})
        if cache_key in cache:
            return cache[cache_key]

        neighbors = {i: set() for i in range(int(p_net.num_nodes))}
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0][i]), int(edge_index[1][i])
            neighbors.setdefault(u, set()).add(v)
            neighbors.setdefault(v, set()).add(u)

        edge_pairs = []
        middle_nodes = []
        for u in range(int(p_net.num_nodes)):
            one_hop = neighbors.get(u, set())
            two_hop = set()
            for k in one_hop:
                two_hop.update(neighbors.get(k, set()))
            two_hop.discard(u)
            two_hop.difference_update(one_hop)
            for w in sorted(two_hop):
                mids = sorted(one_hop.intersection(neighbors.get(w, set())))
                if mids:
                    edge_pairs.append((u, w))
                    middle_nodes.append(mids)

        if edge_pairs:
            edge_index_2hop = np.array(edge_pairs, dtype=np.int64).T
        else:
            edge_index_2hop = np.zeros((2, 0), dtype=np.int64)

        cache[cache_key] = (edge_index_2hop, middle_nodes)
        self._two_hop_topology_cache = cache
        return edge_index_2hop, middle_nodes

    def _construct_p_net_2hop_edges(self, p_net: PhysicalNetwork) -> Dict[str, Any]:
        """Construct only 2-hop edge features.

        PPO-MGT's GAT uses the same physical node features for 1-hop and 2-hop
        message passing.  Reconstructing all node features again for the 2-hop
        branch is unnecessary and makes the ablation experiments much slower.
        """
        edge_index = self.obs_handler.get_link_index_obs(p_net)
        edge_attr = self.obs_handler.get_link_attrs_obs(
            p_net,
            link_attr_types=self.extracted_attr_types,
            link_attr_benchmarks=self.link_attr_benchmarks
        )
        edge_index_2hop, middle_nodes = self._get_cached_two_hop_pairs(p_net, edge_index)

        link_attrs = p_net.get_link_attrs(self.extracted_attr_types)
        link_attr_names = [attr.name for attr in link_attrs]
        num_edge_features = int(edge_attr.shape[1]) if getattr(edge_attr, 'ndim', 0) == 2 else len(link_attr_names)
        attr_idx_map = {name: i for i, name in enumerate(link_attr_names)}
        bw_idx = attr_idx_map.get("bw", None)
        delay_idx = attr_idx_map.get("ltc", None)
        loss_idx = attr_idx_map.get("loss", None)

        edge_dict = {}
        for i in range(edge_index.shape[1]):
            u, v = int(edge_index[0][i]), int(edge_index[1][i])
            edge_dict[(u, v)] = edge_attr[i]
            edge_dict[(v, u)] = edge_attr[i]

        edge_attr_2hop = []
        for pair_idx in range(edge_index_2hop.shape[1]):
            u, w = int(edge_index_2hop[0][pair_idx]), int(edge_index_2hop[1][pair_idx])
            feature = [0.0] * num_edge_features
            min_bw = float("inf")
            delays, losses = [], []
            for k in middle_nodes[pair_idx]:
                edge_uk = edge_dict.get((u, k))
                edge_kw = edge_dict.get((k, w))
                if edge_uk is None or edge_kw is None:
                    continue
                if bw_idx is not None:
                    min_bw = min(min_bw, float(edge_uk[bw_idx]), float(edge_kw[bw_idx]))
                if delay_idx is not None:
                    delays.append(float(edge_uk[delay_idx]) + float(edge_kw[delay_idx]))
                if loss_idx is not None:
                    loss_1 = float(edge_uk[loss_idx])
                    loss_2 = float(edge_kw[loss_idx])
                    losses.append(1.0 - (1.0 - loss_1) * (1.0 - loss_2))
            if bw_idx is not None:
                feature[bw_idx] = 0.0 if min_bw == float("inf") else min_bw
            if delay_idx is not None:
                feature[delay_idx] = float(np.mean(delays)) if delays else 0.0
            if loss_idx is not None:
                feature[loss_idx] = float(np.mean(losses)) if losses else 0.0
            edge_attr_2hop.append(feature)

        if edge_attr_2hop:
            edge_attr_2hop = np.asarray(edge_attr_2hop, dtype=np.float32)
        else:
            edge_attr_2hop = np.zeros((0, num_edge_features), dtype=np.float32)

        return {
            'edge_index': edge_index_2hop,
            'edge_attr': edge_attr_2hop
        }

    # TODO
    def _construct_p_net_features_2hop(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution,
                                       curr_v_node_id: int) -> Dict[str, Any]:
        """
        node_resource, average_distance, p_net_degreees, p_net_nodes_states, v_node_features
        """
        import torch
        import numpy as np
        from torch_geometric.utils import to_dense_adj, dense_to_sparse
        # ====== Node Data of Physical Network ======
        # Node Data 1: Node Attributes
        p_node_attrs_data = self.obs_handler.get_node_attrs_obs(
            p_net,
            node_attr_types=self.extracted_attr_types,
            node_attr_benchmarks=self.node_attr_benchmarks
        )

        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            p_nodes_status = self.obs_handler.get_p_net_nodes_status(
                p_net, v_net, solution['node_slots'], curr_v_node_id
            )
        else:
            p_nodes_status = np.zeros((p_net.num_nodes, 0), dtype=np.float32)

        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            p_node_link_min_data = self.obs_handler.get_link_aggr_attrs_obs(
                p_net, link_attr_types=self.extracted_attr_types, aggr='min',
                link_attr_benchmarks=self.link_attr_benchmarks
            )
            p_node_link_mean_data = self.obs_handler.get_link_aggr_attrs_obs(
                p_net, link_attr_types=self.extracted_attr_types, aggr='mean',
                link_sum_attr_benchmarks=self.link_attr_benchmarks
            )
            p_node_link_max_data = self.obs_handler.get_link_aggr_attrs_obs(
                p_net, link_attr_types=self.extracted_attr_types, aggr='max',
                link_attr_benchmarks=self.link_attr_benchmarks
            )
            p_node_link_sum_data = self.obs_handler.get_link_aggr_attrs_obs(
                p_net, link_attr_types=self.extracted_attr_types, aggr='sum',
                link_sum_attr_benchmarks=self.link_sum_attr_benchmarks
            )
            node_link_aggr_attrs_data = np.concatenate(
                (p_node_link_min_data, p_node_link_mean_data, p_node_link_max_data, p_node_link_sum_data), axis=-1
            )
        else:
            node_link_aggr_attrs_data = np.zeros((p_net.num_nodes, 0), dtype=np.float32)

        # Node Data 4: Topological Metrics
        avg_distance = self.obs_handler.get_average_distance(p_net, solution['node_slots'], normalization=True)

        if self.config.rl.feature_constructor.if_use_degree_metric:
            p_net_degree_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=True, closeness=False, eigenvector=False,
                betweenness=False
            )
        else:
            p_net_degree_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)

        if self.config.rl.feature_constructor.if_use_more_topological_metrics:
            p_net_more_topological_metrics = self.obs_handler.get_node_topological_metrics(
                p_net, self.p_net_topological_metrics, degree=False, closeness=True, eigenvector=True, betweenness=True
            )
        else:
            p_net_more_topological_metrics = np.zeros((p_net.num_nodes, 0), dtype=np.float32)

        p_net_topological_metrics = np.concatenate((avg_distance, p_net_degree_metrics, p_net_more_topological_metrics),
                                                   axis=-1)

        # 合并所有节点特征
        node_data = np.concatenate(
            (p_node_attrs_data, p_nodes_status, node_link_aggr_attrs_data, p_net_topological_metrics), axis=-1)

        # ====== 构造 2-hop 边索引和边属性 ======
        # 获取原始 1-hop 边索引和边属性
        edge_index = self.obs_handler.get_link_index_obs(p_net)  # [2, num_edges]
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long)
        edge_attr = self.obs_handler.get_link_attrs_obs(p_net, link_attr_types=self.extracted_attr_types,
                                                        link_attr_benchmarks=self.link_attr_benchmarks)
        edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32)  # shape [num_edges, num_attrs]

        # 构建稠密邻接矩阵
        adj = to_dense_adj(edge_index_tensor, max_num_nodes=p_net.num_nodes)[0]  # [N, N]

        # 找到所有 2-hop 边
        adj_2hop = torch.matmul(adj, adj)
        adj_2hop[adj > 0] = 0  # 删除原始 1-hop
        adj_2hop.fill_diagonal_(0)  # 删除自环

        edge_index_2hop, _ = dense_to_sparse((adj_2hop > 0).float())  # [2, num_edges_2hop]

        # 构建 2-hop 边属性
        # 注意：self.extracted_attr_types 里面放的是属性“类型”（如 resource/latency/information），
        # 不是具体属性名。以前这里直接用 len(self.extracted_attr_types) 构造 2-hop 边特征，
        # 在第三章 ["resource", "latency"] 下会得到 2 维，但在部分情况下会和真实
        # link_attr 维度不一致；第四章加入 loss 后真实边特征又变成 3 维。
        # 因此这里必须以 p_net.get_link_attrs(...) 返回的真实链路属性列表为准，
        # 保证 1-hop edge_attr 和 2-hop edge_attr 维度完全一致。
        link_attrs = p_net.get_link_attrs(self.extracted_attr_types)
        link_attr_names = [attr.name for attr in link_attrs]
        num_edge_features = int(edge_attr_tensor.shape[1]) if edge_attr_tensor.ndim == 2 else len(link_attr_names)
        attr_idx_map = {name: i for i, name in enumerate(link_attr_names)}
        bw_idx = attr_idx_map.get("bw", None)
        delay_idx = attr_idx_map.get("ltc", None)
        loss_idx = attr_idx_map.get("loss", None)

        num_nodes = p_net.num_nodes
        # 用于快速查找原始边属性
        edge_dict = {}
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0][i], edge_index[1][i]
            edge_dict[(u, v)] = edge_attr_tensor[i]
            edge_dict[(v, u)] = edge_attr_tensor[i]  # 无向图

        edge_attr_2hop = []
        for i in range(edge_index_2hop.shape[1]):
            u, w = edge_index_2hop[0][i].item(), edge_index_2hop[1][i].item()
            # 枚举所有中间节点 k，使得 u-k-w 是 2-hop 路径
            min_bw = float("inf")
            delays = []  # 收集所有路径的延迟
            losses = []  # 收集两跳路径的等效丢包率
            found = False
            for k in range(num_nodes):
                if (u, k) in edge_dict and (k, w) in edge_dict:
                    found = True
                    bw_1 = edge_dict[(u, k)][bw_idx].item() if bw_idx is not None else 0.0
                    bw_2 = edge_dict[(k, w)][bw_idx].item() if bw_idx is not None else 0.0
                    delay_1 = edge_dict[(u, k)][delay_idx].item() if delay_idx is not None else 0.0
                    delay_2 = edge_dict[(k, w)][delay_idx].item() if delay_idx is not None else 0.0
                    min_bw = min(min_bw, bw_1, bw_2)
                    delays.append(delay_1 + delay_2)
                    if loss_idx is not None:
                        loss_1 = edge_dict[(u, k)][loss_idx].item()
                        loss_2 = edge_dict[(k, w)][loss_idx].item()
                        # 两跳都成功的概率为 (1-loss_1)(1-loss_2)，
                        # 因此等效丢包率为 1 - (1-loss_1)(1-loss_2)。
                        losses.append(1.0 - (1.0 - loss_1) * (1.0 - loss_2))

            if found:
                # 聚合所有两跳路径信息
                total_delay = np.mean(delays) if delays else 0.0
                total_loss = np.mean(losses) if losses else 0.0
                feature = [0.0] * num_edge_features
                if bw_idx is not None:
                    feature[bw_idx] = 0.0 if min_bw == float("inf") else min_bw
                if delay_idx is not None:
                    feature[delay_idx] = total_delay
                if loss_idx is not None:
                    feature[loss_idx] = total_loss
                edge_attr_2hop.append(feature)
            else:
                # 如果没有找到 2-hop 路径，填充零值
                feature = [0.0] * num_edge_features
                edge_attr_2hop.append(feature)

        # 如果 edge_attr_2hop 为空（没有 2-hop 边），创建一个零填充的边属性
        if not edge_attr_2hop:
            edge_attr_2hop = [[0.0] * num_edge_features] * edge_index_2hop.shape[1] if \
                edge_index_2hop.shape[1] > 0 else [[0.0] * num_edge_features]

        edge_attr_2hop_tensor = torch.tensor(edge_attr_2hop, dtype=torch.float32)

        # 最终输出
        p_net_obs = {
            'x': node_data,
            'edge_index': edge_index_2hop.numpy(),
            'edge_attr': edge_attr_2hop_tensor.numpy()
        }
        return p_net_obs

    def _construct_v_node_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution,
                                   curr_v_node_id: int) -> Dict[str, Any]:
        if curr_v_node_id >= v_net.num_nodes:
            return {'x': np.array([], dtype=np.float32)}
        # ====== Node Data of Virtual Node ======
        # Node Data 1: Node Attributes
        v_node_demand = self.obs_handler.get_v_node_demand(v_net, curr_v_node_id,
                                                           node_attr_types=self.extracted_attr_types,
                                                           node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            v_net_status = self.obs_handler.get_v_node_status(v_net, curr_v_node_id, p_net.num_nodes)
        else:
            v_net_status = np.zeros((0,), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            v_node_mean_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='mean',
                                                                                    link_attr_types=self.extracted_attr_types,
                                                                                    link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_max_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='max',
                                                                                   link_attr_types=self.extracted_attr_types,
                                                                                   link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_min_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='min',
                                                                                   link_attr_types=self.extracted_attr_types,
                                                                                   link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_sum_link_demend = self.obs_handler.get_v_node_aggr_link_demands(v_net, curr_v_node_id, aggr='sum',
                                                                                   link_attr_types=self.extracted_attr_types,
                                                                                   link_attr_benchmarks=self.link_sum_attr_benchmarks)
            v_node_aggr_attrs_demand = np.concatenate(
                (v_node_mean_link_demend, v_node_max_link_demend, v_node_min_link_demend, v_node_sum_link_demend),
                axis=-1)
        else:
            v_node_aggr_attrs_demand = np.zeros((0,), dtype=np.float32)
        # Node Data 4: Topological Metrics
        num_neighbors = len(v_net.adj[curr_v_node_id]) / v_net.num_nodes
        v_num_neighbors = np.array([num_neighbors], dtype=np.float32)
        # Merging Node Data
        service_phi = self._get_service_phi(v_net)
        compute_flag = self._get_compute_sensitive_flag(v_net, curr_v_node_id)
        v_node_x = np.concatenate([v_node_demand, v_net_status, v_node_aggr_attrs_demand, v_num_neighbors, service_phi, compute_flag], axis=0)
        # ====== Node Data of Virtual Node ======
        # num_node_attrs = len(v_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(v_net.get_link_attrs(self.extracted_attr_types))
        # num_v_node_x = num_node_attrs + 1  # avg distance
        # num_v_node_x += 2 if self.if_use_node_status_flags else 0
        # num_v_node_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_v_node_x += 0 if self.if_use_degree_metric else 0
        # num_v_node_x += 0 if self.if_use_more_topological_metrics else 0
        return {'x': v_node_x}

    def _construct_v_net_features(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution,
                                  curr_v_node_id: int) -> Dict[str, Any]:
        # ====== Node Data of Virtual Network ======
        # Node Data 1: Node Attributes
        v_node_attrs_data = self.obs_handler.get_node_attrs_obs(v_net, node_attr_types=self.extracted_attr_types,
                                                                node_attr_benchmarks=self.node_attr_benchmarks)
        # Node Data 2: Node Status
        if self.config.rl.feature_constructor.if_use_node_status_flags:
            v_node_status = self.obs_handler.get_v_net_nodes_status(v_net, solution['node_slots'], curr_v_node_id,
                                                                    consist_decision=True, neighbor_flags=True)
        else:
            v_node_status = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        # Node Data 3: Link Aggregated Attributes
        if self.config.rl.feature_constructor.if_use_aggregated_link_attrs:
            v_node_link_min_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net,
                                                                                link_attr_types=self.extracted_attr_types,
                                                                                aggr='min',
                                                                                link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_link_max_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net,
                                                                                link_attr_types=self.extracted_attr_types,
                                                                                aggr='max',
                                                                                link_attr_benchmarks=self.link_attr_benchmarks)
            v_node_link_sum_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net,
                                                                                link_attr_types=self.extracted_attr_types,
                                                                                aggr='sum',
                                                                                link_sum_attr_benchmarks=self.link_sum_attr_benchmarks)
            v_node_link_mean_resource = self.obs_handler.get_link_aggr_attrs_obs(v_net,
                                                                                 link_attr_types=self.extracted_attr_types,
                                                                                 aggr='mean',
                                                                                 link_sum_attr_benchmarks=self.link_attr_benchmarks)
            v_node_aggr_attrs_data = np.concatenate((v_node_link_min_resource, v_node_link_max_resource,
                                                     v_node_link_sum_resource, v_node_link_mean_resource), axis=-1)
        else:
            v_node_aggr_attrs_data = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        # Node data 4: topological metrics
        num_neighbors = len(v_net.adj[curr_v_node_id]) / v_net.num_nodes
        v_num_neighbors = np.array([num_neighbors], dtype=np.float32)
        v_num_neighbors = np.expand_dims(v_num_neighbors, axis=1)
        v_num_neighbors = np.ones((v_net.num_nodes, 1), dtype=np.float32) * v_num_neighbors
        if self.config.rl.feature_constructor.if_use_degree_metric:
            v_node_degree_metrics = self.obs_handler.get_node_topological_metrics(
                v_net, self.v_net_topological_metrics, degree=True, closeness=False, eigenvector=False,
                betweenness=False)
        else:
            v_node_degree_metrics = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        if self.config.rl.feature_constructor.if_use_more_topological_metrics:
            v_node_more_topological_metrics = self.obs_handler.get_node_topological_metrics(
                v_net, self.v_net_topological_metrics, degree=False, closeness=True, eigenvector=True, betweenness=True)
        else:
            v_node_more_topological_metrics = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        v_node_topological_metrics = np.concatenate(
            (v_num_neighbors, v_node_degree_metrics, v_node_more_topological_metrics), axis=-1)
        # Node data 4: topological metrics
        # v_node_neighbor_flags = np.ones((v_net.num_nodes, 1), dtype=np.float32) * self.obs_handler.get_v_node_neighbor_flags(v_net, solution['node_slots'], curr_v_node_id).sum() / 10
        # Merging Node Data
        service_phi = self._get_service_phi(v_net)
        if service_phi.size > 0:
            service_phi_data = np.ones((v_net.num_nodes, service_phi.shape[0]), dtype=np.float32) * service_phi
        else:
            service_phi_data = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        if v_net.num_nodes > 0 and 'compute_sensitive' in v_net.nodes[next(iter(v_net.nodes))]:
            compute_flag_data = np.array([[float(v_net.nodes[n].get('compute_sensitive', 0.0) or 0.0)] for n in v_net.nodes], dtype=np.float32)
        else:
            compute_flag_data = np.zeros((v_net.num_nodes, 0), dtype=np.float32)
        node_data = np.concatenate(
            (v_node_attrs_data, v_node_status, v_node_aggr_attrs_data, v_node_topological_metrics, service_phi_data, compute_flag_data), axis=-1)
        # ====== Node Data of Virtual Network ======
        # num_node_attrs = len(v_net.get_node_attrs(self.extracted_attr_types))
        # num_link_attrs = len(v_net.get_link_attrs(self.extracted_attr_types))
        # num_v_net_x = num_node_attrs + 1  # avg distance
        # num_v_net_x += 2 if self.if_use_node_status_flags else 0
        # num_v_net_x += num_link_attrs * 4 if self.if_use_aggregated_link_attrs else 0
        # num_v_net_x += 1 if self.if_use_degree_metric else 0
        # num_v_net_x += 3 if self.if_use_more_topological_metrics else 0
        # Edge Index
        edge_index = self.obs_handler.get_link_index_obs(v_net)
        link_data = self.obs_handler.get_link_attrs_obs(v_net, link_attr_types=self.extracted_attr_types,
                                                        link_attr_benchmarks=self.link_attr_benchmarks)
        v_net_obs = {
            'x': node_data,
            'edge_index': edge_index,
            'edge_attr': link_data,
        }
        return v_net_obs


class FeatureConstructorRegistry:
    """
    Registry for feature constructor classes. Supports registration and retrieval by name.
    """
    _registry: Dict[str, Type[BaseFeatureConstructor]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(handler_cls: Type[BaseFeatureConstructor]):
            if name in cls._registry:
                raise ValueError(f"Reward calculator '{name}' is already registered.")
            cls._registry[name] = handler_cls
            return handler_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseFeatureConstructor]:
        if name not in cls._registry:
            raise NotImplementedError(f"Feature constructor '{name}' is not implemented.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[BaseFeatureConstructor]]:
        return dict(cls._registry)


@FeatureConstructorRegistry.register('p_net')
class PNetFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[
        str, Any]:
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
        }
        return combined_obs


@FeatureConstructorRegistry.register('p_net_v_node')
class PNetVNodeFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[
        str, Any]:
        v_node_obs = self._construct_v_node_features(p_net, v_net, solution, curr_v_node_id)
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_node_x': v_node_obs['x']
        }
        return combined_obs


@FeatureConstructorRegistry.register('p_net_v_net')
class PNetVNetFeatureConstructor(BaseFeatureConstructor):

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[
        str, Any]:
        v_net_obs = self._construct_v_net_features(p_net, v_net, solution, curr_v_node_id)
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        # Only the 2-hop edge index/attributes are needed by the multi-hop GAT.
        # Rebuilding all physical-node features for the 2-hop branch duplicates
        # most of the observation construction cost, so use the lightweight edge
        # builder here.
        p_net_obs_2hop = self._construct_p_net_2hop_edges(p_net)
        # Concatenate the observations
        combined_obs = {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            # TODO
            'p_net_edge_index_2hop': p_net_obs_2hop['edge_index'],
            'p_net_edge_attr_2hop': p_net_obs_2hop['edge_attr'],

            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
        }
        return combined_obs


@FeatureConstructorRegistry.register('p_net_v_net_no_2hop')
class PNetVNetNoTwoHopFeatureConstructor(BaseFeatureConstructor):
    """PNet+VNet observation without constructing 2-hop edges.

    This is used by the ablations that remove M-GAT.  Their physical-network
    encoder is a node-wise MLP and never reads edge_index_2hop/edge_attr_2hop,
    so computing the 2-hop branch only wastes CPU time.
    """

    def construct(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution, curr_v_node_id: int) -> Dict[
        str, Any]:
        v_net_obs = self._construct_v_net_features(p_net, v_net, solution, curr_v_node_id)
        p_net_obs = self._construct_p_net_features(p_net, v_net, solution, curr_v_node_id)
        return {
            'p_net_x': p_net_obs['x'],
            'p_net_edge_index': p_net_obs['edge_index'],
            'p_net_edge_attr': p_net_obs['edge_attr'],
            'v_net_x': v_net_obs['x'],
            'v_net_edge_index': v_net_obs['edge_index'],
            'v_net_edge_attr': v_net_obs['edge_attr'],
        }


def get_selected_p_net_nodes(solution):
    """
    Get the selected physical network nodes based on the solution.
    """
    return list(solution['node_slots'].values())


def get_placed_v_net_nodes(solution):
    """
    Get the placed virtual network nodes based on the solution.
    """
    return list(solution['node_slots'].keys())
