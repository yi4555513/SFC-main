from typing import Callable, Dict, Any, Optional, Type
from abc import ABC, abstractmethod
import numpy as np
from main.core import Solution
from main.network import VirtualNetwork, PhysicalNetwork
from main.utils.service_qos import (
    QOS_DIMS,
    build_phi,
    compute_service_qos_metrics,
    get_service_qos_config,
    get_thresholds,
    is_service_qos_enabled,
)


class BaseRewardCalculator(ABC):
    """
    Abstract base class for reward calculation. Users can extend this for custom reward logic.
    To use a custom reward, subclass this and register your class with RewardCalculatorRegistry.
    """

    def __init__(self, config):
        self.config = config
        self.v_net_reward = 0.0
        intermediate_reward = self.config.rl.reward_calculator.intermediate_reward
        assert intermediate_reward == -1 or intermediate_reward >= 0, 'intermediate_reward should be a non-negative number or -1'

    def _reward_param(self, name: str, default):
        reward_cfg = self.config.rl.reward_calculator
        if hasattr(reward_cfg, 'get'):
            return reward_cfg.get(name, default)
        return getattr(reward_cfg, name, default)

    def _scale_reward(self, reward: float) -> float:
        return float(reward) * float(self._reward_param('reward_scale', 1.0))

    def _clip_reward(self, reward: float) -> float:
        min_v = self._reward_param('clip_min', None)
        max_v = self._reward_param('clip_max', None)
        if min_v is not None:
            reward = max(float(min_v), reward)
        if max_v is not None:
            reward = min(float(max_v), reward)
        return float(reward)

    @abstractmethod
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        pass


class RewardCalculatorRegistry:
    """
    Registry for reward calculator classes. Supports registration and retrieval by name.
    """
    _registry: Dict[str, Type[BaseRewardCalculator]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(calculator_cls: Type[BaseRewardCalculator]):
            if name in cls._registry:
                raise ValueError(f"Reward calculator '{name}' is already registered.")
            cls._registry[name] = calculator_cls
            return calculator_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> Type[BaseRewardCalculator]:
        if name not in cls._registry:
            raise NotImplementedError(f"Reward calculator '{name}' is not implemented.")
        return cls._registry[name]

    @classmethod
    def list_registered(cls) -> Dict[str, Type[BaseRewardCalculator]]:
        return dict(cls._registry)


@RewardCalculatorRegistry.register('gradual_intermediate')
class GradualIntermediateRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        elif solution.get('place_result', False) and solution.get('route_result', False):
            curr_place_progress = get_curr_place_progress(v_net, solution)
            v_net_r2c_ratio = float(solution.get('v_net_r2c_ratio', 0.0))
            reward = 0.1 * curr_place_progress * v_net_r2c_ratio
        else:
            reward = -float(get_curr_place_progress(v_net, solution))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('adaptive_intermediate')
class AdaptiveWeightRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        weight = 1 / v_net.num_nodes
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = weight
        else:
            reward = -weight
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('fixed_intermediate')
class FixedWeightRewardCalculator(BaseRewardCalculator):
    def __init__(self, config):
        super().__init__(config)

    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        intermediate = float(self.config.rl.reward_calculator.intermediate_reward)
        r2c_weight = float(self._reward_param('r2c_weight', 1.0))
        latency_weight = float(self._reward_param('latency_weight', 0.5))
        success_bonus = float(self._reward_param('success_bonus', 0.2))
        failure_penalty = float(self._reward_param('failure_penalty', 0.5))
        max_latency = float(v_net.graph.get('max_latency', self._reward_param('latency_norm', 300.0)) or 300.0)
        latency = float(solution.get('total_latency', 0.0) or 0.0)
        norm_latency = min(latency / max(max_latency, 1e-9), 2.0)

        if solution.get('result', False):
            r2c = float(solution.get('v_net_r2c_ratio', 0.0))
            reward = success_bonus + r2c_weight * r2c - latency_weight * norm_latency
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = intermediate
        else:
            reward = -failure_penalty
        reward = self._clip_reward(self._scale_reward(reward))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('fixed_incremental')
class FixedIncrementalRewardCalculator(BaseRewardCalculator):
    """Chapter 3 reward with dense incremental feedback.

    It keeps the original final reward (R/C and end-to-end delay), and adds a
    light penalty on each successful intermediate step according to the
    newly introduced path delay and resource cost.  This helps reduce the
    credit-assignment problem without changing the deployment constraints.
    """

    def __init__(self, config):
        super().__init__(config)

    def _incremental_step_reward(self, v_net: VirtualNetwork, solution: Solution) -> float:
        intermediate = float(self.config.rl.reward_calculator.intermediate_reward)
        delay_weight = float(self._reward_param('increment_delay_weight', 0.3))
        cost_weight = float(self._reward_param('increment_cost_weight', 0.2))
        latency_norm = float(v_net.graph.get('max_latency', self._reward_param('latency_norm', 300.0)) or 300.0)
        cost_norm = float(self._reward_param('cost_norm', 100.0))

        delta_latency = float(solution.get('step_delta_latency', 0.0) or 0.0)
        delta_cost = float(solution.get('step_delta_cost', 0.0) or 0.0)
        delay_loss = delta_latency / max(latency_norm, 1e-9)
        cost_loss = delta_cost / max(cost_norm, 1e-9)
        return intermediate - delay_weight * delay_loss - cost_weight * cost_loss

    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        r2c_weight = float(self._reward_param('r2c_weight', 1.0))
        latency_weight = float(self._reward_param('latency_weight', 0.5))
        success_bonus = float(self._reward_param('success_bonus', 0.2))
        failure_penalty = float(self._reward_param('failure_penalty', 0.5))
        max_latency = float(v_net.graph.get('max_latency', self._reward_param('latency_norm', 300.0)) or 300.0)
        latency = float(solution.get('total_latency', 0.0) or 0.0)
        norm_latency = min(latency / max(max_latency, 1e-9), 2.0)

        if solution.get('result', False):
            r2c = float(solution.get('v_net_r2c_ratio', 0.0))
            final_reward = success_bonus + r2c_weight * r2c - latency_weight * norm_latency
            reward = self._incremental_step_reward(v_net, solution) + final_reward
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = self._incremental_step_reward(v_net, solution)
        else:
            reward = -failure_penalty
        reward = self._clip_reward(self._scale_reward(reward))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('vanilla')
class VanillaRewardCalculator(BaseRewardCalculator):
    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        if solution.get('result', False):
            reward = float(solution.get('v_net_r2c_ratio', 0.0))
        else:
            reward = 0.0
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward



@RewardCalculatorRegistry.register('service_qos')
class ServiceQosRewardCalculator(BaseRewardCalculator):
    """Service-aware reward for chapter 4.

    The four QoS dimensions are latency, reliability, bandwidth and computing.
    `phi` is not one-hot: the dominant service dimension receives the largest
    weight and the other dimensions keep non-zero weights.
    """

    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        intermediate = float(self.config.rl.reward_calculator.intermediate_reward)
        r2c_weight = float(self._reward_param('r2c_weight', 0.6))
        qos_weight = float(self._reward_param('qos_weight', 1.0))
        success_bonus = float(self._reward_param('success_bonus', 0.2))
        failure_penalty = float(self._reward_param('failure_penalty', 0.6))
        qos_violation_penalty = float(self._reward_param('qos_violation_penalty', 0.4))

        if solution.get('result', False):
            metrics = compute_service_qos_metrics(self.config, p_net, v_net, solution)
            solution.update(metrics)
            qos_score = float(metrics.get('qos_score', 0.0))
            r2c = float(solution.get('v_net_r2c_ratio', 0.0))
            # ??????????????????? + ????? + ??? QoS?
            # ???????? qos_score ????? 1 - latency / max_latency?
            # ???????qos_score ??????????? latency ???
            reward = success_bonus + r2c_weight * r2c + qos_weight * qos_score
            if not metrics.get('qos_satisfied', False):
                reward -= qos_violation_penalty
        elif solution.get('place_result', False) and solution.get('route_result', False):
            # VNF/???? step ????????????
            reward = intermediate
        else:
            # ???????????? reward ???????????
            reward = -failure_penalty
        reward = self._clip_reward(self._scale_reward(reward))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward


@RewardCalculatorRegistry.register('service_qos_incremental')
class ServiceQosIncrementalRewardCalculator(BaseRewardCalculator):
    """Chapter 4 service-aware reward with incremental QoS feedback."""

    def _get_phi_and_thresholds(self, v_net: VirtualNetwork):
        qos_cfg = get_service_qos_config(self.config)
        service_type = v_net.graph.get('service_type', qos_cfg.get('default_service_type', 'delay_sensitive'))
        if all(v_net.graph.get(f'phi_{dim}') is not None for dim in QOS_DIMS):
            phi = {dim: float(v_net.graph.get(f'phi_{dim}')) for dim in QOS_DIMS}
        else:
            phi = build_phi(service_type, qos_cfg)
        return phi, get_thresholds(v_net, qos_cfg)

    def _incremental_step_reward(self, v_net: VirtualNetwork, solution: Solution) -> float:
        intermediate = float(self.config.rl.reward_calculator.intermediate_reward)
        phi, thresholds = self._get_phi_and_thresholds(v_net)

        cost_weight = float(self._reward_param('increment_cost_weight', 0.2))
        delay_weight = float(self._reward_param('increment_delay_weight', 0.3))
        bandwidth_weight = float(self._reward_param('increment_bandwidth_weight', 0.5))
        reliability_weight = float(self._reward_param('increment_reliability_weight', 0.5))
        compute_weight = float(self._reward_param('increment_compute_weight', 0.3))
        cost_norm = float(self._reward_param('cost_norm', 100.0))

        delta_cost = float(solution.get('step_delta_cost', 0.0) or 0.0)
        delta_latency = float(solution.get('step_delta_latency', 0.0) or 0.0)
        delta_compute = float(solution.get('step_delta_compute_delay', 0.0) or 0.0)
        has_new_path = bool(solution.get('step_has_new_path', False))

        delay_loss = delta_latency / max(float(thresholds.get('max_latency', 300.0)), 1e-9)
        cost_loss = delta_cost / max(cost_norm, 1e-9)
        compute_loss = delta_compute / max(float(thresholds.get('max_compute_delay', 80.0)), 1e-9)

        if has_new_path:
            effective_bandwidth = float(solution.get('step_effective_bandwidth', 0.0) or 0.0)
            min_bandwidth = float(thresholds.get('min_bandwidth', 15.0))
            bandwidth_loss = max(0.0, min_bandwidth - effective_bandwidth) / max(min_bandwidth, 1e-9)

            path_success = float(solution.get('step_path_success_prob', 1.0) or 1.0)
            min_success = float(thresholds.get('min_path_success', 0.95))
            reliability_loss = max(0.0, min_success - path_success) / max(min_success, 1e-9)
        else:
            bandwidth_loss = 0.0
            reliability_loss = 0.0

        qos_increment_penalty = (
            phi['latency'] * delay_weight * delay_loss
            + phi['bandwidth'] * bandwidth_weight * bandwidth_loss
            + phi['reliability'] * reliability_weight * reliability_loss
            + phi['computing'] * compute_weight * compute_loss
        )
        return intermediate - cost_weight * cost_loss - qos_increment_penalty

    def compute(self, p_net: PhysicalNetwork, v_net: VirtualNetwork, solution: Solution) -> float:
        r2c_weight = float(self._reward_param('r2c_weight', 0.6))
        qos_weight = float(self._reward_param('qos_weight', 1.0))
        success_bonus = float(self._reward_param('success_bonus', 0.2))
        failure_penalty = float(self._reward_param('failure_penalty', 0.6))
        qos_violation_penalty = float(self._reward_param('qos_violation_penalty', 0.4))

        if solution.get('result', False):
            metrics = compute_service_qos_metrics(self.config, p_net, v_net, solution)
            solution.update(metrics)
            qos_score = float(metrics.get('qos_score', 0.0))
            r2c = float(solution.get('v_net_r2c_ratio', 0.0))
            final_reward = success_bonus + r2c_weight * r2c + qos_weight * qos_score
            if not metrics.get('qos_satisfied', False):
                final_reward -= qos_violation_penalty
            reward = self._incremental_step_reward(v_net, solution) + final_reward
        elif solution.get('place_result', False) and solution.get('route_result', False):
            reward = self._incremental_step_reward(v_net, solution)
        else:
            reward = -failure_penalty
        reward = self._clip_reward(self._scale_reward(reward))
        solution['v_net_reward'] += reward
        self.v_net_reward += reward
        return reward

def get_curr_place_progress(v_net: VirtualNetwork, solution: Solution) -> float:
    """Calculate the current placement progress."""
    node_slots = solution.get('node_slots', None)
    if node_slots is None or not hasattr(node_slots, 'keys'):
        return 0.0
    num_placed_v_net_nodes = len(node_slots.keys())
    total_place_count = v_net.num_nodes - 1 if v_net.num_nodes > 1 else 1
    return num_placed_v_net_nodes / total_place_count
