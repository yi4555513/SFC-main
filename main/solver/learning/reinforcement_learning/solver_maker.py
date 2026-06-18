# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================

import numpy as np
from gym import spaces
from typing import Any, Dict, Tuple, List, Union, Optional, Type, Callable


import numpy as np


from main.network import PhysicalNetwork, VirtualNetwork
from main.core import Controller, Recorder, Counter, Solution, Logger

from main.solver import SolverRegistry
from main.solver.learning.rl_policy import CnnActorCritic
from main.solver.learning.rl_core import JointPRStepInstanceRLEnv, PlaceStepInstanceRLEnv
from main.solver.learning.rl_core.rl_solver import PGSolver, A2CSolver, PPOSolver, A3CSolver
from main.solver.learning.rl_core.instance_agent import InstanceAgent
from main.solver.learning.rl_core.tensor_convertor import TensorConvertor
from main.solver.learning.rl_core.policy_builder import PolicyBuilder
from main.solver.learning.rl_core.feature_constructor import FeatureConstructorRegistry, PNetVNodeFeatureConstructor


def make_solver_class(solver_name: str, instance_env_cls, base_solver_cls, policy_builder, obs_as_tensor):
    """
    Factory for RL solver classes.
    """
    class SolverClass(InstanceAgent, base_solver_cls):
        def __init__(self, controller, recorder, counter, logger, config, **kwargs):
            InstanceAgent.__init__(self, instance_env_cls)
            base_solver_cls.__init__(self, controller, recorder, counter, logger, config, policy_builder, obs_as_tensor, **kwargs)
    SolverClass.__name__ = solver_name
    SolverClass.__qualname__ = solver_name
    SolverRegistry.register(solver_name=solver_name, solver_type='r_learning')(SolverClass)
    return SolverClass