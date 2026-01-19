from dataclasses import dataclass, field
from nerfstudio.configs.base_config import InstantiateConfig
from typing import Any, Dict, List, Tuple, Type, Literal, Optional, Union
from abc import abstractmethod
import torch


# Default loss weight scheduler 
@dataclass
class LossWeightSchedulerConfig(InstantiateConfig):
    """configuration for loss weight scheduling"""

    _target: Type = field(default_factory=lambda: LossWeightScheduler)
    """target class to instanciate"""

class LossWeightScheduler:

    @abstractmethod
    def weight(self, step) -> float: 
        """
        get the weight at step
        """
        raise NotImplementedError


# Constant loss weight
@dataclass
class ConstantLossWeightConfig(LossWeightSchedulerConfig):
    """configuration for constant loss weight"""

    _target: Type = field(default_factory=lambda: ConstantLossWeight)
    """target class to instantiate"""
    weight: float = 1.0

class ConstantLossWeight(LossWeightScheduler):
    def __init__(self, config: ConstantLossWeightConfig):
        self.config = config

    def weight(self, step):
        return self.config.weight    



# Linear loss weight scheduler
@dataclass
class LinearLossWeightSchedulerConfig(LossWeightSchedulerConfig):
    _target: Type = field(default_factory=lambda: LinearLossWeightScheduler)
    start_weight: float = 0.1
    end_weight: float = 1.0
    start_step: int = 0
    end_step: int = 10000

class LinearLossWeightScheduler:
    def __init__(self, config: LinearLossWeightSchedulerConfig):
        self.start_weight = config.start_weight
        self.end_weight = config.end_weight
        self.start_step = config.start_step
        self.end_step = config.end_step

    def weight(self, step):
        t = min(1.0, max(0.0, (step-self.start_step)/(self.end_step-self.start_step)))
        return self.start_weight + (self.end_weight-self.start_weight) * t



# Linear loss weight scheduler
@dataclass
class SigmoidLossWeightSchedulerConfig(LossWeightSchedulerConfig):
    _target: Type = field(default_factory=lambda: SigmoidLossWeightScheduler)
    #start_weight: float = 0.1
    #end_weight: float = 1.0
    #start_step: int = 0
    #end_step: int = 10000
    x_mult: int = 10000
    x_shift: int = 10000


class SigmoidLossWeightScheduler:
    def __init__(self, config: SigmoidLossWeightSchedulerConfig):
        self.config = config

    def weight(self, step):
        step = float(step)
        return torch.sigmoid(torch.tensor((step-self.config.x_shift)/self.config.x_mult))
