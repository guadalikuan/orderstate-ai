from typing import Dict, Any, List
from dataclasses import dataclass
from enum import Enum
import json

class EnvironmentType(Enum):
    """环境类型枚举"""
    SIMPLE = 'simple'
    MEDIUM = 'medium'
    ADVANCED = 'advanced'

class AgentType(Enum):
    """智能体类型枚举"""
    BASELINE = 'baseline'
    ENERGY = 'energy'

class EnvironmentConfig:
    """环境配置类"""
    def __init__(self, width: int = 10, height: int = 10, 
                 type: EnvironmentType = EnvironmentType.ADVANCED,
                 obstacle_density: float = 0.2,
                 num_predators: int = 2,
                 num_moving_obstacles: int = 3):
        self.width = width
        self.height = height
        self.type = type
        self.obstacle_density = obstacle_density
        self.num_predators = num_predators
        self.num_moving_obstacles = num_moving_obstacles

    def to_dict(self) -> Dict[str, Any]:
        return {
            'width': self.width,
            'height': self.height,
            'type': self.type.value,
            'obstacle_density': self.obstacle_density,
            'num_predators': self.num_predators,
            'num_moving_obstacles': self.num_moving_obstacles
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'EnvironmentConfig':
        return cls(
            width=data.get('width', 10),
            height=data.get('height', 10),
            type=EnvironmentType(data.get('type', 'advanced')),
            obstacle_density=data.get('obstacle_density', 0.2),
            num_predators=data.get('num_predators', 2),
            num_moving_obstacles=data.get('num_moving_obstacles', 3)
        )

class AgentConfig:
    """智能体配置类"""
    def __init__(self, initial_energy: float = 100.0, 
                 energy_threshold: float = 20.0,
                 initial_perception_level: int = 1,
                 initial_decision_level: int = 1,
                 max_perception_level: int = 5,
                 max_decision_level: int = 5):
        self.initial_energy = initial_energy
        self.energy_threshold = energy_threshold
        self.initial_perception_level = initial_perception_level
        self.initial_decision_level = initial_decision_level
        self.max_perception_level = max_perception_level
        self.max_decision_level = max_decision_level

    def to_dict(self) -> Dict[str, Any]:
        return {
            'initial_energy': self.initial_energy,
            'energy_threshold': self.energy_threshold,
            'initial_perception_level': self.initial_perception_level,
            'initial_decision_level': self.initial_decision_level,
            'max_perception_level': self.max_perception_level,
            'max_decision_level': self.max_decision_level
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfig':
        return cls(
            initial_energy=data.get('initial_energy', 100.0),
            energy_threshold=data.get('energy_threshold', 20.0),
            initial_perception_level=data.get('initial_perception_level', 1),
            initial_decision_level=data.get('initial_decision_level', 1),
            max_perception_level=data.get('max_perception_level', 5),
            max_decision_level=data.get('max_decision_level', 5)
        )

class ExperimentConfig:
    """实验配置类"""
    def __init__(self):
        self.environment = EnvironmentConfig()
        self.agent = AgentConfig()
        self.experiment = {
            'max_episodes': 100,
            'max_steps_per_episode': 200,
            'save_interval': 100,
            'visualization': True
        }
        self.rewards = {
            'target_reached': 100,
            'step_penalty': -1,
            'obstacle_collision': -10,
            'predator_collision': -20,
            'survival_bonus': 1
        }
        self.evolution = {
            'min_episodes_for_evolution': 10,
            'success_rate_threshold': 0.8,
            'reward_threshold': 50,
            'survival_time_threshold': 60
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            'environment': self.environment.to_dict(),
            'agent': self.agent.to_dict(),
            'experiment': self.experiment,
            'rewards': self.rewards,
            'evolution': self.evolution
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExperimentConfig':
        config = cls()
        if 'environment' in data:
            config.environment = EnvironmentConfig.from_dict(data['environment'])
        if 'agent' in data:
            config.agent = AgentConfig.from_dict(data['agent'])
        if 'experiment' in data:
            config.experiment.update(data['experiment'])
        if 'rewards' in data:
            config.rewards.update(data['rewards'])
        if 'evolution' in data:
            config.evolution.update(data['evolution'])
        return config

    def save(self, filepath: str):
        """保存配置到文件"""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, filepath: str) -> 'ExperimentConfig':
        """从文件加载配置"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        return {
            'environment': self.environment.to_dict(),
            'agent': self.agent.to_dict(),
            'experiment': self.experiment,
            'rewards': self.rewards,
            'evolution': self.evolution
        } 