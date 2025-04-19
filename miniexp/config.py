from typing import Dict, Any
import json
import os

class Config:
    """
    配置系统
    管理实验参数和配置
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        # 默认配置
        self.default_config = {
            # 环境配置
            'env': {
                'width': 20,
                'height': 20,
                'obstacle_density': 0.2,
                'num_predators': 2,
                'num_moving_obstacles': 3
            },
            
            # 智能体配置
            'agent': {
                'initial_perception_level': 1,
                'initial_decision_level': 1,
                'max_perception_level': 5,
                'max_decision_level': 5
            },
            
            # 实验配置
            'experiment': {
                'max_episodes': 1000,
                'max_steps_per_episode': 200,
                'save_interval': 100,
                'visualization': True
            },
            
            # 奖励配置
            'rewards': {
                'target_reached': 100,
                'step_penalty': -1,
                'obstacle_collision': -10,
                'predator_collision': -20,
                'survival_bonus': 1
            },
            
            # 进化配置
            'evolution': {
                'min_episodes_for_evolution': 10,
                'success_rate_threshold': 0.8,
                'reward_threshold': 50,
                'survival_time_threshold': 60
            }
        }
        
        # 加载配置
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = {**self.default_config, **json.load(f)}
        else:
            self.config = self.default_config.copy()
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键
            default: 默认值
            
        Returns:
            Any: 配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
                
        return value
        
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键
            value: 配置值
        """
        keys = key.split('.')
        current = self.config
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
            
        current[keys[-1]] = value
        
    def save(self, config_path: str):
        """
        保存配置
        
        Args:
            config_path: 配置文件路径
        """
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
            
    def get_env_config(self) -> Dict[str, Any]:
        """
        获取环境配置
        
        Returns:
            Dict[str, Any]: 环境配置
        """
        return self.config['env'].copy()
        
    def get_agent_config(self) -> Dict[str, Any]:
        """
        获取智能体配置
        
        Returns:
            Dict[str, Any]: 智能体配置
        """
        return self.config['agent'].copy()
        
    def get_experiment_config(self) -> Dict[str, Any]:
        """
        获取实验配置
        
        Returns:
            Dict[str, Any]: 实验配置
        """
        return self.config['experiment'].copy()
        
    def get_reward_config(self) -> Dict[str, Any]:
        """
        获取奖励配置
        
        Returns:
            Dict[str, Any]: 奖励配置
        """
        return self.config['rewards'].copy()
        
    def get_evolution_config(self) -> Dict[str, Any]:
        """
        获取进化配置
        
        Returns:
            Dict[str, Any]: 进化配置
        """
        return self.config['evolution'].copy() 