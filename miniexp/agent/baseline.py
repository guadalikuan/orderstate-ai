from typing import Dict, Any, Union, List, Tuple
import numpy as np
import random
from miniexp.agent.base import BaseAgent

class BaselineAgent(BaseAgent):
    """
    基准智能体
    实现一个基本的随机动作策略
    """
    
    def __init__(self, env, name: str = "BaselineAgent"):
        """
        初始化基准智能体
        
        Args:
            env: 环境实例
            name: 智能体名称
        """
        super().__init__(name=name)
        self.env = env
        
    def act(self, state: Dict[str, Any]) -> int:
        """
        根据当前状态选择动作
        简单实现：随机选择一个有效动作
        
        Args:
            state: 当前环境状态
            
        Returns:
            int: 选择的动作编号 (0: 上, 1: 下, 2: 左, 3: 右)
        """
        # 不需要考虑状态，随机选择动作
        return random.randint(0, 3)
        
    def reset(self) -> None:
        """
        重置智能体状态
        """
        pass
        
    def get_remaining_energy(self) -> float:
        """
        获取剩余能量
        基准智能体不考虑能量
        
        Returns:
            float: 剩余能量 (NaN)
        """
        return np.nan
        
    def is_energy_exhausted(self) -> bool:
        """
        检查能量是否耗尽
        基准智能体不考虑能量
        
        Returns:
            bool: 始终返回False
        """
        return False 