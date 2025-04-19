from typing import Dict, Any, Optional
import numpy as np
from miniexp.agent.base import BaseAgent

class EnergyAgent(BaseAgent):
    """
    能量智能体
    考虑能量消耗和恢复的智能体
    """
    
    def __init__(self, env, init_energy: float = 100, threshold: float = 20, name: str = "EnergyAgent"):
        """
        初始化能量智能体
        
        Args:
            env: 环境实例
            init_energy: 初始能量
            threshold: 能量阈值，低于此值时策略会发生变化
            name: 智能体名称
        """
        super().__init__(name=name)
        self.env = env
        self.max_energy = init_energy
        self.energy = init_energy
        self.energy_threshold = threshold
        
    def reset(self) -> None:
        """
        重置智能体状态
        """
        super().reset()
        self.energy = self.max_energy
        
    def act(self, state: Dict[str, Any]) -> int:
        """
        根据当前状态和能量水平选择动作
        
        Args:
            state: 当前环境状态
            
        Returns:
            int: 选择的动作
        """
        # 获取当前位置和目标位置
        if isinstance(state, tuple):
            # 如果state是元组，直接使用agent_pos和target_pos
            curr_pos = self.env.agent_pos if hasattr(self.env, 'agent_pos') else (0, 0)
            target_pos = self.env.target_pos if hasattr(self.env, 'target_pos') else None
        else:
            # 如果state是字典，从中获取agent_pos和target_pos
            curr_pos = state.get('agent_pos', (0, 0))
            target_pos = state.get('target_pos')
        
        # 如果能量低于阈值，采取保守策略
        if self.energy < self.energy_threshold:
            # 简单实现：随机移动
            return np.random.randint(0, 4)
            
        # 否则向目标移动
        if target_pos is not None:
            # 计算到目标的方向
            dx = target_pos[0] - curr_pos[0]
            dy = target_pos[1] - curr_pos[1]
            
            # 选择最接近目标方向的动作
            if abs(dx) > abs(dy):
                return 2 if dx < 0 else 3  # 左或右
            else:
                return 0 if dy < 0 else 1  # 上或下
        
        # 如果没有目标，随机移动
        return np.random.randint(0, 4)
        
    def update(self, action: int, reward: float, next_state: Dict[str, Any]) -> None:
        """
        更新智能体状态
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        super().update(action, reward, next_state)
        
        # 每步消耗能量
        self.energy -= 1
        
        # 处理奖励带来的能量变化
        if reward > 0:
            self.energy = min(self.max_energy, self.energy + reward)
            
        # 检查是否因能量耗尽而死亡
        if self.energy <= 0:
            self.alive = False 