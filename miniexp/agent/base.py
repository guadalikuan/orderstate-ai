from typing import Dict, Any, Tuple, Optional
import numpy as np

class BaseAgent:
    """基础智能体类，定义了所有智能体的基本属性和方法"""
    
    def __init__(self, name: str = "BaseAgent"):
        """
        初始化基础智能体
        
        Args:
            name: 智能体名称
        """
        self.name = name
        self.position = None
        self.energy = 100.0
        self.max_energy = 100.0
        self.alive = True
        self.perception = None
        self.decision = None
        
    def reset(self) -> None:
        """重置智能体状态"""
        self.position = None
        self.energy = self.max_energy
        self.alive = True
        
        # 重置能力模块
        if self.perception:
            self.perception.reset()
            
        if self.decision:
            self.decision.reset()
        
    def act(self, state: Dict[str, Any]) -> int:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前环境状态
            
        Returns:
            int: 选择的动作
        """
        raise NotImplementedError("子类必须实现act方法")
        
    def get_state(self) -> Dict[str, Any]:
        """获取智能体当前状态"""
        return {
            'name': self.name,
            'position': self.position,
            'energy': self.energy,
            'alive': self.alive,
            'perception_level': self.perception.level if self.perception else 0,
            'decision_level': self.decision.level if self.decision else 0
        }
        
    def update(self, action: int, reward: float, next_state: Dict[str, Any]) -> None:
        """
        更新智能体状态
        
        Args:
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
        """
        self.energy = min(self.max_energy, self.energy + reward)
        if self.energy <= 0:
            self.alive = False
        
    def get_remaining_energy(self) -> float:
        """
        获取剩余能量
        
        Returns:
            float: 剩余能量
        """
        return self.energy
        
    def is_energy_exhausted(self) -> bool:
        """
        检查能量是否耗尽
        
        Returns:
            bool: 是否耗尽
        """
        return self.energy <= 0
        
    def get_metrics(self) -> Dict[str, Any]:
        """获取智能体性能指标"""
        metrics = {
            'name': self.name,
            'energy': self.energy,
            'alive': self.alive,
        }
        
        # 添加感知能力指标
        if self.perception:
            metrics['perception'] = self.perception.get_metrics()
            
        # 添加决策能力指标
        if self.decision:
            metrics['decision'] = self.decision.get_metrics()
            
        return metrics 