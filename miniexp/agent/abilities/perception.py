from typing import Dict, Any, Tuple
import numpy as np

class PerceptionAbility:
    """
    感知能力模块
    管理智能体的感知范围和感知精度
    """
    
    def __init__(self, level: int = 1):
        """
        初始化感知能力
        
        Args:
            level: 感知能力等级 (1-5)
        """
        self.level = level
        self.range = self._get_range()
        self.precision = self._get_precision()
        
    def _get_range(self) -> int:
        """
        根据等级获取感知范围
        """
        ranges = {
            1: 1,  # 只能感知相邻格子
            2: 2,  # 感知2格范围
            3: 3,  # 感知3格范围
            4: 4,  # 感知4格范围
            5: 5   # 感知5格范围
        }
        return ranges.get(self.level, 1)
        
    def _get_precision(self) -> float:
        """
        根据等级获取感知精度
        """
        precisions = {
            1: 0.6,  # 60%准确率
            2: 0.7,  # 70%准确率
            3: 0.8,  # 80%准确率
            4: 0.9,  # 90%准确率
            5: 1.0   # 100%准确率
        }
        return precisions.get(self.level, 0.6)
        
    def upgrade(self) -> bool:
        """
        升级感知能力
        
        Returns:
            bool: 是否升级成功
        """
        if self.level < 5:
            self.level += 1
            self.range = self._get_range()
            self.precision = self._get_precision()
            return True
        return False
        
    def get_observation(self, pos: Tuple[int, int], env_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取观察结果
        
        Args:
            pos: 当前位置
            env_state: 环境状态
            
        Returns:
            Dict[str, Any]: 观察结果
        """
        # 获取原始观察
        raw_obs = env_state.get('grid', np.zeros((1, 1)))
        
        # 应用感知范围
        y, x = pos
        obs_range = self.range
        obs = raw_obs[
            max(0, y-obs_range):min(raw_obs.shape[0], y+obs_range+1),
            max(0, x-obs_range):min(raw_obs.shape[1], x+obs_range+1)
        ]
        
        # 应用感知精度
        if self.precision < 1.0:
            mask = np.random.random(obs.shape) < self.precision
            obs = np.where(mask, obs, 0)
            
        return {
            'grid': obs,
            'range': self.range,
            'precision': self.precision
        }
        
    def get_metrics(self) -> Dict[str, float]:
        """
        获取感知能力指标
        
        Returns:
            Dict[str, float]: 感知能力指标
        """
        return {
            'level': self.level,
            'range': self.range,
            'precision': self.precision
        } 