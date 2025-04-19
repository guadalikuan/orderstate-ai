import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class ActionModule:
    """
    动作执行模块
    
    负责将决策转化为实际动作，执行并返回结果
    是序态循环中"决策→动作"阶段的实现
    """
    
    def __init__(self, action_space_size: int = 4, energy_costs: Optional[List[float]] = None, 
                 failure_prob: float = 0.05, random_seed: Optional[int] = None):
        """
        初始化动作模块
        
        Args:
            action_space_size: 动作空间大小
            energy_costs: 各动作消耗的能量列表，默认为None，会初始化为[0.8, 1.0, 0.8, 1.0]
            failure_prob: 动作执行失败的概率
            random_seed: 随机种子
        """
        self.action_space_size = action_space_size
        
        # 默认能量消耗 [上, 下, 左, 右]
        self.energy_costs = energy_costs if energy_costs else [0.8, 1.0, 0.8, 1.0]
        
        # 确保能量消耗列表长度与动作空间一致
        if len(self.energy_costs) < action_space_size:
            self.energy_costs.extend([1.0] * (action_space_size - len(self.energy_costs)))
        
        self.failure_prob = failure_prob
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 动作映射
        self.action_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        
        # 动作历史
        self.action_history = []
        
        # 当前动作状态
        self.current_action = None
        
        # 统计信息
        self.stats = {
            "actions_executed": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_energy_consumed": 0.0
        }
    
    def reset(self):
        """
        重置动作模块
        """
        self.action_history = []
        self.current_action = None
        self.stats = {
            "actions_executed": 0,
            "successful_actions": 0,
            "failed_actions": 0,
            "total_energy_consumed": 0.0
        }
    
    def get_energy_cost(self, action: int, anxiety: float = 0.0) -> float:
        """
        获取指定动作的能量消耗
        
        Args:
            action: 动作索引
            anxiety: 焦虑水平，会影响能量消耗
            
        Returns:
            float: 能量消耗值
        """
        if not (0 <= action < len(self.energy_costs)):
            return 1.0  # 默认消耗
            
        base_cost = self.energy_costs[action]
        
        # 焦虑会增加能量消耗
        anxiety_factor = 1.0 + (anxiety * 0.5)  # 焦虑最高增加50%能耗
        
        return base_cost * anxiety_factor
    
    def execute_action(self, action: int, env, anxiety: float = 0.0) -> Dict[str, Any]:
        """
        执行动作
        
        Args:
            action: 动作索引
            env: 环境对象，必须有step方法
            anxiety: 焦虑水平
            
        Returns:
            Dict: 执行结果
        """
        # 计算能量消耗
        energy_cost = self.get_energy_cost(action, anxiety)
        
        # 检查动作执行是否失败（受焦虑影响）
        failure_chance = self.failure_prob * (1.0 + anxiety)
        action_failed = np.random.random() < failure_chance
        
        # 执行动作
        if action_failed:
            # 执行失败，状态不变，但仍消耗能量
            next_state = env.agent_pos
            reward = -0.1  # 失败惩罚
            done = False
            
            # 记录失败
            self.stats["failed_actions"] += 1
        else:
            # 正常执行动作
            next_state, reward, done = env.step(action)
            
            # 记录成功
            self.stats["successful_actions"] += 1
        
        # 生成动作结果
        action_result = {
            "action": action,
            "energy_cost": energy_cost,
            "next_state": next_state,
            "reward": reward,
            "done": done,
            "failed": action_failed,
            "timestamp": np.datetime64('now')
        }
        
        # 更新统计信息
        self.stats["actions_executed"] += 1
        self.stats["total_energy_consumed"] += energy_cost
        
        # 更新当前动作和历史
        self.current_action = action_result
        self.action_history.append(action_result)
        
        return action_result
    
    def get_current_action(self) -> Dict[str, Any]:
        """
        获取当前动作
        
        Returns:
            Dict: 当前动作信息
        """
        return self.current_action if self.current_action else {}
    
    def get_action_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取动作历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 动作历史记录
        """
        return self.action_history[-limit:] if limit > 0 else self.action_history.copy()
    
    def get_success_rate(self) -> float:
        """
        获取动作成功率
        
        Returns:
            float: 动作成功率
        """
        total = self.stats["actions_executed"]
        if total == 0:
            return 0.0
        return self.stats["successful_actions"] / total
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        stats["success_rate"] = self.get_success_rate()
        
        if stats["actions_executed"] > 0:
            stats["avg_energy_per_action"] = stats["total_energy_consumed"] / stats["actions_executed"]
        else:
            stats["avg_energy_per_action"] = 0.0
            
        return stats 