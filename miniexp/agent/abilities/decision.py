from typing import Dict, Any, List, Tuple
import numpy as np

class DecisionAbility:
    """
    决策能力模块
    管理智能体的决策质量和决策速度
    """
    
    def __init__(self, level: int = 1):
        """
        初始化决策能力
        
        Args:
            level: 决策能力等级 (1-5)
        """
        self.level = level
        self.quality = self._get_quality()
        self.speed = self._get_speed()
        self.memory_size = self._get_memory_size()
        self.memory = []
        
    def _get_quality(self) -> float:
        """
        根据等级获取决策质量
        """
        qualities = {
            1: 0.6,  # 60%最优决策
            2: 0.7,  # 70%最优决策
            3: 0.8,  # 80%最优决策
            4: 0.9,  # 90%最优决策
            5: 1.0   # 100%最优决策
        }
        return qualities.get(self.level, 0.6)
        
    def _get_speed(self) -> int:
        """
        根据等级获取决策速度（每步考虑的步数）
        """
        speeds = {
            1: 1,  # 只考虑当前步
            2: 2,  # 考虑未来2步
            3: 3,  # 考虑未来3步
            4: 4,  # 考虑未来4步
            5: 5   # 考虑未来5步
        }
        return speeds.get(self.level, 1)
        
    def _get_memory_size(self) -> int:
        """
        根据等级获取记忆容量
        """
        sizes = {
            1: 10,   # 记住最近10步
            2: 20,   # 记住最近20步
            3: 30,   # 记住最近30步
            4: 40,   # 记住最近40步
            5: 50    # 记住最近50步
        }
        return sizes.get(self.level, 10)
        
    def upgrade(self) -> bool:
        """
        升级决策能力
        
        Returns:
            bool: 是否升级成功
        """
        if self.level < 5:
            self.level += 1
            self.quality = self._get_quality()
            self.speed = self._get_speed()
            self.memory_size = self._get_memory_size()
            return True
        return False
        
    def make_decision(self, observation: Dict[str, Any], 
                     possible_actions: List[int]) -> int:
        """
        做出决策
        
        Args:
            observation: 观察结果
            possible_actions: 可能的动作列表
            
        Returns:
            int: 选择的动作
        """
        # 获取当前状态
        grid = observation.get('grid', np.zeros((1, 1)))
        
        # 计算每个动作的价值
        action_values = []
        for action in possible_actions:
            # 模拟执行动作
            value = self._evaluate_action(action, grid)
            action_values.append(value)
            
        # 根据决策质量选择动作
        if np.random.random() < self.quality:
            # 选择最优动作
            action = possible_actions[np.argmax(action_values)]
        else:
            # 随机选择动作
            action = np.random.choice(possible_actions)
            
        # 更新记忆
        self._update_memory(action, max(action_values))
        
        return action
        
    def _evaluate_action(self, action: int, grid: np.ndarray) -> float:
        """
        评估动作价值
        
        Args:
            action: 动作
            grid: 环境网格
            
        Returns:
            float: 动作价值
        """
        # 简单启发式评估
        # 1. 找到目标位置
        target_pos = np.where(grid == 2)
        if len(target_pos[0]) == 0:
            return 0.0
            
        # 2. 计算动作后的位置
        action_map = {
            0: (-1, 0),   # 上
            1: (1, 0),    # 下
            2: (0, -1),   # 左
            3: (0, 1)     # 右
        }
        dy, dx = action_map[action]
        new_y = max(0, min(grid.shape[0]-1, np.where(grid == 1)[0][0] + dy))
        new_x = max(0, min(grid.shape[1]-1, np.where(grid == 1)[1][0] + dx))
        
        # 3. 计算到目标的距离
        distance = abs(new_y - target_pos[0][0]) + abs(new_x - target_pos[1][0])
        
        # 4. 考虑障碍物
        if grid[new_y, new_x] == 3:  # 障碍物
            return -1.0
            
        # 5. 考虑捕食者
        if grid[new_y, new_x] == 4:  # 捕食者
            return -2.0
            
        return 1.0 / (distance + 1)
        
    def _update_memory(self, action: int, value: float):
        """
        更新记忆
        
        Args:
            action: 执行的动作
            value: 动作价值
        """
        self.memory.append((action, value))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
            
    def get_metrics(self) -> Dict[str, float]:
        """
        获取决策能力指标
        
        Returns:
            Dict[str, float]: 决策能力指标
        """
        return {
            'level': self.level,
            'quality': self.quality,
            'speed': self.speed,
            'memory_size': self.memory_size,
            'memory_usage': len(self.memory) / self.memory_size
        } 