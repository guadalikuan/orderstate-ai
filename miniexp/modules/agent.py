import numpy as np
from typing import Optional, List, Dict, Any, Tuple

from miniexp.modules.energy import EnergyModule
from miniexp.modules.attention import AttentionModule

class BaseAgent:
    """
    基础智能体类。
    能够感知环境并做出反应（无焦虑基线）。
    """
    def __init__(self, obs_dim: int, action_dim: int = 4, random_seed: Optional[int] = None):
        """
        初始化基础智能体。

        Args:
            obs_dim (int): 观察空间维度。
            action_dim (int, optional): 动作空间维度，默认为4。
            random_seed (Optional[int], optional): 随机种子，默认为None。
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 初始化注意力模块 - 将观察转化为动作概率
        self.attention = AttentionModule(feature_dim=obs_dim)
        
        # 智能体状态
        self.position = None  # 当前位置(x, y)
        self.prev_position = None  # 上一步位置
        self.last_action = None  # 上一步动作
        self.cumulative_reward = 0  # 累积奖励
        
        # 性能指标
        self.metrics = {
            "steps": 0,
            "rewards": [],
            "positions": [],
            "actions": []
        }
        
    def reset(self, initial_position: Tuple[int, int] = (0, 0)) -> None:
        """
        重置智能体状态。

        Args:
            initial_position (Tuple[int, int], optional): 初始位置，默认为(0, 0)。
        """
        self.position = initial_position
        self.prev_position = initial_position
        self.last_action = None
        self.cumulative_reward = 0
        
        # 重置指标
        self.metrics = {
            "steps": 0,
            "rewards": [],
            "positions": [],
            "actions": []
        }
        
    def observe(self, observation: np.ndarray) -> np.ndarray:
        """
        处理观察，返回注意力权重。

        Args:
            observation (np.ndarray): 环境观察。

        Returns:
            np.ndarray: 注意力权重。
        """
        # [阶段 2 -> 3: 感知(环境观察) -> 数据(特征)]
        return self.attention.compute_attention(observation)
    
    def act(self, observation: np.ndarray) -> int:
        """
        基于观察选择动作。

        Args:
            observation (np.ndarray): 环境观察。

        Returns:
            int: 选择的动作。
        """
        # 计算注意力权重
        attention_weights = self.observe(observation)
        
        # [阶段 4 -> 5: 信息(注意力权重) -> 决策(动作选择)]
        # 根据注意力权重选择动作（概率采样）
        action = np.random.choice(self.action_dim, p=attention_weights)
        
        # 更新状态
        self.last_action = action
        self.prev_position = self.position
        
        # 更新指标
        self.metrics["steps"] += 1
        self.metrics["actions"].append(action)
        
        return action
    
    def receive_reward(self, reward: float) -> None:
        """
        接收奖励并更新智能体状态。

        Args:
            reward (float): 奖励值。
        """
        self.cumulative_reward += reward
        self.metrics["rewards"].append(reward)
    
    def update_position(self, new_position: Tuple[int, int]) -> None:
        """
        更新智能体位置。

        Args:
            new_position (Tuple[int, int]): 新位置。
        """
        self.prev_position = self.position
        self.position = new_position
        self.metrics["positions"].append(new_position)
        
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取智能体性能指标。

        Returns:
            Dict[str, Any]: 性能指标字典。
        """
        return self.metrics

class EnergyAgent(BaseAgent):
    """
    带能量管理的智能体。
    在基础智能体基础上增加能量系统和生存焦虑机制。
    """
    def __init__(self, obs_dim: int, action_dim: int = 4, random_seed: Optional[int] = None):
        """
        初始化带能量管理的智能体。

        Args:
            obs_dim (int): 观察空间维度。
            action_dim (int, optional): 动作空间维度，默认为4。
            random_seed (Optional[int], optional): 随机种子，默认为None。
        """
        super().__init__(obs_dim, action_dim, random_seed)
        
        # 增加能量模块
        self.energy_module = EnergyModule()
        
        # 扩展指标收集
        self.metrics.update({
            "energy_levels": [],
            "anxiety_levels": [],
            "anxiety_states": [],
            "attention_entropy": []
        })
        
        # 策略调整参数
        self.survival_priority = 0.0  # 生存优先级，随焦虑上升
        self.exploration_rate = 0.1  # 初始探索率
        self.consecutive_rewards = 0  # 连续获得奖励的次数
        self.consecutive_penalties = 0  # 连续受到惩罚的次数
        
        # 行为适应参数
        self.visited_positions = set()  # 记录已访问位置
        self.revisit_count = {}  # 位置重访计数
        self.stuck_count = 0  # 卡住计数
        
    def reset(self, initial_position: Tuple[int, int] = (0, 0)) -> None:
        """
        重置智能体状态。

        Args:
            initial_position (Tuple[int, int], optional): 初始位置，默认为(0, 0)。
        """
        super().reset(initial_position)
        self.energy_module.reset()
        
        # 重置能量相关指标
        self.metrics.update({
            "energy_levels": [],
            "anxiety_levels": [],
            "anxiety_states": [],
            "attention_entropy": []
        })
        
        # 重置行为适应参数
        self.survival_priority = 0.0
        self.exploration_rate = 0.1
        self.consecutive_rewards = 0
        self.consecutive_penalties = 0
        self.visited_positions = set([initial_position])
        self.revisit_count = {initial_position: 1}
        self.stuck_count = 0
        
        # 重置注意力模块状态
        self.attention.reset()
        
    def observe(self, observation: np.ndarray) -> np.ndarray:
        """
        处理观察，返回受能量和焦虑影响的注意力权重。

        Args:
            observation (np.ndarray): 环境观察。

        Returns:
            np.ndarray: 注意力权重。
        """
        # 获取当前能量水平
        current_energy = self.energy_module.get_energy()
        
        # [阶段 1 -> 2: 内部状态(能量) -> 感知(焦虑影响感知)]
        # 计算焦虑度 - 能量越低，焦虑越高
        anxiety = self.energy_module.get_anxiety()
        anxiety_state = self.energy_module.get_anxiety_state()
        
        # 更新生存优先级
        self._update_survival_priority(anxiety, anxiety_state)
        
        # [阶段 2 -> 3: 感知(环境观察+焦虑) -> 数据(特征+焦虑)]
        # 计算受焦虑影响的注意力权重
        attention_weights = self.attention.compute_attention(observation, anxiety)
        
        # 记录指标
        entropy = self.attention.get_attention_entropy(attention_weights)
        self.metrics["energy_levels"].append(current_energy)
        self.metrics["anxiety_levels"].append(anxiety)
        self.metrics["anxiety_states"].append(anxiety_state)
        self.metrics["attention_entropy"].append(entropy)
        
        return attention_weights
    
    def act(self, observation: np.ndarray) -> int:
        """
        基于观察和内部状态（能量、焦虑）选择动作。

        Args:
            observation (np.ndarray): 环境观察。

        Returns:
            int: 选择的动作。
        """
        # 消耗能量 - 生存成本
        self.energy_module.consume_energy(self.energy_module.PASSIVE_CONSUMPTION)
        
        # 计算受焦虑影响的注意力权重
        attention_weights = self.observe(observation)
        
        # [阶段 4 -> 5: 信息(注意力权重) -> 决策(动作选择)]
        # 根据焦虑和生存优先级调整行为
        action = self._select_action_with_survival_priority(attention_weights, observation)
        
        # 消耗能量 - 动作成本
        self.energy_module.consume_energy(self.energy_module.ACTION_CONSUMPTION)
        
        # 更新状态
        self.last_action = action
        self.prev_position = self.position
        
        # 更新指标
        self.metrics["steps"] += 1
        self.metrics["actions"].append(action)
        
        return action
        
    def _select_action_with_survival_priority(self, attention_weights: np.ndarray, 
                                             observation: np.ndarray) -> int:
        """
        根据生存优先级选择动作。

        Args:
            attention_weights (np.ndarray): 注意力权重。
            observation (np.ndarray): 当前观察。

        Returns:
            int: 选择的动作。
        """
        anxiety = self.energy_module.get_anxiety()
        anxiety_state = self.energy_module.get_anxiety_state()
        
        # 当处于极度焦虑状态时，增加找食物的可能性
        if anxiety_state == "CRITICAL" or anxiety > 1.5:
            # 假设观察中包含食物信息
            if len(observation) >= 4:
                food_position = observation[2:4]  # 假设观察的第3-4个元素是食物相对位置
                
                # 如果明确知道食物方向，强制向食物移动
                dx, dy = food_position
                food_direction = -1
                
                # 确定食物方向 [上, 下, 左, 右]
                if abs(dy) > abs(dx):  # 垂直距离更远
                    food_direction = 0 if dy < 0 else 1  # 上或下
                else:  # 水平距离更远
                    food_direction = 2 if dx < 0 else 3  # 左或右
                
                # 高概率选择食物方向
                if food_direction >= 0 and np.random.random() < self.survival_priority:
                    return food_direction
        
        # 根据焦虑状态调整探索与利用的平衡
        if anxiety_state == "SAFE":
            # 安全状态，增加探索
            if np.random.random() < self.exploration_rate:
                return np.random.randint(self.action_dim)
        
        # 检测是否陷入循环
        if self.stuck_count > 3:
            # 陷入循环时，随机选择未尝试的动作
            self.stuck_count = 0
            return np.random.randint(self.action_dim)
        
        # 默认情况：根据注意力权重概率选择
        return np.random.choice(self.action_dim, p=attention_weights)
    
    def _update_survival_priority(self, anxiety: float, anxiety_state: str) -> None:
        """
        更新生存优先级。

        Args:
            anxiety (float): 当前焦虑度。
            anxiety_state (str): 当前焦虑状态。
        """
        # 基于焦虑状态设置生存优先级基础值
        if anxiety_state == "SAFE":
            base_priority = 0.1
        elif anxiety_state == "ALERT":
            base_priority = 0.3
        elif anxiety_state == "DANGER":
            base_priority = 0.6
        else:  # CRITICAL
            base_priority = 0.9
        
        # 根据负面体验(连续惩罚)调整优先级
        priority_adjustment = min(0.3, self.consecutive_penalties * 0.05)
        
        # 更新生存优先级
        self.survival_priority = min(0.95, base_priority + priority_adjustment)
        
        # 调整探索率 - 焦虑越高，探索率越低
        self.exploration_rate = max(0.01, 0.1 - anxiety * 0.05)
    
    def update_position(self, new_position: Tuple[int, int]) -> None:
        """
        更新智能体位置并检测循环行为。

        Args:
            new_position (Tuple[int, int]): 新位置。
        """
        super().update_position(new_position)
        
        # 更新位置访问记录
        if new_position in self.visited_positions:
            if new_position in self.revisit_count:
                self.revisit_count[new_position] += 1
            else:
                self.revisit_count[new_position] = 1
                
            # 检测是否在同一位置徘徊
            if new_position == self.prev_position:
                self.stuck_count += 1
            else:
                self.stuck_count = 0
        else:
            self.visited_positions.add(new_position)
            self.revisit_count[new_position] = 1
            self.stuck_count = 0
    
    def receive_reward(self, reward: float) -> None:
        """
        接收奖励并更新智能体状态。

        Args:
            reward (float): 奖励值。
        """
        super().receive_reward(reward)
        
        # 根据奖励回复能量
        if reward > 0:
            # 积极奖励 - 回复能量（如找到食物）
            self.energy_module.recover_energy(reward * 2)
            self.consecutive_rewards += 1
            self.consecutive_penalties = 0
        elif reward < 0:
            # 负面奖励 - 额外消耗能量（如触碰障碍）
            self.energy_module.consume_energy(abs(reward))
            self.consecutive_rewards = 0
            self.consecutive_penalties += 1
        else:
            # 中性奖励 - 重置连续计数
            self.consecutive_rewards = 0
            self.consecutive_penalties = 0
            
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取智能体性能指标。

        Returns:
            Dict[str, Any]: 性能指标字典。
        """
        metrics = super().get_metrics()
        
        # 添加能量和焦虑相关指标
        metrics.update({
            "energy_levels": self.metrics["energy_levels"],
            "anxiety_levels": self.metrics["anxiety_levels"],
            "anxiety_states": self.metrics["anxiety_states"],
            "attention_entropy": self.metrics["attention_entropy"],
            "survival_priority": self.survival_priority,
            "exploration_rate": self.exploration_rate
        })
        
        return metrics 