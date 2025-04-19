import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class AttentionModule:
    """
    注意力管理模块。
    负责管理智能体的注意力资源分配，包括感知注意力和决策注意力。
    注意力分配受能量状态和焦虑水平的影响。
    """
    
    def __init__(self, 
                 perception_capacity: int = 10,
                 decision_capacity: int = 5,
                 anxiety_influence_rate: float = 1.2,
                 recovery_rate: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        初始化注意力模块。

        Args:
            perception_capacity (int, optional): 感知注意力容量，默认为10。
            decision_capacity (int, optional): 决策注意力容量，默认为5。
            anxiety_influence_rate (float, optional): 焦虑影响率，默认为1.2。
            recovery_rate (float, optional): 注意力恢复率，默认为0.1。
            random_seed (Optional[int], optional): 随机种子，默认为None。
        """
        # 注意力容量参数
        self.perception_capacity = perception_capacity
        self.decision_capacity = decision_capacity
        
        # 焦虑影响参数
        self.anxiety_influence_rate = anxiety_influence_rate
        self.recovery_rate = recovery_rate
        
        # 随机种子设置
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 状态初始化
        self.perception_attention = perception_capacity
        self.decision_attention = decision_capacity
        self.attention_history = []
        
        # 注意力分配历史
        self.food_attention_weight = 0.5  # 食物注意力权重
        self.danger_attention_weight = 0.5  # 危险注意力权重
        
        # 注意力指标
        self.stats = {
            "attention_shifts": 0,  # 注意力转移次数
            "limited_perception_count": 0,  # 感知受限次数
            "limited_decision_count": 0  # 决策受限次数
        }
    
    def reset(self) -> None:
        """
        重置注意力模块到初始状态。
        """
        self.perception_attention = self.perception_capacity
        self.decision_attention = self.decision_capacity
        self.attention_history = []
        self.food_attention_weight = 0.5
        self.danger_attention_weight = 0.5
        self.stats = {
            "attention_shifts": 0,
            "limited_perception_count": 0,
            "limited_decision_count": 0
        }
    
    def update(self, anxiety: float, reward: float) -> Dict[str, float]:
        """
        更新注意力状态。

        Args:
            anxiety (float): 当前焦虑水平 (0.0-1.0)。
            reward (float): 环境奖励值。

        Returns:
            Dict[str, float]: 更新后的注意力相关指标。
        """
        # 记录上一个状态用于比较
        prev_food_attention = self.food_attention_weight
        
        # 基于焦虑调整注意力容量
        # 焦虑越高，注意力容量越低
        if anxiety > 0.3:  # 中等焦虑阈值
            # 焦虑影响感知和决策注意力
            perception_reduction = self.perception_capacity * anxiety * self.anxiety_influence_rate / 5.0
            decision_reduction = self.decision_capacity * anxiety * self.anxiety_influence_rate / 3.0
            
            # 应用注意力减少
            self.perception_attention = max(self.perception_capacity / 2, 
                                          self.perception_capacity - perception_reduction)
            self.decision_attention = max(self.decision_capacity / 2, 
                                        self.decision_capacity - decision_reduction)
            
            # 当焦虑增加时，注意力会更多地分配给食物相关信息
            # 这是生存焦虑的主要影响：高焦虑时优先寻找生存资源
            self.food_attention_weight = min(0.9, self.food_attention_weight + anxiety * 0.2)
            self.danger_attention_weight = 1.0 - self.food_attention_weight
            
            # 记录注意力受限状态
            if self.perception_attention < self.perception_capacity * 0.8:
                self.stats["limited_perception_count"] += 1
            if self.decision_attention < self.decision_capacity * 0.8:
                self.stats["limited_decision_count"] += 1
        else:
            # 低焦虑时，注意力恢复
            self.perception_attention = min(self.perception_capacity, 
                                          self.perception_attention + self.recovery_rate)
            self.decision_attention = min(self.decision_capacity, 
                                        self.decision_attention + self.recovery_rate)
            
            # 低焦虑时更平衡地分配注意力
            self.food_attention_weight = max(0.3, min(0.7, self.food_attention_weight - 0.05))
            self.danger_attention_weight = 1.0 - self.food_attention_weight
        
        # 检测注意力转移
        if abs(prev_food_attention - self.food_attention_weight) > 0.1:
            self.stats["attention_shifts"] += 1
        
        # 记录注意力历史
        self.attention_history.append({
            "perception": self.perception_attention,
            "decision": self.decision_attention,
            "food_weight": self.food_attention_weight,
            "danger_weight": self.danger_attention_weight,
            "anxiety": anxiety
        })
        
        # 返回当前状态
        return {
            "perception_attention": self.perception_attention,
            "decision_attention": self.decision_attention,
            "food_attention_weight": self.food_attention_weight,
            "danger_attention_weight": self.danger_attention_weight
        }
    
    def filter_perception(self, observation: np.ndarray, agent_position: Tuple[int, int]) -> np.ndarray:
        """
        基于当前注意力状态过滤感知信息。
        注意力有限时，智能体只能感知部分环境信息。

        Args:
            observation (np.ndarray): 原始观察 (环境网格)。
            agent_position (Tuple[int, int]): 智能体在网格中的位置。

        Returns:
            np.ndarray: 经过注意力过滤的观察。
        """
        # 如果注意力充足，返回完整观察
        if self.perception_attention >= self.perception_capacity * 0.9:
            return observation.copy()
        
        # 创建过滤后的观察
        filtered_obs = np.zeros_like(observation)
        
        # 计算可感知的范围 (基于当前感知注意力)
        # 注意力越低，感知范围越小
        perception_range = max(1, int(self.perception_attention))
        
        # 获取智能体坐标
        agent_x, agent_y = agent_position
        
        # 确定感知范围的边界
        min_x = max(0, agent_x - perception_range)
        max_x = min(observation.shape[0] - 1, agent_x + perception_range)
        min_y = max(0, agent_y - perception_range)
        max_y = min(observation.shape[1] - 1, agent_y + perception_range)
        
        # 复制可感知范围内的信息到过滤后的观察
        filtered_obs[min_x:max_x+1, min_y:max_y+1] = observation[min_x:max_x+1, min_y:max_y+1]
        
        # 在可感知范围内，基于注意力权重过滤特定类型的信息
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                # 计算与智能体的距离
                distance = np.sqrt((x - agent_x)**2 + (y - agent_y)**2)
                
                # 如果距离超过感知范围的80%，应用注意力过滤
                if distance > perception_range * 0.8:
                    # 假设观察中的1表示食物，2表示障碍物（危险）
                    if observation[x, y] == 1:  # 食物
                        # 根据食物注意力权重决定是否感知
                        if np.random.random() > self.food_attention_weight:
                            filtered_obs[x, y] = 0  # 不感知此食物
                    elif observation[x, y] == 2:  # 障碍物/危险
                        # 根据危险注意力权重决定是否感知
                        if np.random.random() > self.danger_attention_weight:
                            filtered_obs[x, y] = 0  # 不感知此障碍物
        
        return filtered_obs
    
    def filter_action_values(self, action_values: np.ndarray, anxiety: float) -> np.ndarray:
        """
        基于当前决策注意力过滤动作价值。
        决策注意力有限时，智能体可能无法准确评估所有动作的价值。

        Args:
            action_values (np.ndarray): 原始动作价值数组。
            anxiety (float): 当前焦虑水平。

        Returns:
            np.ndarray: 经过注意力过滤的动作价值。
        """
        # 如果决策注意力充足，返回原始动作价值
        if self.decision_attention >= self.decision_capacity * 0.9:
            return action_values.copy()
        
        # 创建过滤后的动作价值
        filtered_values = action_values.copy()
        
        # 计算注意力不足导致的噪声水平
        # 注意力越低，噪声越大
        attention_ratio = self.decision_attention / self.decision_capacity
        noise_level = (1.0 - attention_ratio) * 0.5  # 最大噪声为0.5
        
        # 在焦虑状态下，噪声会更大
        if anxiety > 0.5:
            noise_level *= (1.0 + anxiety)
        
        # 向动作价值添加噪声
        noise = np.random.normal(0, noise_level, size=action_values.shape)
        filtered_values += noise
        
        return filtered_values
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取注意力模块的指标数据。

        Returns:
            Dict[str, Any]: 注意力模块的指标。
        """
        # 计算平均注意力分配
        if self.attention_history:
            avg_perception = np.mean([rec["perception"] for rec in self.attention_history])
            avg_decision = np.mean([rec["decision"] for rec in self.attention_history])
            avg_food_weight = np.mean([rec["food_weight"] for rec in self.attention_history])
            avg_danger_weight = np.mean([rec["danger_weight"] for rec in self.attention_history])
        else:
            avg_perception = self.perception_capacity
            avg_decision = self.decision_capacity
            avg_food_weight = 0.5
            avg_danger_weight = 0.5
        
        return {
            "current_perception": self.perception_attention,
            "current_decision": self.decision_attention,
            "current_food_weight": self.food_attention_weight,
            "current_danger_weight": self.danger_attention_weight,
            "avg_perception": avg_perception,
            "avg_decision": avg_decision,
            "avg_food_weight": avg_food_weight,
            "avg_danger_weight": avg_danger_weight,
            "attention_history": self.attention_history.copy(),
            "stats": self.stats.copy()
        }
    
    def get_attention_features(self) -> np.ndarray:
        """
        获取注意力状态特征向量，用于智能体观察。

        Returns:
            np.ndarray: 注意力状态特征向量。
        """
        # 规范化注意力值
        norm_perception = self.perception_attention / self.perception_capacity
        norm_decision = self.decision_attention / self.decision_capacity
        
        return np.array([
            norm_perception,  # 当前感知注意力（归一化）
            norm_decision,    # 当前决策注意力（归一化）
            self.food_attention_weight,  # 食物注意力权重
            self.danger_attention_weight  # 危险注意力权重
        ], dtype=np.float32)

# 测试代码
if __name__ == '__main__':
    attention_module = AttentionModule(perception_capacity=10, decision_capacity=5)
    print(f"初始感知注意力: {attention_module.perception_attention}, 决策注意力: {attention_module.decision_attention}")
    
    # 测试不同焦虑水平对注意力的影响
    anxiety_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
    for anxiety in anxiety_levels:
        metrics = attention_module.update(anxiety, 0.0)
        print(f"\n焦虑水平 {anxiety:.1f} 的注意力状态:")
        print(f"  感知注意力: {metrics['perception_attention']:.2f}/{attention_module.perception_capacity}")
        print(f"  决策注意力: {metrics['decision_attention']:.2f}/{attention_module.decision_capacity}")
        print(f"  食物注意力权重: {metrics['food_attention_weight']:.2f}")
        print(f"  危险注意力权重: {metrics['danger_attention_weight']:.2f}")
    
    # 测试感知过滤
    print("\n测试感知过滤:")
    # 创建一个简单的10x10网格，包含智能体(3)、食物(1)和障碍物(2)
    grid = np.zeros((10, 10))
    grid[2, 3] = 1  # 食物
    grid[4, 5] = 1  # 食物
    grid[7, 8] = 1  # 食物
    grid[3, 7] = 2  # 障碍物
    grid[6, 2] = 2  # 障碍物
    agent_pos = (5, 5)
    
    # 低焦虑
    attention_module.reset()
    attention_module.update(0.1, 0.0)
    filtered_low_anxiety = attention_module.filter_perception(grid, agent_pos)
    
    # 高焦虑
    attention_module.reset()
    attention_module.update(0.8, 0.0)
    filtered_high_anxiety = attention_module.filter_perception(grid, agent_pos)
    
    print("原始网格:")
    print(grid)
    print("\n低焦虑过滤后:")
    print(filtered_low_anxiety)
    print("\n高焦虑过滤后:")
    print(filtered_high_anxiety) 