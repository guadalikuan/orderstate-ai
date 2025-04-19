import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class EnergyModule:
    """
    能量与焦虑管理模块。
    负责智能体的能量消耗、恢复和焦虑水平计算。
    """
    
    def __init__(self, 
                 initial_energy: float = 100.0,
                 max_energy: float = 100.0,
                 min_energy: float = 0.0,
                 energy_decay_rate: float = 0.5,
                 energy_recovery_rate: float = 2.0,
                 anxiety_threshold: float = 30.0,
                 anxiety_sensitivity: float = 1.5,
                 anxiety_recovery_rate: float = 0.2,
                 random_seed: Optional[int] = None):
        """
        初始化能量模块。

        Args:
            initial_energy (float, optional): 初始能量值，默认为100.0。
            max_energy (float, optional): 最大能量值，默认为100.0。
            min_energy (float, optional): 最小能量值，默认为0.0。
            energy_decay_rate (float, optional): 能量衰减率，默认为0.5。
            energy_recovery_rate (float, optional): 能量恢复率，默认为2.0。
            anxiety_threshold (float, optional): 焦虑阈值，默认为30.0。
            anxiety_sensitivity (float, optional): 焦虑敏感度，默认为1.5。
            anxiety_recovery_rate (float, optional): 焦虑恢复率，默认为0.2。
            random_seed (Optional[int], optional): 随机种子，默认为None。
        """
        # 能量参数
        self.initial_energy = initial_energy
        self.max_energy = max_energy
        self.min_energy = min_energy
        self.energy_decay_rate = energy_decay_rate
        self.energy_recovery_rate = energy_recovery_rate
        
        # 焦虑参数
        self.anxiety_threshold = anxiety_threshold
        self.anxiety_sensitivity = anxiety_sensitivity
        self.anxiety_recovery_rate = anxiety_recovery_rate
        
        # 随机种子设置
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 状态初始化
        self.energy = initial_energy
        self.anxiety = 0.0
        self.energy_history = [initial_energy]
        self.anxiety_history = [0.0]
        
        # 监控数据
        self.stats = {
            "low_energy_count": 0,
            "high_anxiety_count": 0,
            "starved_count": 0,
            "food_consumed": 0
        }
    
    def reset(self) -> None:
        """
        重置能量模块到初始状态。
        """
        self.energy = self.initial_energy
        self.anxiety = 0.0
        self.energy_history = [self.initial_energy]
        self.anxiety_history = [0.0]
        self.stats = {
            "low_energy_count": 0,
            "high_anxiety_count": 0,
            "starved_count": 0,
            "food_consumed": 0
        }
    
    def update(self, reward: float, action_cost: float = 0.0) -> Dict[str, float]:
        """
        更新能量状态。

        Args:
            reward (float): 环境奖励值。
            action_cost (float, optional): 动作能量消耗，默认为0.0。

        Returns:
            Dict[str, float]: 更新后的能量相关指标。
        """
        # 计算能量变化
        energy_change = reward - action_cost - self.energy_decay_rate
        
        # 食物恢复能量
        if reward > 1.0:  # 假设大于1的奖励是来自食物
            energy_change += self.energy_recovery_rate
            self.stats["food_consumed"] += 1
        
        # 更新能量值
        self.energy = max(min(self.energy + energy_change, self.max_energy), self.min_energy)
        
        # 计算焦虑水平
        # 焦虑是能量低于阈值时产生的，且随能量降低而增加
        if self.energy < self.anxiety_threshold:
            # 能量越低，焦虑增长越快
            energy_ratio = (self.anxiety_threshold - self.energy) / self.anxiety_threshold
            anxiety_increase = energy_ratio * self.anxiety_sensitivity
            
            # 焦虑增长
            self.anxiety = min(self.anxiety + anxiety_increase, 1.0)
            
            # 统计数据
            self.stats["low_energy_count"] += 1
            if self.anxiety > 0.7:  # 高焦虑阈值
                self.stats["high_anxiety_count"] += 1
            if self.energy <= self.min_energy:
                self.stats["starved_count"] += 1
        else:
            # 能量充足时焦虑逐渐恢复
            self.anxiety = max(0.0, self.anxiety - self.anxiety_recovery_rate)
        
        # 记录历史
        self.energy_history.append(self.energy)
        self.anxiety_history.append(self.anxiety)
        
        # 返回当前状态
        return {
            "energy": self.energy,
            "anxiety": self.anxiety,
            "energy_change": energy_change,
            "is_low_energy": self.energy < self.anxiety_threshold,
            "is_starving": self.energy <= 10.0  # 极低能量
        }
    
    def get_survival_priority(self) -> float:
        """
        计算当前的生存优先级。
        生存优先级决定智能体对能量资源的探索程度。

        Returns:
            float: 生存优先级比例 (0.0-1.0)。
        """
        # 生存优先级由焦虑水平直接决定
        # 能量越低，焦虑越高，生存优先级越高
        return self.anxiety
    
    def compute_action_priority(self, action_values: np.ndarray) -> np.ndarray:
        """
        基于生存优先级调整动作优先级。

        Args:
            action_values (np.ndarray): 原始动作价值估计。

        Returns:
            np.ndarray: 经过生存优先级调整后的动作价值。
        """
        # 获取当前生存优先级
        survival_priority = self.get_survival_priority()
        
        # 生存优先级权重
        # 焦虑越高，智能体越倾向于寻找食物
        # 在高焦虑状态下，食物相关动作的价值被增强
        # 在低焦虑状态下，保持原始决策
        if action_values.size > 0:
            # 获取最值得探索的动作（假设是寻找食物）
            # 在实际应用中，这可能需要更复杂的启发式方法
            food_seeking_action = np.argmax(action_values)
            
            # 创建调整后的动作价值数组
            adjusted_values = action_values.copy()
            
            # 根据焦虑水平调整
            if survival_priority > 0.3:  # 中高焦虑阈值
                # 增强寻找食物动作的价值
                boost_factor = 1.0 + survival_priority
                adjusted_values[food_seeking_action] *= boost_factor
            
            return adjusted_values
        
        return action_values
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取能量模块的指标数据。

        Returns:
            Dict[str, Any]: 能量模块的指标。
        """
        return {
            "current_energy": self.energy,
            "current_anxiety": self.anxiety,
            "energy_history": self.energy_history.copy(),
            "anxiety_history": self.anxiety_history.copy(),
            "stats": self.stats.copy()
        }
    
    def get_state_features(self) -> np.ndarray:
        """
        获取能量状态特征向量，用于智能体观察。

        Returns:
            np.ndarray: 能量状态特征向量。
        """
        # 规范化能量值
        norm_energy = self.energy / self.max_energy
        
        # 能量变化趋势（短期）
        energy_trend = 0.0
        if len(self.energy_history) >= 3:
            recent = self.energy_history[-3:]
            if len(recent) >= 2:
                energy_trend = (recent[-1] - recent[0]) / (len(recent) - 1) / self.max_energy
        
        return np.array([
            norm_energy,          # 当前能量水平（归一化）
            self.anxiety,         # 当前焦虑水平
            energy_trend,         # 能量变化趋势
            float(self.energy < self.anxiety_threshold)  # 是否处于低能量状态
        ], dtype=np.float32)

# 测试代码
if __name__ == '__main__':
    energy_module = EnergyModule(initial_energy=100, max_energy=100, min_energy=0, energy_decay_rate=0.5, energy_recovery_rate=2.0, anxiety_threshold=30, anxiety_sensitivity=1.5, anxiety_recovery_rate=0.2)
    print(f"初始能量: {energy_module.energy}, 焦虑度: {energy_module.anxiety}")
    
    # 测试能量消耗和焦虑度变化
    for i in range(5):
        energy_module.update(10)
        print(f"剩余能量: {energy_module.energy}, 焦虑度: {energy_module.anxiety}")
    
    # 测试能量回复对焦虑的影响
    print("\n测试能量回复:")
    energy_module.update(10)
    print(f"能量回复后: {energy_module.energy}, 焦虑度: {energy_module.anxiety}")
    
    # 测试适应性
    print("\n测试焦虑适应性:")
    for i in range(3):
        print(f"适应周期 {i+1}, 基准焦虑: {energy_module.anxiety:.2f}")
        # 在同一能量水平停留一段时间
        for j in range(3):
            print(f"  时间点 {j+1}, 焦虑度: {energy_module.anxiety:.2f}")
            energy_module.update(10) 