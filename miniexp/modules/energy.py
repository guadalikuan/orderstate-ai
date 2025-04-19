import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class EnergyModule:
    """
    能量与焦虑管理模块。
    负责智能体的能量消耗、恢复和焦虑水平计算。
    
    在八阶段序态循环中，本模块负责"能量→信号"的起始阶段和"决策→行动→能量"的收尾阶段，
    形成完整的闭环。模拟生物体内能量系统如何驱动整个认知决策过程。
    
    核心概念：
    - 能量约束：所有行动和认知活动都需要消耗能量
    - 能量衰减：随时间自然衰减，模拟基础代谢
    - 焦虑机制：能量低于阈值时产生焦虑，影响决策和注意力
    - 行动成本：不同行动消耗不同能量
    - 资源探索：寻找食物以恢复能量
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
            initial_energy (float, optional): 初始能量值，智能体开始时的能量储备，默认为100.0。
            max_energy (float, optional): 最大能量值，能量上限，模拟生物体能量储存能力，默认为100.0。
            min_energy (float, optional): 最小能量值，能量下限，通常为0表示能量耗尽，默认为0.0。
            energy_decay_rate (float, optional): 能量衰减率，每步骤自然消耗的能量，模拟基础代谢，默认为0.5。
            energy_recovery_rate (float, optional): 能量恢复率，获取食物后恢复的能量量，默认为2.0。
            anxiety_threshold (float, optional): 焦虑阈值，低于此能量水平开始产生焦虑，默认为30.0。
            anxiety_sensitivity (float, optional): 焦虑敏感度，能量降低时焦虑增加的速率，默认为1.5。
            anxiety_recovery_rate (float, optional): 焦虑恢复率，能量充足时焦虑降低的速率，默认为0.2。
            random_seed (Optional[int], optional): 随机种子，用于确保实验可重复性，默认为None。
            
        注意：
            - 参数配置决定了智能体的"生存压力"大小
            - energy_decay_rate越大，生存压力越大
            - anxiety_threshold越高，智能体越早开始产生焦虑反应
            - anxiety_sensitivity越高，能量不足时焦虑增长越快
        """
        # 能量参数设置 - 控制能量系统的基本特性
        self.initial_energy = initial_energy      # 初始/重置时的能量值
        self.max_energy = max_energy              # 能量上限，不会超过此值
        self.min_energy = min_energy              # 能量下限，不会低于此值
        self.energy_decay_rate = energy_decay_rate  # 每步骤自然消耗的能量
        self.energy_recovery_rate = energy_recovery_rate  # 获取食物时的能量恢复量
        
        # 焦虑参数设置 - 控制生存焦虑如何产生和变化
        self.anxiety_threshold = anxiety_threshold  # 低于此能量水平开始产生焦虑
        self.anxiety_sensitivity = anxiety_sensitivity  # 焦虑增长的敏感度/速率
        self.anxiety_recovery_rate = anxiety_recovery_rate  # 焦虑降低的速率
        
        # 随机种子设置 - 用于保证实验的可重复性
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 状态初始化 - 设置初始能量和焦虑状态
        self.energy = initial_energy          # 当前能量水平
        self.anxiety = 0.0                    # 当前焦虑水平(0-1)，初始为0(无焦虑)
        self.energy_history = [initial_energy]  # 能量变化历史记录
        self.anxiety_history = [0.0]           # 焦虑变化历史记录
        
        # 监控数据 - 用于跟踪和分析能量系统的关键指标
        self.stats = {
            "low_energy_count": 0,   # 处于低能量状态的次数（能量低于焦虑阈值）
            "high_anxiety_count": 0, # 处于高焦虑状态的次数
            "starved_count": 0,      # 能量耗尽的次数
            "food_consumed": 0       # 摄取食物的次数
        }
    
    def reset(self) -> None:
        """
        重置能量模块到初始状态。
        
        在每次新的实验或回合开始时调用此方法，将所有能量参数恢复到默认值。
        重置内容包括：
        - 能量水平恢复到初始值
        - 焦虑水平重置为0
        - 清空历史记录
        - 重置统计指标
        """
        # 恢复初始能量状态
        self.energy = self.initial_energy  # 重置当前能量为初始值
        self.anxiety = 0.0                # 重置焦虑为0（无焦虑状态）
        
        # 清空历史记录，仅保留初始值
        self.energy_history = [self.initial_energy]
        self.anxiety_history = [0.0]
        
        # 重置所有统计计数器
        self.stats = {
            "low_energy_count": 0,    # 重置低能量计数
            "high_anxiety_count": 0,  # 重置高焦虑计数
            "starved_count": 0,       # 重置能量耗尽计数
            "food_consumed": 0        # 重置食物摄取计数
        }
    
    def update(self, reward: float, action_cost: float = 0.0) -> Dict[str, float]:
        """
        更新能量状态，这是能量动态调整的核心方法。
        模拟生物体能量消耗和焦虑产生的过程，是"行动→能量→信号"阶段的关键实现。

        Args:
            reward (float): 环境奖励值，表示从环境获得的反馈。
                - 正值：通常表示获取了资源（如食物）
                - 负值：通常表示遇到了危险或做出了不良决策
                - 0：中性反馈，无特殊资源获取
            action_cost (float, optional): 动作能量消耗，表示执行特定动作的额外能量花费，默认为0.0。
                - 不同动作可能有不同的能量消耗
                - 更复杂或剧烈的动作消耗更多能量

        Returns:
            Dict[str, float]: 更新后的能量相关指标，包括：
                - energy: 当前能量水平
                - anxiety: 当前焦虑水平
                - energy_change: 本次更新的能量变化量
                - is_low_energy: 是否处于低能量状态
                - is_starving: 是否处于极低能量(濒临耗尽)状态
        """
        # 计算能量变化 - 基于奖励、动作成本和自然衰减
        # 能量变化 = 环境奖励 - 动作消耗 - 基础代谢消耗
        energy_change = reward - action_cost - self.energy_decay_rate
        
        # 食物恢复能量 - 当奖励大于特定阈值时，视为获取了食物资源
        if reward > 1.0:  # 假设大于1的奖励是来自食物
            # 食物提供额外的能量恢复
            energy_change += self.energy_recovery_rate
            # 记录食物获取次数
            self.stats["food_consumed"] += 1
        
        # 更新能量值 - 确保在最小值和最大值之间
        # max()确保不低于最小能量，min()确保不超过最大能量
        self.energy = max(min(self.energy + energy_change, self.max_energy), self.min_energy)
        
        # 计算焦虑水平 - 基于当前能量状态
        # 焦虑是能量低于阈值时产生的，且随能量降低而增加
        if self.energy < self.anxiety_threshold:
            # 计算能量不足比例：越接近能量耗尽，比例越高
            # 能量比例 = (焦虑阈值 - 当前能量) / 焦虑阈值
            energy_ratio = (self.anxiety_threshold - self.energy) / self.anxiety_threshold
            
            # 焦虑增长量与能量不足比例和焦虑敏感度相关
            # 能量越低，焦虑增长越快
            anxiety_increase = energy_ratio * self.anxiety_sensitivity
            
            # 更新焦虑值，确保不超过1.0（最大焦虑）
            self.anxiety = min(self.anxiety + anxiety_increase, 1.0)
            
            # 更新统计数据，记录低能量和高焦虑状态
            self.stats["low_energy_count"] += 1  # 记录低能量状态
            
            if self.anxiety > 0.7:  # 高焦虑阈值
                self.stats["high_anxiety_count"] += 1  # 记录高焦虑状态
                
            if self.energy <= self.min_energy:
                self.stats["starved_count"] += 1  # 记录能量耗尽状态
        else:
            # 能量充足时焦虑逐渐恢复/降低
            # 随着能量回复，焦虑水平以固定速率下降
            self.anxiety = max(0.0, self.anxiety - self.anxiety_recovery_rate)
        
        # 记录历史数据，用于后续分析和可视化
        self.energy_history.append(self.energy)      # 记录当前能量
        self.anxiety_history.append(self.anxiety)    # 记录当前焦虑
        
        # 返回当前状态的完整信息
        return {
            "energy": self.energy,         # 当前能量水平
            "anxiety": self.anxiety,       # 当前焦虑水平
            "energy_change": energy_change,  # 本次能量变化量
            "is_low_energy": self.energy < self.anxiety_threshold,  # 是否处于低能量状态
            "is_starving": self.energy <= 10.0  # 是否处于极低能量状态（濒临耗尽）
        }
    
    def get_survival_priority(self) -> float:
        """
        计算当前的生存优先级。
        生存优先级决定智能体对能量资源的探索程度。
        这是八阶段循环中"信号→数据"阶段的关键，将焦虑转化为对环境的感知偏好。

        Returns:
            float: 生存优先级比例 (0.0-1.0)。
                - 0.0: 无生存压力，可以自由探索
                - 1.0: 最高生存压力，必须专注于寻找能量资源
        """
        # 生存优先级由焦虑水平直接决定
        # 焦虑是能量状态的函数，而生存优先级是焦虑的函数
        # 能量越低→焦虑越高→生存优先级越高→越专注于寻找食物
        return self.anxiety  # 直接返回当前焦虑水平作为生存优先级
    
    def compute_action_priority(self, action_values: np.ndarray) -> np.ndarray:
        """
        基于生存优先级调整动作优先级。
        实现八阶段循环中"知识→智慧→决策"阶段的能量影响：焦虑水平影响最终决策偏好。
        
        在高焦虑状态下，智能体会更倾向于选择能获取能量资源的动作，
        即使这些动作在其他方面可能不是最优的。

        Args:
            action_values (np.ndarray): 原始动作价值估计，通常来自价值评估或策略网络。
                - 一维数组，每个元素对应一个动作的预期价值
                - 例如：[动作0价值, 动作1价值, 动作2价值, ...]

        Returns:
            np.ndarray: 经过生存优先级调整后的动作价值。
                - 在高焦虑状态下，寻找食物的动作价值被提升
                - 在低焦虑状态下，保持原始价值评估
        """
        # 获取当前生存优先级（基于焦虑水平）
        survival_priority = self.get_survival_priority()
        
        # 生存优先级权重调整
        # 焦虑越高，智能体越倾向于选择寻找食物的动作
        # 这是生存焦虑对决策过程的直接影响
        if action_values.size > 0:  # 确保动作数组非空
            # 获取最值得探索的动作（假设是寻找食物的动作）
            # 注：在实际系统中，可能需要更复杂的方法来识别寻找食物的动作
            food_seeking_action = np.argmax(action_values)
            
            # 创建调整后的动作价值数组（初始为原始值的副本）
            adjusted_values = action_values.copy()
            
            # 根据焦虑水平调整行动偏好
            if survival_priority > 0.3:  # 中高焦虑阈值
                # 计算增强系数：焦虑越高，增强越大
                boost_factor = 1.0 + survival_priority
                
                # 增强寻找食物动作的价值
                # 这会增加智能体选择该动作的概率
                adjusted_values[food_seeking_action] *= boost_factor
            
            return adjusted_values
        
        # 如果动作数组为空，直接返回原始值
        return action_values
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取能量模块的指标数据，用于分析和可视化。
        收集当前状态和历史统计信息，便于理解能量和焦虑的动态变化。

        Returns:
            Dict[str, Any]: 包含各种能量指标的字典，包括：
                - 当前能量和焦虑水平
                - 完整的能量和焦虑历史记录
                - 统计指标（低能量次数、高焦虑次数等）
        """
        # 返回包含所有相关指标的字典
        return {
            "current_energy": self.energy,  # 当前能量水平
            "current_anxiety": self.anxiety,  # 当前焦虑水平
            "energy_history": self.energy_history.copy(),  # 能量历史记录
            "anxiety_history": self.anxiety_history.copy(),  # 焦虑历史记录
            "stats": self.stats.copy()  # 统计指标
        }
    
    def get_state_features(self) -> np.ndarray:
        """
        获取能量状态特征向量，用于智能体观察。
        将当前能量状态转换为特征向量，可以作为智能体状态的一部分。

        Returns:
            np.ndarray: 能量状态特征向量，包含4个元素：
                - 第1个元素：归一化的能量水平(0-1)
                - 第2个元素：当前焦虑水平(0-1)
                - 第3个元素：能量变化趋势(短期)
                - 第4个元素：是否处于低能量状态(二元值0或1)
        """
        # 规范化能量值，将能量转换到0-1范围
        # 这样便于与其他特征整合，保持一致的数值范围
        norm_energy = self.energy / self.max_energy  # 归一化能量值(0-1)
        
        # 计算能量变化趋势（短期）
        # 负值表示能量下降趋势，正值表示能量上升趋势
        energy_trend = 0.0  # 默认为0（无变化）
        if len(self.energy_history) >= 3:
            # 取最近3个时间步的能量记录
            recent = self.energy_history[-3:]
            if len(recent) >= 2:
                # 计算平均变化率（斜率）
                # (最新值-最早值)/(时间步数-1)，再归一化
                energy_trend = (recent[-1] - recent[0]) / (len(recent) - 1) / self.max_energy
        
        # 返回4维特征向量
        return np.array([
            norm_energy,          # 当前能量水平（归一化）
            self.anxiety,         # 当前焦虑水平
            energy_trend,         # 能量变化趋势（短期）
            float(self.energy < self.anxiety_threshold)  # 是否处于低能量状态（二元指标）
        ], dtype=np.float32)

# 测试代码
if __name__ == '__main__':
    # 创建能量模块实例
    energy_module = EnergyModule(
        initial_energy=100,     # 初始能量为满值
        max_energy=100,         # 最大能量为100
        min_energy=0,           # 最小能量为0
        energy_decay_rate=0.5,  # 每步衰减0.5能量
        energy_recovery_rate=2.0,  # 食物恢复2.0能量
        anxiety_threshold=30,   # 能量低于30时开始产生焦虑
        anxiety_sensitivity=1.5,  # 焦虑敏感度为1.5
        anxiety_recovery_rate=0.2  # 焦虑每步恢复0.2
    )
    print(f"初始能量: {energy_module.energy}, 焦虑度: {energy_module.anxiety}")
    
    # 测试能量消耗和焦虑度变化
    for i in range(5):
        # 模拟获得奖励为10的情况（如获取食物）
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