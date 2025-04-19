import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class AttentionModule:
    """
    注意力管理模块。
    负责管理智能体的注意力资源分配，包括感知注意力和决策注意力。
    注意力分配受能量状态和焦虑水平的影响。
    
    在八阶段序态循环中，本模块负责"数据→信息"阶段的处理，
    通过注意力机制过滤和聚焦于重要的感知信息，同时调整决策过程。
    
    核心概念：
    - 注意力资源有限：感知和决策都有容量限制
    - 焦虑影响注意力：高焦虑导致注意力分配更加集中，但总容量下降
    - 注意力恢复：随着焦虑降低，注意力资源可以恢复
    - 注意力偏好：在高焦虑状态下，注意力偏向于生存相关资源
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
            perception_capacity (int, optional): 感知注意力容量，表示智能体能够同时处理的环境信息量，默认为10。
            decision_capacity (int, optional): 决策注意力容量，表示智能体能够同时评估的行动选项数量，默认为5。
            anxiety_influence_rate (float, optional): 焦虑影响率，控制焦虑对注意力的影响强度，值越大影响越强，默认为1.2。
            recovery_rate (float, optional): 注意力恢复率，控制注意力在低焦虑状态下的恢复速度，默认为0.1。
            random_seed (Optional[int], optional): 随机种子，用于确保实验可重复性，默认为None。
        
        注意：
            - perception_capacity越大，智能体的感知范围越广
            - decision_capacity越大，智能体评估行动的准确性越高
            - anxiety_influence_rate控制生存焦虑对注意力的影响程度
            - recovery_rate控制智能体从高焦虑状态恢复的速度
        """
        # 注意力容量参数 - 这些是智能体的基础属性，表示理想状态下的最大注意力资源
        self.perception_capacity = perception_capacity  # 最大感知注意力容量
        self.decision_capacity = decision_capacity      # 最大决策注意力容量
        
        # 焦虑影响参数 - 控制焦虑如何影响注意力动态变化
        self.anxiety_influence_rate = anxiety_influence_rate  # 焦虑对注意力的影响强度
        self.recovery_rate = recovery_rate                   # 低焦虑时注意力的恢复速率
        
        # 随机种子设置 - 用于保证实验的可重复性
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 当前注意力状态 - 初始状态下，注意力资源处于最大容量
        self.perception_attention = perception_capacity  # 当前可用的感知注意力
        self.decision_attention = decision_capacity      # 当前可用的决策注意力
        self.attention_history = []                      # 用于记录注意力状态的历史变化
        
        # 注意力分配权重 - 控制注意力如何分配给不同类型的环境特征
        self.food_attention_weight = 0.5    # 对食物/资源类信息的注意力权重，初始为中性值
        self.danger_attention_weight = 0.5  # 对危险/障碍类信息的注意力权重，初始为中性值
        
        # 统计指标 - 用于记录和分析注意力变化的相关统计数据
        self.stats = {
            "attention_shifts": 0,           # 记录注意力显著转移的次数
            "limited_perception_count": 0,   # 记录感知注意力受限的次数
            "limited_decision_count": 0      # 记录决策注意力受限的次数
        }
    
    def reset(self) -> None:
        """
        重置注意力模块到初始状态。
        
        当开始新的实验或回合时调用此方法，将所有注意力参数恢复到默认值。
        重置内容包括：
        - 注意力资源恢复到最大容量
        - 注意力分配权重重置为平衡状态
        - 清空历史记录
        - 重置统计指标
        """
        # 恢复注意力资源到最大容量
        self.perception_attention = self.perception_capacity
        self.decision_attention = self.decision_capacity
        
        # 清空历史记录
        self.attention_history = []
        
        # 重置注意力分配为平衡状态
        self.food_attention_weight = 0.5    # 食物注意力权重恢复为中性值
        self.danger_attention_weight = 0.5  # 危险注意力权重恢复为中性值
        
        # 重置统计指标
        self.stats = {
            "attention_shifts": 0,           # 注意力转移次数
            "limited_perception_count": 0,   # 感知受限次数
            "limited_decision_count": 0      # 决策受限次数
        }
    
    def update(self, anxiety: float, reward: float) -> Dict[str, float]:
        """
        更新注意力状态，基于当前焦虑水平和环境反馈。
        这是注意力动态调整的核心方法，模拟了生物在不同焦虑水平下的注意力变化。

        Args:
            anxiety (float): 当前焦虑水平，范围为0.0-1.0，值越大表示焦虑程度越高。
                - 0.0-0.3: 低焦虑，注意力趋于恢复和平衡
                - 0.3-0.7: 中等焦虑，注意力开始受限并偏向生存资源
                - 0.7-1.0: 高焦虑，注意力严重受限并高度聚焦于生存需求
            reward (float): 环境奖励值，表示智能体上一次行动的反馈。
                - 正值：良好反馈，可能略微缓解焦虑影响
                - 负值：不良反馈，可能加剧焦虑影响

        Returns:
            Dict[str, float]: 更新后的注意力相关指标，包括：
                - perception_attention: 当前感知注意力资源
                - decision_attention: 当前决策注意力资源
                - food_attention_weight: 对食物/资源的注意力偏好
                - danger_attention_weight: 对危险/障碍的注意力偏好
        """
        # 记录上一个状态用于比较，以检测注意力显著转移
        prev_food_attention = self.food_attention_weight
        
        # 基于焦虑调整注意力容量
        # 焦虑越高，注意力容量越低，模拟生物在压力下注意力资源受限的现象
        if anxiety > 0.3:  # 中等焦虑阈值
            # 计算焦虑导致的注意力资源减少量
            # 焦虑影响感知和决策注意力，但影响程度不同
            perception_reduction = self.perception_capacity * anxiety * self.anxiety_influence_rate / 5.0
            decision_reduction = self.decision_capacity * anxiety * self.anxiety_influence_rate / 3.0
            
            # 应用注意力减少，但保证至少保留一半的注意力资源
            self.perception_attention = max(self.perception_capacity / 2, 
                                          self.perception_capacity - perception_reduction)
            self.decision_attention = max(self.decision_capacity / 2, 
                                        self.decision_capacity - decision_reduction)
            
            # 当焦虑增加时，注意力会更多地分配给食物相关信息
            # 这是生存焦虑的主要影响：高焦虑时优先寻找生存资源
            # 注意力权重上限为0.9，确保至少保留10%的注意力给其他信息
            self.food_attention_weight = min(0.9, self.food_attention_weight + anxiety * 0.2)
            self.danger_attention_weight = 1.0 - self.food_attention_weight  # 确保两类权重之和为1
            
            # 记录注意力受限状态，用于后续分析
            if self.perception_attention < self.perception_capacity * 0.8:
                self.stats["limited_perception_count"] += 1
            if self.decision_attention < self.decision_capacity * 0.8:
                self.stats["limited_decision_count"] += 1
        else:
            # 低焦虑时，注意力资源逐渐恢复，但不超过最大容量
            self.perception_attention = min(self.perception_capacity, 
                                          self.perception_attention + self.recovery_rate)
            self.decision_attention = min(self.decision_capacity, 
                                        self.decision_attention + self.recovery_rate)
            
            # 低焦虑时注意力分配更加平衡，但保持在0.3-0.7的范围内
            # 缓慢向中性值0.5靠拢，模拟注意力偏好的渐进调整
            self.food_attention_weight = max(0.3, min(0.7, self.food_attention_weight - 0.05))
            self.danger_attention_weight = 1.0 - self.food_attention_weight  # 确保两类权重之和为1
        
        # 检测注意力显著转移(注意力权重变化超过0.1)
        if abs(prev_food_attention - self.food_attention_weight) > 0.1:
            self.stats["attention_shifts"] += 1
        
        # 记录注意力历史，用于后续分析和可视化
        self.attention_history.append({
            "perception": self.perception_attention,       # 记录当前感知注意力
            "decision": self.decision_attention,           # 记录当前决策注意力
            "food_weight": self.food_attention_weight,     # 记录食物注意力权重
            "danger_weight": self.danger_attention_weight, # 记录危险注意力权重
            "anxiety": anxiety                             # 记录当前焦虑水平
        })
        
        # 返回当前注意力状态
        return {
            "perception_attention": self.perception_attention,     # 当前感知注意力
            "decision_attention": self.decision_attention,         # 当前决策注意力
            "food_attention_weight": self.food_attention_weight,   # 当前食物注意力权重
            "danger_attention_weight": self.danger_attention_weight # 当前危险注意力权重
        }
    
    def filter_perception(self, observation: np.ndarray, agent_position: Tuple[int, int]) -> np.ndarray:
        """
        基于当前注意力状态过滤感知信息。
        实现八阶段循环中的"数据→信息"阶段：通过注意力机制，将原始感知数据筛选为有用信息。
        注意力有限时，智能体只能感知部分环境信息，尤其是周围和注意力焦点区域的信息。

        Args:
            observation (np.ndarray): 原始观察数组，表示环境中的完整信息。
                - 通常是一个二维数组，表示网格环境
                - 不同的值代表不同类型的对象(0:空白, 1:食物, 2:障碍物等)
            agent_position (Tuple[int, int]): 智能体在环境中的位置坐标 (x, y)。
                - 这是注意力的中心点
                - 距离此点越远的信息被感知到的可能性越低

        Returns:
            np.ndarray: 经过注意力过滤的观察数组，与输入数组形状相同，但部分信息可能被过滤掉(置为0)。
        """
        # 如果注意力充足(超过90%容量)，返回完整观察，不进行过滤
        # 这表示智能体在注意力充沛时能够全面感知环境
        if self.perception_attention >= self.perception_capacity * 0.9:
            return observation.copy()
        
        # 创建过滤后的观察数组，初始全为0（表示未感知到任何信息）
        filtered_obs = np.zeros_like(observation)
        
        # 计算可感知的范围半径 (基于当前感知注意力)
        # 注意力越低，感知范围越小，至少为1(只能感知紧邻位置)
        perception_range = max(1, int(self.perception_attention))
        
        # 获取智能体坐标 - 注意力的中心点
        agent_x, agent_y = agent_position
        
        # 确定感知范围的边界，确保不超出环境边界
        min_x = max(0, agent_x - perception_range)  # 左边界
        max_x = min(observation.shape[0] - 1, agent_x + perception_range)  # 右边界
        min_y = max(0, agent_y - perception_range)  # 上边界
        max_y = min(observation.shape[1] - 1, agent_y + perception_range)  # 下边界
        
        # 复制可感知范围内的信息到过滤后的观察
        # 这表示智能体能够感知到周围一定范围内的所有信息
        filtered_obs[min_x:max_x+1, min_y:max_y+1] = observation[min_x:max_x+1, min_y:max_y+1]
        
        # 在可感知范围内，基于注意力权重和距离对特定类型的信息进行二次过滤
        # 这模拟了注意力选择性：即使在感知范围内，不同类型的对象被注意到的概率也不同
        for x in range(min_x, max_x+1):
            for y in range(min_y, max_y+1):
                # 计算与智能体的欧几里得距离
                distance = np.sqrt((x - agent_x)**2 + (y - agent_y)**2)
                
                # 如果距离超过感知范围的80%，应用注意力偏好过滤
                # 越远的物体越容易被忽略，特别是不符合当前注意力偏好的对象
                if distance > perception_range * 0.8:
                    # 假设观察中的1表示食物，2表示障碍物（危险）
                    if observation[x, y] == 1:  # 食物
                        # 根据食物注意力权重决定是否感知
                        # 权重越高，被忽略的概率越低
                        if np.random.random() > self.food_attention_weight:
                            filtered_obs[x, y] = 0  # 不感知此食物
                    elif observation[x, y] == 2:  # 障碍物/危险
                        # 根据危险注意力权重决定是否感知
                        # 权重越高，被忽略的概率越低
                        if np.random.random() > self.danger_attention_weight:
                            filtered_obs[x, y] = 0  # 不感知此障碍物
        
        return filtered_obs
    
    def filter_action_values(self, action_values: np.ndarray, anxiety: float) -> np.ndarray:
        """
        基于当前决策注意力过滤动作价值。
        实现"信息→决策"阶段中的注意力影响：决策注意力有限时，智能体可能无法准确评估所有动作。
        
        在高焦虑状态下，决策过程会受到更大干扰，可能导致非最优决策。

        Args:
            action_values (np.ndarray): 原始动作价值数组，表示每个可能动作的估计价值。
                - 通常是一维数组，每个元素对应一个动作的价值
                - 例如：[上、下、左、右]对应的价值分数
            anxiety (float): 当前焦虑水平(0-1)，影响注意力过滤的强度。
                - 值越高，决策过程受到的干扰越大

        Returns:
            np.ndarray: 经过注意力过滤的动作价值数组，可能包含噪声。
                - 同样是一维数组，但价值可能被扭曲
                - 决策注意力越低，扭曲越严重
        """
        # 如果决策注意力充足(超过90%容量)，返回原始动作价值，不添加噪声
        # 这表示智能体在注意力充沛时能够准确评估所有可能的行动
        if self.decision_attention >= self.decision_capacity * 0.9:
            return action_values.copy()
        
        # 创建过滤后的动作价值数组（初始为原始值的副本）
        filtered_values = action_values.copy()
        
        # 计算注意力不足导致的噪声水平
        # 注意力比例：当前决策注意力与最大容量的比值
        attention_ratio = self.decision_attention / self.decision_capacity
        # 噪声水平与注意力成反比：注意力越低，噪声越大，最大噪声为0.5
        noise_level = (1.0 - attention_ratio) * 0.5
        
        # 在高焦虑状态下，噪声会进一步增加
        # 这模拟了焦虑状态下决策能力的额外下降
        if anxiety > 0.5:  # 中高焦虑阈值
            noise_level *= (1.0 + anxiety)  # 焦虑越高，噪声增幅越大
        
        # 生成随机噪声并添加到动作价值中
        # 使用正态分布生成噪声，均值为0，标准差为noise_level
        noise = np.random.normal(0, noise_level, size=action_values.shape)
        filtered_values += noise
        
        return filtered_values
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        获取注意力模块的指标数据，用于分析和可视化。
        收集当前状态和历史统计信息，便于理解注意力动态变化。

        Returns:
            Dict[str, Any]: 包含各种注意力指标的字典，包括：
                - 当前注意力状态（感知、决策、偏好权重）
                - 平均注意力分配（历史平均值）
                - 完整的注意力历史记录
                - 统计指标（注意力转移次数、受限次数等）
        """
        # 计算历史平均注意力分配
        # 如果有历史记录，计算平均值；否则使用当前值
        if self.attention_history:
            # 计算感知注意力的历史平均值
            avg_perception = np.mean([rec["perception"] for rec in self.attention_history])
            # 计算决策注意力的历史平均值
            avg_decision = np.mean([rec["decision"] for rec in self.attention_history])
            # 计算食物注意力权重的历史平均值
            avg_food_weight = np.mean([rec["food_weight"] for rec in self.attention_history])
            # 计算危险注意力权重的历史平均值
            avg_danger_weight = np.mean([rec["danger_weight"] for rec in self.attention_history])
        else:
            # 无历史记录时使用当前值
            avg_perception = self.perception_capacity
            avg_decision = self.decision_capacity
            avg_food_weight = 0.5
            avg_danger_weight = 0.5
        
        # 返回完整的指标字典
        return {
            # 当前注意力状态
            "current_perception": self.perception_attention,  # 当前感知注意力
            "current_decision": self.decision_attention,      # 当前决策注意力
            "current_food_weight": self.food_attention_weight,  # 当前食物注意力权重
            "current_danger_weight": self.danger_attention_weight,  # 当前危险注意力权重
            
            # 历史平均值
            "avg_perception": avg_perception,     # 平均感知注意力
            "avg_decision": avg_decision,         # 平均决策注意力
            "avg_food_weight": avg_food_weight,   # 平均食物注意力权重
            "avg_danger_weight": avg_danger_weight,  # 平均危险注意力权重
            
            # 完整历史记录和统计数据
            "attention_history": self.attention_history.copy(),  # 注意力历史记录
            "stats": self.stats.copy()  # 统计指标
        }
    
    def get_attention_features(self) -> np.ndarray:
        """
        获取注意力状态特征向量，用于智能体观察。
        将当前注意力状态转换为特征向量，可以作为智能体状态的一部分。

        Returns:
            np.ndarray: 注意力状态特征向量，包含4个元素：
                - 第1个元素：归一化的感知注意力(0-1)
                - 第2个元素：归一化的决策注意力(0-1)
                - 第3个元素：食物注意力权重(0-1)
                - 第4个元素：危险注意力权重(0-1)
        """
        # 规范化注意力值，将感知和决策注意力转换到0-1范围
        # 这样便于与其他特征整合，保持一致的数值范围
        norm_perception = self.perception_attention / self.perception_capacity  # 归一化感知注意力
        norm_decision = self.decision_attention / self.decision_capacity        # 归一化决策注意力
        
        # 返回包含4个元素的特征向量
        return np.array([
            norm_perception,             # 当前感知注意力（归一化）
            norm_decision,               # 当前决策注意力（归一化）
            self.food_attention_weight,  # 食物注意力权重
            self.danger_attention_weight # 危险注意力权重
        ], dtype=np.float32)

# 测试代码
if __name__ == '__main__':
    # 创建注意力模块实例
    attention_module = AttentionModule(perception_capacity=10, decision_capacity=5)
    print(f"初始感知注意力: {attention_module.perception_attention}, 决策注意力: {attention_module.decision_attention}")
    
    # 测试不同焦虑水平对注意力的影响
    anxiety_levels = [0.1, 0.3, 0.5, 0.7, 0.9]  # 从低焦虑到高焦虑
    for anxiety in anxiety_levels:
        # 更新注意力状态
        metrics = attention_module.update(anxiety, 0.0)
        # 打印当前焦虑水平下的注意力状态
        print(f"\n焦虑水平 {anxiety:.1f} 的注意力状态:")
        print(f"  感知注意力: {metrics['perception_attention']:.2f}/{attention_module.perception_capacity}")
        print(f"  决策注意力: {metrics['decision_attention']:.2f}/{attention_module.decision_capacity}")
        print(f"  食物注意力权重: {metrics['food_attention_weight']:.2f}")
        print(f"  危险注意力权重: {metrics['danger_attention_weight']:.2f}")
    
    # 测试感知过滤功能
    print("\n测试感知过滤:")
    # 创建一个简单的10x10网格，包含智能体(3)、食物(1)和障碍物(2)
    grid = np.zeros((10, 10))
    grid[2, 3] = 1  # 食物
    grid[4, 5] = 1  # 食物
    grid[7, 8] = 1  # 食物
    grid[3, 7] = 2  # 障碍物
    grid[6, 2] = 2  # 障碍物
    agent_pos = (5, 5)  # 智能体位置
    
    # 测试低焦虑状态下的感知过滤
    attention_module.reset()  # 重置模块
    attention_module.update(0.1, 0.0)  # 设置低焦虑
    filtered_low_anxiety = attention_module.filter_perception(grid, agent_pos)
    
    # 测试高焦虑状态下的感知过滤
    attention_module.reset()  # 重置模块
    attention_module.update(0.8, 0.0)  # 设置高焦虑
    filtered_high_anxiety = attention_module.filter_perception(grid, agent_pos)
    
    # 打印结果进行比较
    print("原始网格:")
    print(grid)
    print("\n低焦虑过滤后:")
    print(filtered_low_anxiety)
    print("\n高焦虑过滤后:")
    print(filtered_high_anxiety) 