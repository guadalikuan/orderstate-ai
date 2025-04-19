import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class WisdomModule:
    """
    智慧处理模块
    
    负责根据知识形成决策原则，进行价值评估和行动策略规划
    是序态循环中"知识→智慧"阶段的实现
    """
    
    def __init__(self, action_space_size: int = 4, 
                 temperature: float = 1.0, 
                 energy_priority_factor: float = 0.5,
                 random_seed: Optional[int] = None):
        """
        初始化智慧处理模块
        
        Args:
            action_space_size: 动作空间大小
            temperature: 决策温度参数，控制探索和利用的平衡
            energy_priority_factor: 能量优先级因子，决定能量因素的重要性
            random_seed: 随机种子
        """
        self.action_space_size = action_space_size
        self.temperature = temperature
        self.energy_priority_factor = energy_priority_factor
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 行动策略
        self.strategy_name = "balanced"  # 默认策略
        
        # 决策历史
        self.wisdom_history = []
        
        # 当前决策智慧
        self.current_wisdom = None
        
        # 统计数据
        self.stats = {
            "decision_count": 0,
            "strategy_history": [],
            "confidence_history": []
        }
    
    def reset(self):
        """
        重置智慧处理模块
        """
        self.strategy_name = "balanced"
        self.wisdom_history = []
        self.current_wisdom = None
        self.stats = {
            "decision_count": 0,
            "strategy_history": [],
            "confidence_history": []
        }
    
    def determine_strategy(self, knowledge_data: Dict[str, Any], 
                          anxiety: float,
                          energy_level: float) -> str:
        """
        根据当前状态确定策略
        
        Args:
            knowledge_data: 知识数据
            anxiety: 焦虑水平 (0-1)
            energy_level: 能量水平
            
        Returns:
            str: 策略名称
        """
        # 根据焦虑程度和能量水平确定策略
        if anxiety > 0.7:
            # 高焦虑，采用保守策略
            strategy = "conservative"
        elif anxiety > 0.3:
            # 中等焦虑，采用平衡策略
            strategy = "balanced"
        else:
            # 低焦虑，根据能量水平决定
            if energy_level > 70:
                # 能量充足，可以探索
                strategy = "exploratory"
            else:
                # 能量适中，平衡策略
                strategy = "balanced"
        
        # 更新策略历史
        self.stats["strategy_history"].append(strategy)
        
        return strategy
    
    def apply_action_constraints(self, action_values: np.ndarray,
                               anxiety: float,
                               energy_level: float) -> np.ndarray:
        """
        应用行动约束，根据当前状态调整动作价值
        
        Args:
            action_values: 动作价值数组
            anxiety: 焦虑水平 (0-1)
            energy_level: 能量水平
            
        Returns:
            np.ndarray: 调整后的动作价值
        """
        constrained_values = action_values.copy()
        
        # 焦虑影响 - 高焦虑时更倾向于选择最高价值的动作
        if anxiety > 0.5:
            # 找出最大值索引
            max_idx = np.argmax(constrained_values)
            
            # 增强最大值，抑制其他值
            boost_factor = 1.0 + anxiety
            constrained_values[max_idx] *= boost_factor
        
        # 能量影响 - 低能量时更倾向于选择能量效率高的动作
        if energy_level < 50:
            # 简单假设：较小的索引对应较节能的动作
            energy_efficiency = np.array([1.0, 0.9, 0.8, 0.7])
            
            # 根据能量水平调整能量效率权重
            energy_weight = self.energy_priority_factor * (1.0 - energy_level / 100.0)
            
            # 应用能量效率调整
            constrained_values = constrained_values * (1.0 - energy_weight) + energy_efficiency * energy_weight
            
        return constrained_values
    
    def calculate_action_probabilities(self, action_values: np.ndarray) -> np.ndarray:
        """
        根据动作价值计算选择概率（Softmax）
        
        Args:
            action_values: 动作价值数组
            
        Returns:
            np.ndarray: 动作选择概率
        """
        # 避免数值溢出的Softmax实现
        exp_values = np.exp((action_values - np.max(action_values)) / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        return probabilities
    
    def find_best_action(self, action_values: np.ndarray, 
                        action_probabilities: np.ndarray) -> Tuple[int, float]:
        """
        根据策略确定最佳动作
        
        Args:
            action_values: 动作价值数组
            action_probabilities: 动作选择概率
            
        Returns:
            Tuple[int, float]: (最佳动作索引, 决策置信度)
        """
        # 根据当前策略确定选择方法
        if self.strategy_name == "conservative":
            # 保守策略 - 选择价值最高的动作
            best_action_idx = np.argmax(action_values)
            confidence = action_probabilities[best_action_idx]
        elif self.strategy_name == "exploratory":
            # 探索策略 - 按概率随机选择
            best_action_idx = np.random.choice(len(action_probabilities), p=action_probabilities)
            confidence = action_probabilities[best_action_idx]
        else:  # "balanced"
            # 平衡策略 - 有偏概率选择
            if np.random.random() < 0.7:  # 70%概率选择最优
                best_action_idx = np.argmax(action_values)
            else:  # 30%概率随机探索
                # 防止总是选到最大值，先把最大值排除
                temp_probs = action_probabilities.copy()
                max_idx = np.argmax(temp_probs)
                temp_probs[max_idx] = 0
                
                # 重新归一化并选择
                if np.sum(temp_probs) > 0:
                    temp_probs = temp_probs / np.sum(temp_probs)
                    best_action_idx = np.random.choice(len(temp_probs), p=temp_probs)
                else:
                    best_action_idx = np.random.choice(len(action_probabilities))
                    
            confidence = action_probabilities[best_action_idx]
        
        return best_action_idx, confidence
    
    def process_knowledge(self, knowledge_data: Dict[str, Any], 
                         anxiety: float,
                         energy_level: float) -> Dict[str, Any]:
        """
        处理知识，生成智慧
        
        Args:
            knowledge_data: 知识数据
            anxiety: 焦虑水平 (0-1)
            energy_level: 能量水平
            
        Returns:
            Dict: 智慧数据
        """
        # 获取动作价值和可能的动作
        action_values = np.array(knowledge_data.get('action_values', [0] * self.action_space_size))
        possible_actions = knowledge_data.get('possible_actions', list(range(self.action_space_size)))
        
        # 确定当前策略
        self.strategy_name = self.determine_strategy(knowledge_data, anxiety, energy_level)
        
        # 应用行动约束
        constrained_values = self.apply_action_constraints(action_values, anxiety, energy_level)
        
        # 计算动作概率
        action_probabilities = self.calculate_action_probabilities(constrained_values)
        
        # 根据策略选择最佳动作
        best_action_idx, confidence = self.find_best_action(constrained_values, action_probabilities)
        
        # 生成智慧数据
        wisdom_data = {
            'raw_action_values': action_values.tolist(),
            'constrained_values': constrained_values.tolist(),
            'action_probabilities': action_probabilities.tolist(),
            'best_action_index': int(best_action_idx),
            'strategy': self.strategy_name,
            'confidence': float(confidence),
            'anxiety': anxiety,
            'energy_level': energy_level,
            'timestamp': np.datetime64('now')
        }
        
        # 更新统计
        self.stats["decision_count"] += 1
        self.stats["confidence_history"].append(confidence)
        
        # 更新当前智慧和历史
        self.current_wisdom = wisdom_data
        self.wisdom_history.append(wisdom_data)
        
        return wisdom_data
    
    def get_current_wisdom(self) -> Dict[str, Any]:
        """
        获取当前智慧
        
        Returns:
            Dict: 当前智慧数据
        """
        return self.current_wisdom if self.current_wisdom else {}
    
    def get_wisdom_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取智慧历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 智慧历史记录
        """
        return self.wisdom_history[-limit:] if limit > 0 else self.wisdom_history.copy()
    
    def get_strategy_stats(self) -> Dict[str, Any]:
        """
        获取策略统计
        
        Returns:
            Dict: 策略使用统计
        """
        if not self.stats["strategy_history"]:
            return {"message": "暂无数据"}
            
        strategy_counts = {}
        for strategy in self.stats["strategy_history"]:
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            
        avg_confidence = np.mean(self.stats["confidence_history"]) if self.stats["confidence_history"] else 0.0
        
        return {
            "strategy_counts": strategy_counts,
            "total_decisions": self.stats["decision_count"],
            "avg_confidence": float(avg_confidence)
        } 