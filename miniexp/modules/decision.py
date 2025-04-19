import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class DecisionModule:
    """
    决策模块
    
    负责将智慧转化为具体的决策选择
    是序态循环中"智慧→决策"阶段的实现
    """
    
    def __init__(self, action_space_size: int = 4, random_seed: Optional[int] = None):
        """
        初始化决策模块
        
        Args:
            action_space_size: 动作空间大小
            random_seed: 随机种子
        """
        self.action_space_size = action_space_size
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 动作映射
        self.action_mapping = {
            0: "上",
            1: "下",
            2: "左",
            3: "右"
        }
        
        # 决策历史
        self.decision_history = []
        
        # 当前决策
        self.current_decision = None
        
        # 统计信息
        self.stats = {
            "decisions_made": 0,
            "action_counts": {i: 0 for i in range(action_space_size)}
        }
    
    def reset(self):
        """
        重置决策模块
        """
        self.decision_history = []
        self.current_decision = None
        self.stats = {
            "decisions_made": 0,
            "action_counts": {i: 0 for i in range(self.action_space_size)}
        }
    
    def make_decision(self, wisdom_data: Dict[str, Any], 
                     override_action: Optional[int] = None) -> int:
        """
        根据智慧数据做出决策
        
        Args:
            wisdom_data: 智慧数据
            override_action: 可选的覆盖动作，用于强制选择特定动作
            
        Returns:
            int: 决策的动作索引
        """
        if override_action is not None and 0 <= override_action < self.action_space_size:
            # 使用覆盖动作
            action = override_action
        else:
            # 从智慧数据中获取最佳动作
            action = wisdom_data.get('best_action_index', 0)
            
            # 确保动作在有效范围内
            if not (0 <= action < self.action_space_size):
                action = np.random.randint(0, self.action_space_size)
        
        # 决策信息
        decision_info = {
            'action': action,
            'action_name': self.action_mapping.get(action, "未知"),
            'raw_wisdom': wisdom_data,
            'override': override_action is not None,
            'timestamp': np.datetime64('now')
        }
        
        # 更新统计信息
        self.stats["decisions_made"] += 1
        self.stats["action_counts"][action] = self.stats["action_counts"].get(action, 0) + 1
        
        # 更新当前决策和历史
        self.current_decision = decision_info
        self.decision_history.append(decision_info)
        
        return action
    
    def get_current_decision(self) -> Dict[str, Any]:
        """
        获取当前决策
        
        Returns:
            Dict: 当前决策信息
        """
        return self.current_decision if self.current_decision else {}
    
    def get_decision_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取决策历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 决策历史记录
        """
        return self.decision_history[-limit:] if limit > 0 else self.decision_history.copy()
    
    def get_action_distribution(self) -> Dict[str, Any]:
        """
        获取动作分布统计
        
        Returns:
            Dict: 动作分布统计
        """
        if self.stats["decisions_made"] == 0:
            return {"message": "暂无决策数据"}
            
        action_counts = self.stats["action_counts"]
        total_counts = self.stats["decisions_made"]
        
        distribution = {}
        for action, count in action_counts.items():
            distribution[self.action_mapping.get(action, f"动作{action}")] = {
                "count": count,
                "percentage": count / total_counts * 100 if total_counts > 0 else 0
            }
            
        return {
            "total_decisions": total_counts,
            "distribution": distribution
        }
    
    def format_decision_reason(self, wisdom_data: Dict[str, Any]) -> str:
        """
        根据智慧数据格式化决策理由
        
        Args:
            wisdom_data: 智慧数据
            
        Returns:
            str: 决策理由说明
        """
        # 从智慧数据中提取信息
        strategy = wisdom_data.get('strategy', 'balanced')
        confidence = wisdom_data.get('confidence', 0.0)
        anxiety = wisdom_data.get('anxiety', 0.0)
        energy_level = wisdom_data.get('energy_level', 0.0)
        
        # 动作价值
        raw_values = wisdom_data.get('raw_action_values', [])
        constrained_values = wisdom_data.get('constrained_values', [])
        
        # 最佳动作
        best_action_idx = wisdom_data.get('best_action_index', 0)
        action_name = self.action_mapping.get(best_action_idx, "未知")
        
        # 构建决策理由
        reason = f"选择动作: {action_name} (索引: {best_action_idx})\n"
        reason += f"策略: {strategy}, 置信度: {confidence:.2f}\n"
        reason += f"焦虑水平: {anxiety:.2f}, 能量水平: {energy_level:.2f}\n"
        
        # 添加动作价值信息
        if raw_values and constrained_values:
            reason += "动作价值 (原始 → 调整):\n"
            for i in range(min(len(raw_values), len(constrained_values))):
                action_label = self.action_mapping.get(i, f"动作{i}")
                reason += f"  {action_label}: {raw_values[i]:.2f} → {constrained_values[i]:.2f}"
                if i == best_action_idx:
                    reason += " [选择]"
                reason += "\n"
        
        return reason 