import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class KnowledgeModule:
    """
    知识处理模块
    
    负责整合信息并提取可用的知识，包括对环境的理解和可能的行动方案
    是序态循环中"信息→知识"阶段的实现
    """
    
    def __init__(self, action_space_size: int = 4, learning_rate: float = 0.1, 
                discount_factor: float = 0.9, random_seed: Optional[int] = None):
        """
        初始化知识处理模块
        
        Args:
            action_space_size: 动作空间大小
            learning_rate: 学习率，用于更新知识
            discount_factor: 折扣因子，用于价值评估
            random_seed: 随机种子
        """
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 动作到方向的映射
        self.action_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # 上、下、左、右
        
        # 知识库：简单的Q值表，实际中可以使用更复杂的表示
        self.action_values = np.zeros(action_space_size)
        
        # 环境知识
        self.environment_model = {
            'grid_size': (0, 0),
            'target_position': (0, 0),
            'visited_positions': set(),
            'reward_history': []
        }
        
        # 知识历史
        self.knowledge_history = []
        
        # 当前知识
        self.current_knowledge = None
        
        # 累计统计
        self.stats = {
            'updates': 0,
            'exploration_rate': 1.0,
            'action_selection_history': []
        }
    
    def reset(self):
        """
        重置知识处理模块
        """
        self.action_values = np.zeros(self.action_space_size)
        self.environment_model = {
            'grid_size': (0, 0),
            'target_position': (0, 0),
            'visited_positions': set(),
            'reward_history': []
        }
        self.knowledge_history = []
        self.current_knowledge = None
        self.stats = {
            'updates': 0,
            'exploration_rate': 1.0,
            'action_selection_history': []
        }
    
    def update_environment_model(self, position: Tuple[int, int], 
                               target_position: Tuple[int, int],
                               grid_size: Tuple[int, int],
                               reward: float = 0.0):
        """
        更新环境模型
        
        Args:
            position: 当前位置
            target_position: 目标位置
            grid_size: 网格大小
            reward: 获得的奖励
        """
        self.environment_model['grid_size'] = grid_size
        self.environment_model['target_position'] = target_position
        self.environment_model['visited_positions'].add(position)
        self.environment_model['reward_history'].append(reward)
    
    def compute_action_values(self, weighted_features: np.ndarray, 
                             position: Tuple[int, int]) -> np.ndarray:
        """
        计算动作价值
        
        Args:
            weighted_features: 加权特征（信息）
            position: 当前位置
            
        Returns:
            np.ndarray: 动作价值数组
        """
        # 如果没有目标位置信息，使用默认值
        if not self.environment_model['target_position']:
            return np.zeros(self.action_space_size)
            
        target_pos = self.environment_model['target_position']
        grid_size = self.environment_model['grid_size']
        
        # 计算每个动作的期望价值
        action_values = np.zeros(self.action_space_size)
        
        for action in range(self.action_space_size):
            # 预测执行此动作后的位置
            dr, dc = self.action_directions[action]
            next_r, next_c = position[0] + dr, position[1] + dc
            
            # 检查边界
            if 0 <= next_r < grid_size[0] and 0 <= next_c < grid_size[1]:
                # 计算到目标的曼哈顿距离
                current_distance = abs(position[0] - target_pos[0]) + abs(position[1] - target_pos[1])
                next_distance = abs(next_r - target_pos[0]) + abs(next_c - target_pos[1])
                
                # 距离减小，给予正价值
                if next_distance < current_distance:
                    action_values[action] += 1.0
                # 距离增加，给予负价值
                elif next_distance > current_distance:
                    action_values[action] -= 0.5
                    
                # 如果下一位置是目标，额外奖励
                if (next_r, next_c) == target_pos:
                    action_values[action] += 2.0
                    
                # 避免重复访问（鼓励探索）
                if (next_r, next_c) in self.environment_model['visited_positions']:
                    action_values[action] -= 0.2
            else:
                # 出界，强烈负价值
                action_values[action] -= 1.0
                
        # 平滑更新动作价值
        self.action_values = (1 - self.learning_rate) * self.action_values + self.learning_rate * action_values
        
        return self.action_values
    
    def process_information(self, information_data: Dict[str, Any], 
                          current_position: Tuple[int, int],
                          target_position: Tuple[int, int],
                          grid_size: Tuple[int, int]) -> Dict[str, Any]:
        """
        处理信息，生成知识
        
        Args:
            information_data: 信息数据
            current_position: 当前位置
            target_position: 目标位置
            grid_size: 网格大小
            
        Returns:
            Dict: 知识数据
        """
        # 更新环境模型
        self.update_environment_model(
            position=current_position,
            target_position=target_position,
            grid_size=grid_size
        )
        
        # 从信息中提取加权特征
        weighted_features = information_data.get('weighted_features', np.zeros(4))
        
        # 计算动作价值
        action_values = self.compute_action_values(weighted_features, current_position)
        
        # 动作空间
        possible_actions = list(range(self.action_space_size))
        
        # 目标方向（简单启发式）
        target_direction = None
        if current_position and target_position:
            dr = target_position[0] - current_position[0]
            dc = target_position[1] - current_position[1]
            
            if abs(dr) > abs(dc):
                target_direction = 0 if dr < 0 else 1  # 上或下
            else:
                target_direction = 2 if dc < 0 else 3  # 左或右
        
        # 生成知识数据
        knowledge_data = {
            'action_values': action_values.tolist(),
            'possible_actions': possible_actions,
            'target_direction': target_direction,
            'current_position': current_position,
            'target_position': target_position,
            'visited_count': len(self.environment_model['visited_positions']),
            'exploration_rate': self.stats['exploration_rate'],
            'timestamp': np.datetime64('now')
        }
        
        # 更新统计
        self.stats['updates'] += 1
        
        # 降低探索率
        self.stats['exploration_rate'] = max(
            0.1, 
            self.stats['exploration_rate'] * 0.995
        )
        
        # 更新当前知识和历史
        self.current_knowledge = knowledge_data
        self.knowledge_history.append(knowledge_data)
        
        return knowledge_data
    
    def get_current_knowledge(self) -> Dict[str, Any]:
        """
        获取当前知识
        
        Returns:
            Dict: 当前知识数据
        """
        return self.current_knowledge if self.current_knowledge else {}
    
    def get_knowledge_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取知识历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 知识历史记录
        """
        return self.knowledge_history[-limit:] if limit > 0 else self.knowledge_history.copy()
    
    def get_environment_model(self) -> Dict[str, Any]:
        """
        获取环境模型
        
        Returns:
            Dict: 环境模型
        """
        model = self.environment_model.copy()
        model['visited_positions'] = list(model['visited_positions'])  # 转换set为list以便序列化
        return model 