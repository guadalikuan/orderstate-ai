import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class InformationModule:
    """
    信息处理模块
    
    负责使用注意力机制过滤和处理数据，从数据中提取有价值的信息
    是序态循环中"数据→信息"阶段的实现
    """
    
    def __init__(self, feature_dim: int = 4, 
                 attention_capacity: float = 0.7,
                 anxiety_influence: float = 1.2,
                 random_seed: Optional[int] = None):
        """
        初始化信息处理模块
        
        Args:
            feature_dim: 特征向量维度
            attention_capacity: 注意力容量 (0-1)，表示能够关注的特征比例
            anxiety_influence: 焦虑影响系数，决定焦虑对注意力分配的影响程度
            random_seed: 随机种子
        """
        self.feature_dim = feature_dim
        self.attention_capacity = attention_capacity
        self.anxiety_influence = anxiety_influence
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 注意力权重（初始均匀分布）
        self.attention_weights = np.ones(feature_dim) / feature_dim
        
        # 信息历史
        self.information_history = []
        
        # 当前信息
        self.current_information = None
        
        # 统计信息
        self.stats = {
            "processed_data_count": 0,
            "attention_entropy_history": [],  # 注意力分布的熵
            "feature_importance": np.zeros(feature_dim)  # 特征重要性累积
        }
    
    def reset(self):
        """
        重置信息处理模块
        """
        self.attention_weights = np.ones(self.feature_dim) / self.feature_dim
        self.information_history = []
        self.current_information = None
        self.stats = {
            "processed_data_count": 0,
            "attention_entropy_history": [],
            "feature_importance": np.zeros(self.feature_dim)
        }
    
    def compute_attention(self, features: np.ndarray, anxiety: float) -> np.ndarray:
        """
        计算注意力权重
        
        Args:
            features: 特征向量
            anxiety: 焦虑度 (0-1)
            
        Returns:
            np.ndarray: 注意力权重向量
        """
        # 特征显著性（简单地使用特征的绝对值）
        feature_salience = np.abs(features)
        
        # 应用焦虑影响
        # 焦虑度越高，注意力越集中（分布越尖锐）
        # 焦虑度越低，注意力越分散（分布越平坦）
        if anxiety > 0:
            # 焦虑时，强化最显著特征，抑制其他特征
            sharpness = 1.0 + (anxiety * self.anxiety_influence)
            feature_salience = np.power(feature_salience, sharpness)
        
        # 归一化得到注意力权重
        if np.sum(feature_salience) > 0:
            attention = feature_salience / np.sum(feature_salience)
        else:
            attention = np.ones_like(feature_salience) / len(feature_salience)
        
        # 应用注意力容量限制
        # 仅保留最显著的特征，总权重为attention_capacity
        sorted_indices = np.argsort(attention)[::-1]  # 从大到小排序
        
        limited_attention = np.zeros_like(attention)
        
        # 按重要性依次分配注意力，直到达到容量上限
        remaining_capacity = self.attention_capacity
        for idx in sorted_indices:
            if remaining_capacity <= 0:
                break
                
            weight = min(attention[idx], remaining_capacity)
            limited_attention[idx] = weight
            remaining_capacity -= weight
        
        # 再次归一化
        if np.sum(limited_attention) > 0:
            limited_attention /= np.sum(limited_attention)
        
        # 更新注意力权重（平滑更新）
        self.attention_weights = 0.7 * self.attention_weights + 0.3 * limited_attention
        
        return self.attention_weights
    
    def apply_attention(self, features: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        """
        应用注意力权重到特征向量
        
        Args:
            features: 特征向量
            attention_weights: 注意力权重
            
        Returns:
            np.ndarray: 加权后的特征向量（信息）
        """
        # 加权特征
        weighted_features = features * attention_weights
        return weighted_features
    
    def process_data(self, features: np.ndarray, anxiety: float, 
                    raw_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        处理数据，提取信息
        
        Args:
            features: 特征向量
            anxiety: 焦虑度 (0-1)
            raw_data: 原始数据
            
        Returns:
            Dict: 提取的信息
        """
        # 计算注意力权重
        attention_weights = self.compute_attention(features, anxiety)
        
        # 应用注意力获取加权特征（信息）
        weighted_features = self.apply_attention(features, attention_weights)
        
        # 计算注意力熵（衡量注意力分散程度）
        non_zero_weights = attention_weights[attention_weights > 0]
        if len(non_zero_weights) > 0:
            entropy = -np.sum(non_zero_weights * np.log2(non_zero_weights))
        else:
            entropy = 0.0
        
        # 生成信息数据
        information_data = {
            "raw_features": features,
            "weighted_features": weighted_features,
            "attention_weights": attention_weights,
            "anxiety": anxiety,
            "attention_entropy": entropy,
            "raw_data": raw_data,
            "timestamp": np.datetime64('now')
        }
        
        # 更新统计信息
        self.stats["processed_data_count"] += 1
        self.stats["attention_entropy_history"].append(entropy)
        self.stats["feature_importance"] += attention_weights
        
        # 更新当前信息和历史
        self.current_information = information_data
        self.information_history.append(information_data)
        
        return information_data
    
    def get_current_information(self) -> Dict[str, Any]:
        """
        获取当前信息
        
        Returns:
            Dict: 当前信息数据
        """
        return self.current_information if self.current_information else {}
    
    def get_information_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取信息历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 信息历史记录
        """
        return self.information_history[-limit:] if limit > 0 else self.information_history.copy()
    
    def get_feature_importance(self) -> Dict[str, Any]:
        """
        获取特征重要性
        
        Returns:
            Dict: 特征重要性信息
        """
        if self.stats["processed_data_count"] > 0:
            normalized_importance = self.stats["feature_importance"] / self.stats["processed_data_count"]
        else:
            normalized_importance = np.zeros_like(self.stats["feature_importance"])
            
        return {
            "importance": normalized_importance.tolist(),
            "attention_entropy_avg": np.mean(self.stats["attention_entropy_history"]) if self.stats["attention_entropy_history"] else 0.0
        } 