import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union

class DataModule:
    """
    数据处理模块
    
    负责将原始信号转换为结构化特征数据
    是序态循环中"信号→数据"阶段的实现
    """
    
    def __init__(self, feature_dim: int = 4, random_seed: Optional[int] = None):
        """
        初始化数据处理模块
        
        Args:
            feature_dim: 特征向量维度
            random_seed: 随机种子，用于可重复性
        """
        self.feature_dim = feature_dim
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 数据历史
        self.data_history = []
        
        # 当前数据
        self.current_data = None
        
        # 统计信息
        self.stats = {
            "processed_signals": 0,
            "feature_means": np.zeros(feature_dim),
            "feature_vars": np.zeros(feature_dim)
        }
    
    def reset(self):
        """
        重置数据处理模块
        """
        self.data_history = []
        self.current_data = None
        self.stats = {
            "processed_signals": 0,
            "feature_means": np.zeros(self.feature_dim),
            "feature_vars": np.zeros(self.feature_dim)
        }
    
    def extract_features(self, signal_data: Dict[str, Any], 
                        grid_size: Tuple[int, int]) -> np.ndarray:
        """
        从信号中提取特征
        
        Args:
            signal_data: 信号数据
            grid_size: 网格大小 (height, width)
            
        Returns:
            np.ndarray: 特征向量
        """
        # 解析信号数据
        position = signal_data.get("position", (0, 0))
        target = signal_data.get("target", (0, 0))
        manhattan_distance = signal_data.get("manhattan_distance", 0)
        energy_level = signal_data.get("energy_level", 100.0)
        
        # 网格大小
        grid_height, grid_width = grid_size
        
        # 构建特征向量
        features = np.zeros(self.feature_dim, dtype=np.float32)
        
        # 位置特征（归一化）
        features[0] = position[0] / (grid_height - 1) if grid_height > 1 else 0
        features[1] = position[1] / (grid_width - 1) if grid_width > 1 else 0
        
        # 距离特征（归一化）
        max_distance = grid_height + grid_width - 2  # 最大曼哈顿距离
        features[2] = manhattan_distance / max_distance if max_distance > 0 else 0
        
        # 能量特征（归一化）
        features[3] = energy_level / 100.0
        
        # 处理后的数据
        processed_data = {
            "features": features,
            "raw_signal": signal_data,
            "timestamp": np.datetime64('now')
        }
        
        # 更新统计信息
        self.stats["processed_signals"] += 1
        
        # 更新特征统计（用于在线计算均值和方差）
        n = self.stats["processed_signals"]
        if n > 1:
            delta = features - self.stats["feature_means"]
            self.stats["feature_means"] += delta / n
            delta2 = features - self.stats["feature_means"]
            self.stats["feature_vars"] += delta * delta2
        else:
            self.stats["feature_means"] = features.copy()
        
        # 更新当前数据和历史
        self.current_data = processed_data
        self.data_history.append(processed_data)
        
        return features
    
    def get_current_data(self) -> Dict[str, Any]:
        """
        获取当前数据
        
        Returns:
            Dict: 当前处理后的数据
        """
        return self.current_data if self.current_data else {}
    
    def get_data_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取数据历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 数据历史记录
        """
        return self.data_history[-limit:] if limit > 0 else self.data_history.copy()
    
    def get_feature_statistics(self) -> Dict[str, Any]:
        """
        获取特征统计信息
        
        Returns:
            Dict: 特征统计信息，包括均值和方差
        """
        n = self.stats["processed_signals"]
        feature_stds = np.sqrt(self.stats["feature_vars"] / max(1, n - 1))
        
        return {
            "means": self.stats["feature_means"].tolist(),
            "stds": feature_stds.tolist(),
            "count": n
        }
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        特征标准化（Z-Score标准化）
        
        Args:
            features: 原始特征向量
            
        Returns:
            np.ndarray: 标准化后的特征向量
        """
        if self.stats["processed_signals"] < 2:
            return features  # 不足以计算标准差
        
        stds = np.sqrt(self.stats["feature_vars"] / (self.stats["processed_signals"] - 1))
        stds = np.where(stds == 0, 1, stds)  # 避免除零
        
        normalized = (features - self.stats["feature_means"]) / stds
        return normalized 