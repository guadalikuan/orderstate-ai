import numpy as np
from typing import Dict, Any, Tuple, List, Optional

class SignalModule:
    """
    信号处理模块
    
    负责处理从环境或智能体自身接收到的原始信号
    是序态循环中"能量→信号"阶段的实现
    """
    
    def __init__(self, signal_noise: float = 0.1, random_seed: Optional[int] = None):
        """
        初始化信号处理模块
        
        Args:
            signal_noise: 信号噪声水平，数值越大噪声越大
            random_seed: 随机种子
        """
        self.signal_noise = signal_noise
        
        # 设置随机种子
        if random_seed is not None:
            np.random.seed(random_seed)
            
        # 信号历史
        self.signal_history = []
        
        # 当前信号
        self.current_signal = None
        
        # 统计信息
        self.stats = {
            "total_signals": 0,
            "noise_levels": []
        }
    
    def reset(self):
        """
        重置信号处理模块
        """
        self.signal_history = []
        self.current_signal = None
        self.stats = {
            "total_signals": 0,
            "noise_levels": []
        }
    
    def process_environment_signal(self, env_state: Tuple[int, int], 
                                  target_pos: Tuple[int, int], 
                                  energy_level: float) -> Dict[str, Any]:
        """
        处理来自环境的信号
        
        Args:
            env_state: 环境状态，通常是智能体的位置坐标 (row, col)
            target_pos: 目标位置坐标 (row, col)
            energy_level: 当前能量水平
            
        Returns:
            Dict: 处理后的信号信息
        """
        # 添加噪声（根据能量水平调整噪声大小）
        # 能量越低，噪声越大
        energy_factor = max(0.1, min(1.0, energy_level / 100.0))
        actual_noise = self.signal_noise / energy_factor
        
        # 应用噪声到位置感知
        noisy_position = list(env_state)
        for i in range(len(noisy_position)):
            noise = np.random.normal(0, actual_noise)
            # 位置是整数，所以四舍五入
            noisy_position[i] = round(noisy_position[i] + noise)
        
        # 计算到目标的曼哈顿距离
        manhattan_distance = abs(env_state[0] - target_pos[0]) + abs(env_state[1] - target_pos[1])
        
        # 生成信号数据
        signal_data = {
            "position": tuple(env_state),  # 原始位置
            "perceived_position": tuple(noisy_position),  # 感知位置（有噪声）
            "target": target_pos,
            "manhattan_distance": manhattan_distance,
            "energy_level": energy_level,
            "noise_level": actual_noise,
            "timestamp": np.datetime64('now')
        }
        
        # 更新统计信息
        self.stats["total_signals"] += 1
        self.stats["noise_levels"].append(actual_noise)
        
        # 更新当前信号和历史
        self.current_signal = signal_data
        self.signal_history.append(signal_data)
        
        return signal_data
    
    def get_current_signal(self) -> Dict[str, Any]:
        """
        获取当前信号
        
        Returns:
            Dict: 当前信号数据
        """
        return self.current_signal if self.current_signal else {}
    
    def get_signal_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取信号历史
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict]: 信号历史记录
        """
        return self.signal_history[-limit:] if limit > 0 else self.signal_history.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        if self.stats["noise_levels"]:
            stats["avg_noise"] = np.mean(self.stats["noise_levels"])
            stats["max_noise"] = np.max(self.stats["noise_levels"])
            stats["min_noise"] = np.min(self.stats["noise_levels"])
        return stats 