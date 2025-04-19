from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from miniexp.visualization.base import BaseVisualizer

class MetricsVisualizer(BaseVisualizer):
    """实验指标可视化器"""
    
    def __init__(self):
        """初始化实验指标可视化器"""
        super().__init__()
        self.reward_line = None
        self.success_rate_line = None
        self.survival_time_line = None
        self.episodes = []
        self.reward_history = []
        self.success_rate_history = []
        self.survival_time_history = []
        
    def render(self, metrics: Dict[str, Any]) -> None:
        """
        渲染实验指标
        
        Args:
            metrics: 实验指标
        """
        self.initialize()
        self.clear()
        
        # 设置子图
        self.ax.set_title('实验性能指标')
        self.ax.set_xlabel('回合')
        self.ax.set_ylabel('值')
        
        # 初始化数据
        self.episodes = [0]
        self.reward_history = [metrics.get('avg_reward', 0)]
        self.success_rate_history = [metrics.get('success_rate', 0)]
        self.survival_time_history = [metrics.get('avg_survival_time', 0)]
        
        # 绘制初始状态
        self.reward_line, = self.ax.plot(self.episodes, self.reward_history,
                                       label='平均奖励', color='blue')
        self.success_rate_line, = self.ax.plot(self.episodes, self.success_rate_history,
                                             label='成功率', color='green')
        self.survival_time_line, = self.ax.plot(self.episodes, self.survival_time_history,
                                              label='平均生存时间', color='red')
        
        # 添加图例
        self.ax.legend()
        
    def update(self, metrics: Dict[str, Any]) -> None:
        """
        更新实验指标
        
        Args:
            metrics: 实验指标
        """
        if not self.initialized:
            self.render(metrics)
            return
            
        # 更新回合数
        self.episodes.append(len(self.episodes))
        
        # 更新数据
        self.reward_history.append(metrics.get('avg_reward', 0))
        self.success_rate_history.append(metrics.get('success_rate', 0))
        self.survival_time_history.append(metrics.get('avg_survival_time', 0))
        
        # 更新线条
        self.reward_line.set_data(self.episodes, self.reward_history)
        self.success_rate_line.set_data(self.episodes, self.success_rate_history)
        self.survival_time_line.set_data(self.episodes, self.survival_time_history)
        
        # 调整坐标轴范围
        self.ax.relim()
        self.ax.autoscale_view()
        
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        获取历史指标数据
        
        Returns:
            Dict[str, List[float]]: 历史指标数据
        """
        return {
            'episodes': self.episodes,
            'reward': self.reward_history,
            'success_rate': self.success_rate_history,
            'survival_time': self.survival_time_history
        } 