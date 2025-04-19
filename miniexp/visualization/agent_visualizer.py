from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from miniexp.visualization.base import BaseVisualizer

class AgentStateVisualizer(BaseVisualizer):
    """智能体状态可视化器"""
    
    def __init__(self):
        """初始化智能体状态可视化器"""
        super().__init__()
        self.energy_line = None
        self.attention_line = None
        self.perception_line = None
        self.decision_line = None
        self.time_steps = []
        self.energy_history = []
        self.attention_history = []
        self.perception_history = []
        self.decision_history = []
        
    def render(self, state: Dict[str, Any]) -> None:
        """
        渲染智能体状态
        
        Args:
            state: 智能体状态
        """
        self.initialize()
        self.clear()
        
        # 设置子图
        self.ax.set_title('智能体状态监控')
        self.ax.set_xlabel('时间步')
        self.ax.set_ylabel('值')
        
        # 初始化数据
        self.time_steps = [0]
        self.energy_history = [state.get('energy', 0)]
        self.attention_history = [state.get('attention_level', 0)]
        self.perception_history = [state.get('perception_level', 0)]
        self.decision_history = [state.get('decision_level', 0)]
        
        # 绘制初始状态
        self.energy_line, = self.ax.plot(self.time_steps, self.energy_history, 
                                       label='能量', color='blue')
        self.attention_line, = self.ax.plot(self.time_steps, self.attention_history,
                                          label='注意力', color='red')
        self.perception_line, = self.ax.plot(self.time_steps, self.perception_history,
                                           label='感知', color='green')
        self.decision_line, = self.ax.plot(self.time_steps, self.decision_history,
                                         label='决策', color='purple')
        
        # 添加图例
        self.ax.legend()
        
    def update(self, state: Dict[str, Any]) -> None:
        """
        更新智能体状态
        
        Args:
            state: 智能体状态
        """
        if not self.initialized:
            self.render(state)
            return
            
        # 更新时间步
        self.time_steps.append(len(self.time_steps))
        
        # 更新数据
        self.energy_history.append(state.get('energy', 0))
        self.attention_history.append(state.get('attention_level', 0))
        self.perception_history.append(state.get('perception_level', 0))
        self.decision_history.append(state.get('decision_level', 0))
        
        # 更新线条
        self.energy_line.set_data(self.time_steps, self.energy_history)
        self.attention_line.set_data(self.time_steps, self.attention_history)
        self.perception_line.set_data(self.time_steps, self.perception_history)
        self.decision_line.set_data(self.time_steps, self.decision_history)
        
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
            'time_steps': self.time_steps,
            'energy': self.energy_history,
            'attention': self.attention_history,
            'perception': self.perception_history,
            'decision': self.decision_history
        } 