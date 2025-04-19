from typing import Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from miniexp.visualization.base import BaseVisualizer

class EnvironmentVisualizer(BaseVisualizer):
    """环境可视化器"""
    
    def __init__(self, width: int, height: int):
        """
        初始化环境可视化器
        
        Args:
            width: 环境宽度
            height: 环境高度
        """
        super().__init__()
        self.width = width
        self.height = height
        self.grid = None
        self.agent_patch = None
        self.target_patch = None
        self.obstacle_patches = []
        self.predator_patches = []
        self.moving_obstacle_patches = []
        
    def render(self, state: Dict[str, Any]) -> None:
        """
        渲染环境状态
        
        Args:
            state: 环境状态
        """
        self.initialize()
        self.clear()
        
        # 设置坐标轴
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_xticks(np.arange(0, self.width + 1, 1))
        self.ax.set_yticks(np.arange(0, self.height + 1, 1))
        self.ax.grid(True)
        
        # 绘制网格
        self.grid = np.zeros((self.height, self.width))
        
        # 绘制障碍物
        self.obstacle_patches = []
        for obstacle in state.get('obstacles', []):
            rect = Rectangle((obstacle[0], obstacle[1]), 1, 1, 
                           facecolor='gray', edgecolor='black')
            self.ax.add_patch(rect)
            self.obstacle_patches.append(rect)
            
        # 绘制移动障碍物
        self.moving_obstacle_patches = []
        for obstacle in state.get('moving_obstacles', []):
            rect = Rectangle((obstacle[0], obstacle[1]), 1, 1,
                           facecolor='orange', edgecolor='black')
            self.ax.add_patch(rect)
            self.moving_obstacle_patches.append(rect)
            
        # 绘制捕食者
        self.predator_patches = []
        for predator in state.get('predators', []):
            circle = Circle((predator[0] + 0.5, predator[1] + 0.5), 0.4,
                          facecolor='red', edgecolor='black')
            self.ax.add_patch(circle)
            self.predator_patches.append(circle)
            
        # 绘制目标
        target_pos = state.get('target_position')
        if target_pos is not None:
            self.target_patch = Rectangle((target_pos[0], target_pos[1]), 1, 1,
                                        facecolor='green', edgecolor='black')
            self.ax.add_patch(self.target_patch)
            
        # 绘制智能体
        agent_pos = state.get('agent_position')
        if agent_pos is not None:
            self.agent_patch = Circle((agent_pos[0] + 0.5, agent_pos[1] + 0.5), 0.4,
                                    facecolor='blue', edgecolor='black')
            self.ax.add_patch(self.agent_patch)
            
        # 添加图例
        self.ax.legend(['障碍物', '移动障碍物', '捕食者', '目标', '智能体'],
                      loc='upper right')
            
    def update(self, state: Dict[str, Any]) -> None:
        """
        更新环境状态
        
        Args:
            state: 环境状态
        """
        if not self.initialized:
            self.render(state)
            return
            
        # 更新智能体位置
        agent_pos = state.get('agent_position')
        if agent_pos is not None and self.agent_patch is not None:
            self.agent_patch.center = (agent_pos[0] + 0.5, agent_pos[1] + 0.5)
            
        # 更新目标位置
        target_pos = state.get('target_position')
        if target_pos is not None and self.target_patch is not None:
            self.target_patch.set_xy((target_pos[0], target_pos[1]))
            
        # 更新移动障碍物位置
        for i, obstacle in enumerate(state.get('moving_obstacles', [])):
            if i < len(self.moving_obstacle_patches):
                self.moving_obstacle_patches[i].set_xy((obstacle[0], obstacle[1]))
                
        # 更新捕食者位置
        for i, predator in enumerate(state.get('predators', [])):
            if i < len(self.predator_patches):
                self.predator_patches[i].center = (predator[0] + 0.5, predator[1] + 0.5)
                
        # 刷新图形
        self.fig.canvas.draw()
        self.fig.canvas.flush_events() 