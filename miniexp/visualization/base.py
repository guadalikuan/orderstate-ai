from abc import ABC, abstractmethod
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np

class BaseVisualizer(ABC):
    """可视化系统基类"""
    
    def __init__(self):
        """初始化可视化器"""
        self.fig = None
        self.ax = None
        self.initialized = False
        
    def initialize(self):
        """初始化图形界面"""
        if not self.initialized:
            self.fig, self.ax = plt.subplots(figsize=(10, 8))
            self.initialized = True
            
    @abstractmethod
    def render(self, state: Dict[str, Any]) -> None:
        """
        渲染当前状态
        
        Args:
            state: 当前环境状态
        """
        pass
        
    @abstractmethod
    def update(self, state: Dict[str, Any]) -> None:
        """
        更新可视化
        
        Args:
            state: 当前环境状态
        """
        pass
        
    def clear(self):
        """清除图形"""
        if self.initialized:
            self.ax.clear()
            
    def show(self):
        """显示图形"""
        if self.initialized:
            plt.show()
            
    def save(self, filename: str):
        """
        保存图形
        
        Args:
            filename: 保存的文件名
        """
        if self.initialized:
            self.fig.savefig(filename)
            
    def close(self):
        """关闭图形"""
        if self.initialized:
            plt.close(self.fig)
            self.initialized = False 