from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from matplotlib.widgets import Button, Slider
import matplotlib.gridspec as gridspec

class StateVisualizer:
    """
    状态可视化器
    展示智能体的当前状态和环境
    """
    
    def __init__(self, width: int, height: int):
        """
        初始化可视化器
        
        Args:
            width: 环境宽度
            height: 环境高度
        """
        # 设置样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 创建图形
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])
        
        # 主网格图
        self.ax_grid = self.fig.add_subplot(gs[:, 0])
        self.grid = np.zeros((height, width))
        self.im = self.ax_grid.imshow(self.grid, cmap='viridis')
        
        # 状态信息图
        self.ax_info = self.fig.add_subplot(gs[0, 1])
        self.ax_info.axis('off')
        
        # 能力信息图
        self.ax_abilities = self.fig.add_subplot(gs[1, 1])
        self.ax_abilities.axis('off')
        
        # 设置颜色映射
        self.cmap = {
            0: [0.9, 0.9, 0.9],  # 空地
            1: [0.2, 0.6, 1.0],  # 智能体
            2: [0.2, 0.8, 0.2],  # 目标
            3: [0.5, 0.5, 0.5],  # 障碍物
            4: [0.8, 0.2, 0.2]   # 捕食者
        }
        
        # 添加控制按钮
        self.ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.button = Button(self.ax_button, '暂停/继续')
        self.paused = False
        
        # 添加速度滑块
        self.ax_slider = plt.axes([0.1, 0.05, 0.2, 0.04])
        self.slider = Slider(self.ax_slider, '速度', 0.1, 2.0, valinit=1.0)
        
        # 设置回调
        self.button.on_clicked(self.toggle_pause)
        self.slider.on_changed(self.update_speed)
        
    def toggle_pause(self, event):
        """
        切换暂停状态
        """
        self.paused = not self.paused
        
    def update_speed(self, val):
        """
        更新显示速度
        """
        self.speed = val
        
    def update(self, state: Dict[str, Any]):
        """
        更新状态显示
        
        Args:
            state: 环境状态
        """
        if self.paused:
            return
            
        # 清空网格
        self.grid.fill(0)
        
        # 更新智能体位置
        agent_pos = state.get('agent_pos', (0, 0))
        self.grid[agent_pos[0], agent_pos[1]] = 1
        
        # 更新目标位置
        target_pos = state.get('target_pos', (self.height-1, self.width-1))
        self.grid[target_pos[0], target_pos[1]] = 2
        
        # 更新障碍物
        for obstacle in state.get('obstacles', []):
            self.grid[obstacle[0], obstacle[1]] = 3
            
        # 更新捕食者
        for predator in state.get('predators', []):
            self.grid[predator[0], predator[1]] = 4
            
        # 更新显示
        self.im.set_array(self.grid)
        
        # 更新状态信息
        self._update_info(state)
        
        # 更新能力信息
        self._update_abilities(state)
        
        plt.draw()
        plt.pause(0.1 / self.speed)
        
    def _update_info(self, state: Dict[str, Any]):
        """
        更新状态信息显示
        """
        self.ax_info.clear()
        self.ax_info.axis('off')
        
        info_text = (
            f"步数: {state.get('steps', 0)}\n"
            f"奖励: {state.get('reward', 0):.2f}\n"
            f"到目标距离: {state.get('distance_to_target', 0):.2f}\n"
            f"避障次数: {state.get('obstacles_avoided', 0)}\n"
            f"避捕食者次数: {state.get('predators_avoided', 0)}"
        )
        
        self.ax_info.text(0.1, 0.9, info_text, fontsize=12)
        
    def _update_abilities(self, state: Dict[str, Any]):
        """
        更新能力信息显示
        """
        self.ax_abilities.clear()
        self.ax_abilities.axis('off')
        
        abilities_text = (
            f"感知能力等级: {state.get('perception_level', 1)}\n"
            f"感知范围: {state.get('perception_range', 1)}\n"
            f"感知精度: {state.get('perception_precision', 0.6):.2f}\n"
            f"决策能力等级: {state.get('decision_level', 1)}\n"
            f"决策质量: {state.get('decision_quality', 0.6):.2f}\n"
            f"决策速度: {state.get('decision_speed', 1)}"
        )
        
        self.ax_abilities.text(0.1, 0.9, abilities_text, fontsize=12)
        
    def show(self):
        """
        显示当前状态
        """
        plt.show()
        
class EvolutionVisualizer:
    """
    进化可视化器
    展示智能体的进化过程和表现
    """
    
    def __init__(self):
        """
        初始化可视化器
        """
        # 设置样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 创建图形
        self.fig = plt.figure(figsize=(20, 15))
        gs = gridspec.GridSpec(3, 2)
        
        # 创建子图
        self.axs = {
            'rewards': self.fig.add_subplot(gs[0, 0]),
            'abilities': self.fig.add_subplot(gs[0, 1]),
            'survival': self.fig.add_subplot(gs[1, 0]),
            'success': self.fig.add_subplot(gs[1, 1]),
            'obstacles': self.fig.add_subplot(gs[2, 0]),
            'predators': self.fig.add_subplot(gs[2, 1])
        }
        
        # 初始化数据
        self.metrics = {
            'episode': [],
            'rewards': [],
            'perception_level': [],
            'decision_level': [],
            'survival_time': [],
            'target_reached': [],
            'obstacles_avoided': [],
            'predators_avoided': []
        }
        
    def update(self, metrics: Dict[str, List[float]]):
        """
        更新进化数据
        
        Args:
            metrics: 评估指标
        """
        self.metrics = metrics
        
        # 清空图表
        for ax in self.axs.values():
            ax.clear()
            
        # 绘制奖励曲线
        self.axs['rewards'].plot(self.metrics['episode'], self.metrics['rewards'], 
                                label='奖励', linewidth=2)
        self.axs['rewards'].set_title('奖励变化', fontsize=14)
        self.axs['rewards'].set_xlabel('周期', fontsize=12)
        self.axs['rewards'].set_ylabel('奖励', fontsize=12)
        self.axs['rewards'].grid(True)
        self.axs['rewards'].legend()
        
        # 绘制能力等级
        self.axs['abilities'].plot(self.metrics['episode'], self.metrics['perception_level'], 
                                 label='感知能力', linewidth=2)
        self.axs['abilities'].plot(self.metrics['episode'], self.metrics['decision_level'],
                                 label='决策能力', linewidth=2)
        self.axs['abilities'].set_title('能力等级变化', fontsize=14)
        self.axs['abilities'].set_xlabel('周期', fontsize=12)
        self.axs['abilities'].set_ylabel('等级', fontsize=12)
        self.axs['abilities'].grid(True)
        self.axs['abilities'].legend()
        
        # 绘制生存时间
        self.axs['survival'].plot(self.metrics['episode'], self.metrics['survival_time'],
                                label='生存时间', linewidth=2)
        self.axs['survival'].set_title('生存时间变化', fontsize=14)
        self.axs['survival'].set_xlabel('周期', fontsize=12)
        self.axs['survival'].set_ylabel('时间(秒)', fontsize=12)
        self.axs['survival'].grid(True)
        self.axs['survival'].legend()
        
        # 绘制成功率
        success_rate = np.cumsum(self.metrics['target_reached']) / np.arange(1, len(self.metrics['target_reached'])+1)
        self.axs['success'].plot(self.metrics['episode'], success_rate,
                               label='成功率', linewidth=2)
        self.axs['success'].set_title('成功率变化', fontsize=14)
        self.axs['success'].set_xlabel('周期', fontsize=12)
        self.axs['success'].set_ylabel('成功率', fontsize=12)
        self.axs['success'].grid(True)
        self.axs['success'].legend()
        
        # 绘制避障统计
        self.axs['obstacles'].plot(self.metrics['episode'], self.metrics['obstacles_avoided'],
                                 label='避障次数', linewidth=2)
        self.axs['obstacles'].set_title('避障统计', fontsize=14)
        self.axs['obstacles'].set_xlabel('周期', fontsize=12)
        self.axs['obstacles'].set_ylabel('次数', fontsize=12)
        self.axs['obstacles'].grid(True)
        self.axs['obstacles'].legend()
        
        # 绘制避捕食者统计
        self.axs['predators'].plot(self.metrics['episode'], self.metrics['predators_avoided'],
                                 label='避捕食者次数', linewidth=2)
        self.axs['predators'].set_title('避捕食者统计', fontsize=14)
        self.axs['predators'].set_xlabel('周期', fontsize=12)
        self.axs['predators'].set_ylabel('次数', fontsize=12)
        self.axs['predators'].grid(True)
        self.axs['predators'].legend()
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def show(self):
        """
        显示进化图表
        """
        plt.show()
        
class MetricsVisualizer:
    """
    指标可视化器
    展示详细的评估指标
    """
    
    def __init__(self):
        """
        初始化可视化器
        """
        # 设置样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 创建图形
        self.fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        
        # 创建子图
        self.ax_metrics = self.fig.add_subplot(gs[0])
        self.ax_summary = self.fig.add_subplot(gs[1])
        
    def update(self, metrics: Dict[str, List[float]], title: str = "评估指标"):
        """
        更新指标显示
        
        Args:
            metrics: 评估指标
            title: 图表标题
        """
        # 清空图表
        self.ax_metrics.clear()
        self.ax_summary.clear()
        
        # 准备数据
        data = []
        labels = []
        for key, values in metrics.items():
            if key != 'episode':
                data.append(values[-1])
                labels.append(key)
                
        # 绘制柱状图
        x = np.arange(len(labels))
        bars = self.ax_metrics.bar(x, data)
        self.ax_metrics.set_xticks(x)
        self.ax_metrics.set_xticklabels(labels, rotation=45)
        self.ax_metrics.set_title(title, fontsize=14)
        self.ax_metrics.grid(True)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            self.ax_metrics.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.2f}',
                               ha='center', va='bottom')
        
        # 绘制总结信息
        summary_text = (
            f"总周期数: {len(metrics['episode'])}\n"
            f"平均奖励: {np.mean(metrics['rewards']):.2f}\n"
            f"平均生存时间: {np.mean(metrics['survival_time']):.2f}\n"
            f"最终感知能力: {metrics['perception_level'][-1]}\n"
            f"最终决策能力: {metrics['decision_level'][-1]}"
        )
        
        self.ax_summary.text(0.1, 0.5, summary_text, fontsize=12)
        self.ax_summary.axis('off')
        
        plt.tight_layout()
        plt.draw()
        plt.pause(0.1)
        
    def show(self):
        """
        显示指标图表
        """
        plt.show() 