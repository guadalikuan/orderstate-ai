import numpy as np
import pandas as pd
from collections import defaultdict

# 检查matplotlib是否可用
try:
    import matplotlib.pyplot as plt
    matplotlib_available = True
except ImportError:
    matplotlib_available = False
    print("Warning: matplotlib not available. Visualization will be disabled.")

class MetricsRecorder:
    """
    记录和汇总实验评价指标。
    可以跟踪多个智能体的性能指标并生成比较报告。
    """
    def __init__(self):
        """
        初始化指标记录器。
        """
        # 使用defaultdict存储每个智能体的指标
        self.results = defaultdict(lambda: defaultdict(list))

    def record(self, agent_name: str, success: bool, steps: int, energy_left: float = np.nan, energy_exhausted: bool = False):
        """
        记录一次实验的结果。

        Args:
            agent_name (str): 智能体名称。
            success (bool): 是否成功到达目标。
            steps (int): 实验步数。
            energy_left (float, optional): 剩余能量。对BaselineAgent可设为np.nan。
            energy_exhausted (bool, optional): 是否因能量耗尽而失败。默认为False。
        """
        self.results[agent_name]['success'].append(success)
        self.results[agent_name]['steps'].append(steps)
        self.results[agent_name]['energy_left'].append(energy_left)
        self.results[agent_name]['energy_exhausted'].append(energy_exhausted)

    def summarize(self, plot: bool = True):
        """
        汇总并打印实验结果，可选地绘制对比图表。

        Args:
            plot (bool, optional): 是否绘制图表。需要matplotlib。默认为True。

        Returns:
            pd.DataFrame: 包含汇总指标的DataFrame。
        """
        if not self.results:
            print("No data available.")
            return None

        # 准备汇总数据
        summary_data = {}
        
        for agent_name, metrics in self.results.items():
            # 确保有数据
            if not metrics['success']:
                continue
                
            # 计算关键指标
            success_rate = np.mean(metrics['success']) * 100
            avg_steps = np.mean(metrics['steps'])
            
            # 计算成功实验的平均步数
            success_steps = [s for s, succ in zip(metrics['steps'], metrics['success']) if succ]
            avg_steps_success = np.mean(success_steps) if success_steps else np.nan
            
            # 能量相关指标 (如果适用)
            energy_values = [e for e in metrics['energy_left'] if not np.isnan(e)]
            avg_energy_left = np.mean(energy_values) if energy_values else np.nan
            
            energy_exhausted_rate = np.mean(metrics['energy_exhausted']) * 100
            
            summary_data[agent_name] = {
                '成功率 (%)': success_rate,
                '平均步数': avg_steps,
                '成功时平均步数': avg_steps_success,
                '平均剩余能量': avg_energy_left,
                '能量耗尽失败率 (%)': energy_exhausted_rate,
                '实验次数': len(metrics['success'])
            }
        
        # 创建DataFrame并打印
        summary_df = pd.DataFrame(summary_data).T
        print("\n--- 实验结果汇总 ---")
        print(summary_df)
        
        # 可选地绘制图表
        if plot and matplotlib_available and len(summary_data) > 0:
            self._plot_comparison(summary_df)
            
        return summary_df
    
    def _plot_comparison(self, summary_df):
        """
        绘制多个智能体之间的性能对比图。

        Args:
            summary_df (pd.DataFrame): 包含汇总指标的DataFrame。
        """
        try:
            # 选择要绘制的指标
            metrics_to_plot = ['成功率 (%)', '平均步数', '成功时平均步数', '平均剩余能量', '能量耗尽失败率 (%)']
            
            # 创建一个合适大小的图形
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # 使用pandas的绘图功能
            summary_df[metrics_to_plot].plot(kind='bar', ax=ax)
            
            # 添加标题和标签
            ax.set_title('智能体性能对比')
            ax.set_ylabel('值')
            ax.set_xlabel('智能体')
            ax.legend(title='指标')
            
            # 添加数据标签
            for container in ax.containers:
                ax.bar_label(container, fmt='%.1f', fontsize=8)
            
            # 调整布局
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"绘图时出错: {e}")

# 测试代码
if __name__ == '__main__':
    # 创建指标记录器
    recorder = MetricsRecorder()
    
    # 记录一些测试数据
    # BaselineAgent (无能量管理)
    for i in range(10):
        success = i > 3  # 60% 成功率
        steps = 10 + i if success else 20
        recorder.record(agent_name='BaselineAgent', success=success, steps=steps)
    
    # EnergyAgent (有能量管理)
    for i in range(10):
        success = i > 2  # 70% 成功率
        energy_exhausted = i == 1 or i == 2  # 20% 能量耗尽率
        steps = 8 + i if success else 18
        energy_left = 10 - i if success else 0
        recorder.record(
            agent_name='EnergyAgent',
            success=success,
            steps=steps,
            energy_left=energy_left,
            energy_exhausted=energy_exhausted
        )
    
    # 汇总并绘图
    recorder.summarize(plot=matplotlib_available) 