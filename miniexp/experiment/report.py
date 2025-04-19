from typing import Dict, Any, List
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Template
import matplotlib as mpl
import matplotlib.font_manager as fm
from miniexp.utils import setup_chinese_font  # 导入统一的中文字体设置工具

# 初始化中文字体
setup_chinese_font()

class ExperimentReport:
    """
    实验报告生成器
    生成详细的实验报告，包括数据分析和可视化
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: str = "reports"):
        """
        初始化实验报告生成器
        
        Args:
            config: 实验配置
            output_dir: 报告输出目录
        """
        self.experiment_id = config.get('experiment_id')
        self.start_time = datetime.now()
        self.end_time = None
        self.metrics = {
            'episodes': [],
            'rewards': [],
            'success_rates': [],
            'survival_times': [],
            'energy_levels': [],
            'attention_levels': [],
            'perception_levels': [],
            'decision_levels': []
        }
        self.evolution_events = []
        
    def update_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        更新实验指标
        
        Args:
            metrics: 实验指标
        """
        self.metrics['episodes'].append(len(self.metrics['episodes']))
        self.metrics['rewards'].append(metrics.get('avg_reward', 0))
        self.metrics['success_rates'].append(metrics.get('success_rate', 0))
        self.metrics['survival_times'].append(metrics.get('avg_survival_time', 0))
        self.metrics['energy_levels'].append(metrics.get('avg_energy', 0))
        self.metrics['attention_levels'].append(metrics.get('avg_attention', 0))
        self.metrics['perception_levels'].append(metrics.get('avg_perception', 0))
        self.metrics['decision_levels'].append(metrics.get('avg_decision', 0))
        
    def record_evolution(self, event: Dict[str, Any]) -> None:
        """
        记录进化事件
        
        Args:
            event: 进化事件
        """
        self.evolution_events.append({
            'episode': len(self.metrics['episodes']),
            'timestamp': datetime.now().isoformat(),
            'event_type': event.get('type', 'unknown'),
            'description': event.get('description', ''),
            'metrics': event.get('metrics', {})
        })
        
    def generate_plots(self, output_dir: str) -> None:
        """
        生成实验数据图表
        
        Args:
            output_dir: 输出目录
        """
        # 设置样式
        sns.set_style('whitegrid')
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 性能指标趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['episodes'], self.metrics['rewards'], label='平均奖励')
        plt.plot(self.metrics['episodes'], self.metrics['success_rates'], label='成功率')
        plt.plot(self.metrics['episodes'], self.metrics['survival_times'], label='平均生存时间')
        plt.xlabel('回合')
        plt.ylabel('值')
        plt.title('性能指标趋势')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'performance_trend.png'))
        plt.close()
        
        # 智能体状态趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(self.metrics['episodes'], self.metrics['energy_levels'], label='能量水平')
        plt.plot(self.metrics['episodes'], self.metrics['attention_levels'], label='注意力水平')
        plt.plot(self.metrics['episodes'], self.metrics['perception_levels'], label='感知水平')
        plt.plot(self.metrics['episodes'], self.metrics['decision_levels'], label='决策水平')
        plt.xlabel('回合')
        plt.ylabel('值')
        plt.title('智能体状态趋势')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'agent_state_trend.png'))
        plt.close()
        
        # 进化事件分布图
        if self.evolution_events:
            event_types = [event['event_type'] for event in self.evolution_events]
            plt.figure(figsize=(8, 6))
            sns.countplot(x=event_types)
            plt.xlabel('事件类型')
            plt.ylabel('数量')
            plt.title('进化事件分布')
            plt.savefig(os.path.join(output_dir, 'evolution_events.png'))
            plt.close()
            
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        计算实验统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        return {
            'total_episodes': len(self.metrics['episodes']),
            'avg_reward': np.mean(self.metrics['rewards']),
            'max_reward': np.max(self.metrics['rewards']),
            'min_reward': np.min(self.metrics['rewards']),
            'final_success_rate': self.metrics['success_rates'][-1] if self.metrics['success_rates'] else 0,
            'avg_survival_time': np.mean(self.metrics['survival_times']),
            'evolution_events_count': len(self.evolution_events),
            'experiment_duration': (self.end_time - self.start_time).total_seconds() if self.end_time else 0
        }
        
    def generate_report(self, output_dir: str) -> str:
        """
        生成实验报告
        
        Args:
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        self.end_time = datetime.now()
        
        # 生成图表
        self.generate_plots(output_dir)
        
        # 计算统计信息
        stats = self.calculate_statistics()
        
        # 生成HTML报告
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>实验报告 - {{ experiment_id }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .section { margin-bottom: 30px; }
                .metric { margin: 10px 0; }
                img { max-width: 100%; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>实验报告 - {{ experiment_id }}</h1>
            
            <div class="section">
                <h2>实验信息</h2>
                <div class="metric">开始时间: {{ start_time }}</div>
                <div class="metric">结束时间: {{ end_time }}</div>
                <div class="metric">持续时间: {{ duration }}秒</div>
            </div>
            
            <div class="section">
                <h2>性能指标</h2>
                <div class="metric">总回合数: {{ stats.total_episodes }}</div>
                <div class="metric">平均奖励: {{ stats.avg_reward|round(2) }}</div>
                <div class="metric">最大奖励: {{ stats.max_reward|round(2) }}</div>
                <div class="metric">最小奖励: {{ stats.min_reward|round(2) }}</div>
                <div class="metric">最终成功率: {{ (stats.final_success_rate * 100)|round(2) }}%</div>
                <div class="metric">平均生存时间: {{ stats.avg_survival_time|round(2) }}</div>
            </div>
            
            <div class="section">
                <h2>性能趋势</h2>
                <img src="performance_trend.png" alt="性能趋势图">
            </div>
            
            <div class="section">
                <h2>智能体状态</h2>
                <img src="agent_state_trend.png" alt="智能体状态趋势图">
            </div>
            
            <div class="section">
                <h2>进化事件</h2>
                <div class="metric">总事件数: {{ stats.evolution_events_count }}</div>
                <img src="evolution_events.png" alt="进化事件分布图">
                <table>
                    <tr>
                        <th>回合</th>
                        <th>时间</th>
                        <th>类型</th>
                        <th>描述</th>
                    </tr>
                    {% for event in evolution_events %}
                    <tr>
                        <td>{{ event.episode }}</td>
                        <td>{{ event.timestamp }}</td>
                        <td>{{ event.event_type }}</td>
                        <td>{{ event.description }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """)
        
        report_html = template.render(
            experiment_id=self.experiment_id,
            start_time=self.start_time.isoformat(),
            end_time=self.end_time.isoformat(),
            duration=stats['experiment_duration'],
            stats=stats,
            evolution_events=self.evolution_events
        )
        
        # 保存报告
        report_path = os.path.join(output_dir, f'report_{self.experiment_id}.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_html)
            
        # 保存原始数据
        data_path = os.path.join(output_dir, f'data_{self.experiment_id}.json')
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': self.metrics,
                'evolution_events': self.evolution_events,
                'statistics': stats
            }, f, ensure_ascii=False, indent=2)
            
        return report_path 