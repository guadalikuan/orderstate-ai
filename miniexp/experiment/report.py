from typing import Dict, Any, List
import json
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from jinja2 import Environment, FileSystemLoader

class ExperimentReport:
    """
    实验报告生成器
    生成详细的实验报告，包括数据分析和可视化
    """
    
    def __init__(self, experiment_name: str, config: Dict[str, Any]):
        """
        初始化报告生成器
        
        Args:
            experiment_name: 实验名称
            config: 实验配置
        """
        self.experiment_name = experiment_name
        self.config = config
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
        self.evolution_history = []
        
        # 创建报告目录
        self.report_dir = f"reports/{experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.report_dir, exist_ok=True)
        
        # 初始化Jinja2环境
        self.env = Environment(loader=FileSystemLoader('templates'))
        
    def update_metrics(self, metrics: Dict[str, List[float]]):
        """
        更新实验指标
        
        Args:
            metrics: 实验指标
        """
        self.metrics = metrics
        
    def record_evolution(self, episode: int, reason: str):
        """
        记录进化事件
        
        Args:
            episode: 进化发生的周期
            reason: 进化原因
        """
        self.evolution_history.append({
            'episode': episode,
            'reason': reason,
            'perception_level': self.metrics['perception_level'][-1],
            'decision_level': self.metrics['decision_level'][-1]
        })
        
    def _generate_plots(self):
        """
        生成实验数据图表
        """
        # 设置样式
        plt.style.use('seaborn')
        sns.set_palette("husl")
        
        # 创建图表目录
        plots_dir = os.path.join(self.report_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        
        # 设置图表大小和DPI
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 100
        
        # 生成奖励曲线
        plt.figure()
        plt.plot(self.metrics['episode'], self.metrics['rewards'], 
                 linewidth=2, color='#3498db', alpha=0.8)
        plt.fill_between(self.metrics['episode'], self.metrics['rewards'],
                        alpha=0.2, color='#3498db')
        plt.title('奖励变化趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('奖励', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'rewards.png'), bbox_inches='tight')
        plt.close()
        
        # 生成能力等级曲线
        plt.figure()
        plt.plot(self.metrics['episode'], self.metrics['perception_level'], 
                 label='感知能力', linewidth=2, color='#2ecc71')
        plt.plot(self.metrics['episode'], self.metrics['decision_level'], 
                 label='决策能力', linewidth=2, color='#e74c3c')
        plt.fill_between(self.metrics['episode'], self.metrics['perception_level'],
                        alpha=0.2, color='#2ecc71')
        plt.fill_between(self.metrics['episode'], self.metrics['decision_level'],
                        alpha=0.2, color='#e74c3c')
        plt.title('能力等级变化趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('等级', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'abilities.png'), bbox_inches='tight')
        plt.close()
        
        # 生成生存时间曲线
        plt.figure()
        plt.plot(self.metrics['episode'], self.metrics['survival_time'],
                 linewidth=2, color='#9b59b6', alpha=0.8)
        plt.fill_between(self.metrics['episode'], self.metrics['survival_time'],
                        alpha=0.2, color='#9b59b6')
        plt.title('生存时间变化趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('时间(秒)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'survival.png'), bbox_inches='tight')
        plt.close()
        
        # 生成成功率曲线
        success_rate = np.cumsum(self.metrics['target_reached']) / np.arange(1, len(self.metrics['target_reached'])+1)
        plt.figure()
        plt.plot(self.metrics['episode'], success_rate,
                 linewidth=2, color='#f1c40f', alpha=0.8)
        plt.fill_between(self.metrics['episode'], success_rate,
                        alpha=0.2, color='#f1c40f')
        plt.title('成功率变化趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('成功率', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'success_rate.png'), bbox_inches='tight')
        plt.close()
        
        # 生成避障统计
        plt.figure()
        plt.plot(self.metrics['episode'], self.metrics['obstacles_avoided'],
                 linewidth=2, color='#1abc9c', alpha=0.8)
        plt.fill_between(self.metrics['episode'], self.metrics['obstacles_avoided'],
                        alpha=0.2, color='#1abc9c')
        plt.title('避障统计趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('次数', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'obstacles.png'), bbox_inches='tight')
        plt.close()
        
        # 生成避捕食者统计
        plt.figure()
        plt.plot(self.metrics['episode'], self.metrics['predators_avoided'],
                 linewidth=2, color='#e67e22', alpha=0.8)
        plt.fill_between(self.metrics['episode'], self.metrics['predators_avoided'],
                        alpha=0.2, color='#e67e22')
        plt.title('避捕食者统计趋势', fontsize=14, pad=20)
        plt.xlabel('周期', fontsize=12)
        plt.ylabel('次数', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'predators.png'), bbox_inches='tight')
        plt.close()
        
        # 生成综合性能雷达图
        plt.figure()
        categories = ['奖励', '感知能力', '决策能力', '生存时间', '成功率', '避障', '避捕食者']
        values = [
            np.mean(self.metrics['rewards']),
            np.mean(self.metrics['perception_level']),
            np.mean(self.metrics['decision_level']),
            np.mean(self.metrics['survival_time']),
            np.mean(success_rate),
            np.mean(self.metrics['obstacles_avoided']),
            np.mean(self.metrics['predators_avoided'])
        ]
        
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, linewidth=2, color='#3498db')
        ax.fill(angles, values, alpha=0.25, color='#3498db')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        plt.title('综合性能评估', fontsize=14, pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'performance_radar.png'), bbox_inches='tight')
        plt.close()
        
    def _calculate_statistics(self) -> Dict[str, Any]:
        """
        计算实验统计数据
        
        Returns:
            Dict[str, Any]: 统计数据
        """
        return {
            'total_episodes': len(self.metrics['episode']),
            'average_reward': np.mean(self.metrics['rewards']),
            'max_reward': np.max(self.metrics['rewards']),
            'min_reward': np.min(self.metrics['rewards']),
            'final_perception_level': self.metrics['perception_level'][-1],
            'final_decision_level': self.metrics['decision_level'][-1],
            'average_survival_time': np.mean(self.metrics['survival_time']),
            'success_rate': np.mean(self.metrics['target_reached']),
            'total_obstacles_avoided': np.sum(self.metrics['obstacles_avoided']),
            'total_predators_avoided': np.sum(self.metrics['predators_avoided']),
            'evolution_count': len(self.evolution_history)
        }
        
    def generate_report(self):
        """
        生成实验报告
        """
        # 生成图表
        self._generate_plots()
        
        # 计算统计数据
        statistics = self._calculate_statistics()
        
        # 准备报告数据
        report_data = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'config': self.config,
            'statistics': statistics,
            'evolution_history': self.evolution_history,
            'plots': {
                'rewards': 'plots/rewards.png',
                'abilities': 'plots/abilities.png',
                'survival': 'plots/survival.png',
                'success_rate': 'plots/success_rate.png',
                'obstacles': 'plots/obstacles.png',
                'predators': 'plots/predators.png',
                'performance_radar': 'plots/performance_radar.png'
            }
        }
        
        # 加载报告模板
        template = self.env.get_template('report_template.html')
        
        # 渲染报告
        html_content = template.render(**report_data)
        
        # 保存报告
        with open(os.path.join(self.report_dir, 'report.html'), 'w', encoding='utf-8') as f:
            f.write(html_content)
            
        # 保存原始数据
        with open(os.path.join(self.report_dir, 'metrics.json'), 'w') as f:
            json.dump(self.metrics, f, indent=4)
            
        print(f"实验报告已生成: {os.path.join(self.report_dir, 'report.html')}") 