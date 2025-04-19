from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime

class ExperimentMetrics:
    """
    实验评估指标
    定义和计算实验评估标准
    """
    
    def __init__(self):
        """
        初始化评估指标
        """
        self.metrics = {
            'episode': [],
            'steps': [],
            'rewards': [],
            'perception_level': [],
            'decision_level': [],
            'survival_time': [],
            'target_reached': [],
            'obstacles_avoided': [],
            'predators_avoided': [],
            'energy_level': [],
            'anxiety_level': []
        }
        self.start_time = datetime.now()
        
    def update(self, episode_info: Dict[str, Any]):
        """
        更新评估指标
        
        Args:
            episode_info: 周期信息
        """
        # 记录基本指标
        self.metrics['episode'].append(episode_info.get('episode', 0))
        self.metrics['steps'].append(episode_info.get('steps', 0))
        self.metrics['rewards'].append(episode_info.get('rewards', 0))
        self.metrics['perception_level'].append(episode_info.get('perception_level', 1))
        self.metrics['decision_level'].append(episode_info.get('decision_level', 1))
        self.metrics['survival_time'].append(episode_info.get('survival_time', 0))
        self.metrics['target_reached'].append(episode_info.get('target_reached', False))
        self.metrics['obstacles_avoided'].append(episode_info.get('obstacles_avoided', 0))
        self.metrics['predators_avoided'].append(episode_info.get('predators_avoided', 0))
        self.metrics['energy_level'].append(episode_info.get('energy_level', 1.0))
        self.metrics['anxiety_level'].append(episode_info.get('anxiety_level', 0.0))
        
    def calculate_performance(self) -> Dict[str, float]:
        """
        计算性能指标
        
        Returns:
            Dict[str, float]: 性能指标
        """
        if not self.metrics['episode']:
            return {}
            
        # 计算平均指标
        avg_steps = np.mean(self.metrics['steps'])
        avg_rewards = np.mean(self.metrics['rewards'])
        avg_survival = np.mean(self.metrics['survival_time'])
        success_rate = np.mean(self.metrics['target_reached'])
        
        # 计算能力提升
        perception_growth = self.metrics['perception_level'][-1] - self.metrics['perception_level'][0]
        decision_growth = self.metrics['decision_level'][-1] - self.metrics['decision_level'][0]
        
        # 计算避障能力
        avg_obstacles_avoided = np.mean(self.metrics['obstacles_avoided'])
        avg_predators_avoided = np.mean(self.metrics['predators_avoided'])
        
        # 计算能量管理
        avg_energy = np.mean(self.metrics['energy_level'])
        avg_anxiety = np.mean(self.metrics['anxiety_level'])
        
        return {
            'avg_steps': avg_steps,
            'avg_rewards': avg_rewards,
            'avg_survival': avg_survival,
            'success_rate': success_rate,
            'perception_growth': perception_growth,
            'decision_growth': decision_growth,
            'avg_obstacles_avoided': avg_obstacles_avoided,
            'avg_predators_avoided': avg_predators_avoided,
            'avg_energy': avg_energy,
            'avg_anxiety': avg_anxiety
        }
        
    def evaluate_evolution(self) -> Tuple[bool, str]:
        """
        评估进化效果
        
        Returns:
            Tuple[bool, str]: (是否成功进化, 评估结果)
        """
        performance = self.calculate_performance()
        
        # 检查是否达到进化标准
        if performance['success_rate'] > 0.8:
            return True, "高成功率"
            
        if performance['avg_rewards'] > 50:
            return True, "高奖励"
            
        if performance['avg_survival'] > 60:
            return True, "长生存时间"
            
        if performance['perception_growth'] > 2 or performance['decision_growth'] > 2:
            return True, "显著能力提升"
            
        return False, "未达到进化标准"
        
    def get_summary(self) -> Dict[str, Any]:
        """
        获取实验总结
        
        Returns:
            Dict[str, Any]: 实验总结
        """
        performance = self.calculate_performance()
        evolution_success, evolution_reason = self.evaluate_evolution()
        
        return {
            'total_episodes': len(self.metrics['episode']),
            'total_time': (datetime.now() - self.start_time).total_seconds(),
            'performance': performance,
            'evolution_success': evolution_success,
            'evolution_reason': evolution_reason,
            'final_perception_level': self.metrics['perception_level'][-1],
            'final_decision_level': self.metrics['decision_level'][-1]
        }
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        获取所有评估指标
        
        Returns:
            Dict[str, List[float]]: 评估指标
        """
        return self.metrics.copy()
        
    def save_results(self, filepath: str):
        """
        保存实验结果
        
        Args:
            filepath: 文件路径
        """
        import json
        with open(filepath, 'w') as f:
            json.dump({
                'metrics': self.metrics,
                'summary': self.get_summary()
            }, f, indent=4) 