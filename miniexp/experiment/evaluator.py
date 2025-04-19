from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from collections import defaultdict
from datetime import datetime

class ExperimentEvaluator:
    """
    实验评估器
    用于评估实验过程中的进度和智能体性能
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化评估器
        
        Args:
            config: 实验配置
        """
        self.config = config
        self.episode_metrics = defaultdict(list)
        self.evolution_metrics = {
            'success_threshold': 0.6,  # 成功率阈值
            'survival_threshold': 50,  # 生存时间阈值
            'progress_indicators': [],
            'last_evolution_point': 0,
            'current_stage': 'initial',
            'stages': ['initial', 'perception', 'decision', 'advanced'],
        }
        self.current_episode = 0
        self.evolution_stages = []
        self.start_time = datetime.now()
        
        # 评估指标
        self.metrics = {
            'rewards': [],
            'success_rates': [],
            'survival_times': [],
            'energy_levels': [],
            'attention_levels': [],
            'perception_levels': [],
            'decision_levels': []
        }
        
    def update_episode(self, metrics: Dict[str, Any]) -> None:
        """
        更新单个回合的指标
        
        Args:
            metrics: 单回合的指标数据
        """
        self.current_episode += 1
        
        # 更新defaultdict格式的指标
        for key, value in metrics.items():
            self.episode_metrics[key].append(value)
        
        # 更新累积指标
        self.metrics['rewards'].append(metrics.get('total_reward', 0))
        self.metrics['success_rates'].append(metrics.get('success', 0))
        self.metrics['survival_times'].append(metrics.get('survival_time', 0))
        self.metrics['energy_levels'].append(metrics.get('avg_energy', 0))
        self.metrics['attention_levels'].append(metrics.get('avg_attention', 0))
        self.metrics['perception_levels'].append(metrics.get('avg_perception', 0))
        self.metrics['decision_levels'].append(metrics.get('avg_decision', 0))
        
    def evaluate_progress(self) -> Dict[str, Any]:
        """
        评估实验进展
        
        Returns:
            Dict[str, Any]: 评估结果
        """
        # 计算窗口统计量
        window_size = min(100, len(self.metrics['rewards']))
        if window_size == 0:
            return {
                'episode': self.current_episode,
                'avg_reward': 0.0,
                'success_rate': 0.0,
                'avg_survival_time': 0.0,
                'avg_energy': 0.0,
                'avg_attention': 0.0,
                'avg_perception': 0.0,
                'avg_decision': 0.0,
                'evolution_status': {
                    'current_stage': self.evolution_metrics['current_stage'],
                    'perception_growth': 0.0,
                    'decision_growth': 0.0,
                    'needs_evolution': False
                }
            }
            
        recent_rewards = self.metrics['rewards'][-window_size:]
        recent_success = self.metrics['success_rates'][-window_size:]
        
        # 计算性能指标
        avg_reward = np.mean(recent_rewards)
        success_rate = np.mean(recent_success)
        avg_survival = np.mean(self.metrics['survival_times'][-window_size:])
        
        # 计算能力指标
        avg_energy = np.mean(self.metrics['energy_levels'][-window_size:])
        avg_attention = np.mean(self.metrics['attention_levels'][-window_size:])
        avg_perception = np.mean(self.metrics['perception_levels'][-window_size:])
        avg_decision = np.mean(self.metrics['decision_levels'][-window_size:])
        
        # 评估进化状态
        evolution_status = self._evaluate_evolution()
        
        return {
            'episode': self.current_episode,
            'avg_reward': avg_reward,
            'success_rate': success_rate,
            'avg_survival_time': avg_survival,
            'avg_energy': avg_energy,
            'avg_attention': avg_attention,
            'avg_perception': avg_perception,
            'avg_decision': avg_decision,
            'evolution_status': evolution_status
        }
        
    def _check_evolution_needs(self, success_rate: float, avg_survival_time: float, episode_count: int) -> bool:
        """
        检查是否需要进化
        
        Args:
            success_rate: 成功率
            avg_survival_time: 平均生存时间
            episode_count: 当前回合数
        
        Returns:
            bool: 是否需要进化
        """
        # 记录进度指标
        self.evolution_metrics['progress_indicators'].append({
            'success_rate': success_rate,
            'survival_time': avg_survival_time,
            'episode': episode_count
        })
        
        # 检查自上次进化后的回合数
        episodes_since_last_evolution = episode_count - self.evolution_metrics['last_evolution_point']
        
        # 如果回合数不足，不进化
        if episodes_since_last_evolution < 10:
            return False
        
        # 评估进化条件
        if success_rate >= self.evolution_metrics['success_threshold'] and \
           avg_survival_time >= self.evolution_metrics['survival_threshold']:
            # 满足条件，记录进化点
            self.evolution_metrics['last_evolution_point'] = episode_count
            
            # 更新阶段
            current_index = self.evolution_metrics['stages'].index(self.evolution_metrics['current_stage'])
            if current_index < len(self.evolution_metrics['stages']) - 1:
                self.evolution_metrics['current_stage'] = self.evolution_metrics['stages'][current_index + 1]
                
            # 提高阈值
            self.evolution_metrics['success_threshold'] += 0.1
            self.evolution_metrics['survival_threshold'] *= 1.2
            
            return True
            
        return False
        
    def _calculate_progress(self) -> float:
        """
        计算总体进度
        
        Returns:
            float: 进度百分比
        """
        current_stage_index = self.evolution_metrics['stages'].index(self.evolution_metrics['current_stage'])
        total_stages = len(self.evolution_metrics['stages'])
        
        # 基础进度
        base_progress = current_stage_index / total_stages
        
        # 如果已经是最后阶段，考虑成功率作为额外进度
        if current_stage_index == total_stages - 1 and self.episode_metrics.get('success'):
            recent_success = np.mean(self.episode_metrics['success'][-10:])
            extra_progress = recent_success / total_stages
        else:
            extra_progress = 0
            
        return min(0.99, base_progress + extra_progress)
    
    def summarize(self) -> Dict[str, Any]:
        """
        生成实验评估摘要
        
        Returns:
            Dict: 评估摘要
        """
        if not self.episode_metrics:
            return {"status": "No data available for evaluation"}
            
        # 计算总体指标
        overall_success_rate = np.mean(self.episode_metrics.get('success', [0]))
        overall_avg_reward = np.mean(self.episode_metrics.get('total_reward', [0]))
        overall_avg_survival = np.mean(self.episode_metrics.get('survival_time', [0]))
        
        # 能力指标
        perception_level = np.max(self.episode_metrics.get('avg_perception', [0]))
        decision_level = np.max(self.episode_metrics.get('avg_decision', [0]))
        
        # 学习曲线 (简化为5个点)
        rewards_curve = self._get_learning_curve(self.episode_metrics.get('total_reward', [0]), 5)
        success_curve = self._get_learning_curve(self.episode_metrics.get('success', [0]), 5)
        
        return {
            "overall": {
                "success_rate": overall_success_rate,
                "avg_reward": overall_avg_reward,
                "avg_survival_time": overall_avg_survival,
                "final_perception_level": perception_level,
                "final_decision_level": decision_level,
                "evolution_stage": self.evolution_metrics['current_stage'],
                "progress": self._calculate_progress()
            },
            "learning_curves": {
                "rewards": rewards_curve,
                "success_rate": success_curve
            },
            "evolution_history": self.evolution_metrics['progress_indicators']
        }
    
    def _get_learning_curve(self, data: List[float], num_points: int) -> List[float]:
        """
        获取学习曲线数据
        
        Args:
            data: 原始数据
            num_points: 采样点数
        
        Returns:
            List: 采样后的学习曲线数据
        """
        if not data:
            return [0] * num_points
            
        # 如果数据点少于采样点，返回原始数据
        if len(data) <= num_points:
            return data
            
        # 将数据分成num_points段，计算每段的平均值
        result = []
        segment_size = len(data) // num_points
        
        for i in range(num_points):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_points - 1 else len(data)
            segment_avg = np.mean(data[start_idx:end_idx])
            result.append(segment_avg)
            
        return result
        
    def _evaluate_evolution(self) -> Dict[str, Any]:
        """
        评估进化状态
        
        Returns:
            Dict[str, Any]: 进化状态
        """
        # 计算能力提升
        perception_growth = self._calculate_growth('perception_levels')
        decision_growth = self._calculate_growth('decision_levels')
        
        # 评估进化阶段
        current_stage = self._determine_stage(perception_growth, decision_growth)
        
        # 检查是否需要进化
        needs_evolution = self._check_evolution_need(current_stage)
        
        return {
            'current_stage': current_stage,
            'perception_growth': perception_growth,
            'decision_growth': decision_growth,
            'needs_evolution': needs_evolution
        }
        
    def _calculate_growth(self, metric: str) -> float:
        """
        计算能力增长
        
        Args:
            metric: 指标名称
            
        Returns:
            float: 增长百分比
        """
        if len(self.metrics[metric]) < 2:
            return 0.0
            
        initial = np.mean(self.metrics[metric][:10])
        recent = np.mean(self.metrics[metric][-10:])
        
        if initial == 0:
            return 0.0
            
        return (recent - initial) / initial * 100
        
    def _determine_stage(self, perception_growth: float, decision_growth: float) -> str:
        """
        确定当前进化阶段
        
        Args:
            perception_growth: 感知能力增长
            decision_growth: 决策能力增长
            
        Returns:
            str: 进化阶段
        """
        if perception_growth < 10 and decision_growth < 10:
            return '初始阶段'
        elif perception_growth < 30 and decision_growth < 30:
            return '探索阶段'
        elif perception_growth < 50 and decision_growth < 50:
            return '适应阶段'
        elif perception_growth < 80 and decision_growth < 80:
            return '优化阶段'
        else:
            return '成熟阶段'
            
    def _check_evolution_need(self, current_stage: str) -> bool:
        """
        检查是否需要进化
        
        Args:
            current_stage: 当前阶段
            
        Returns:
            bool: 是否需要进化
        """
        if not self.evolution_stages:
            return True
            
        last_stage = self.evolution_stages[-1]
        if current_stage != last_stage:
            self.evolution_stages.append(current_stage)
            return True
            
        return False
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        获取所有指标数据
        
        Returns:
            Dict[str, List[float]]: 指标数据
        """
        return self.metrics
        
    def get_evolution_history(self) -> List[str]:
        """
        获取进化历史
        
        Returns:
            List[str]: 进化阶段列表
        """
        return self.evolution_stages 