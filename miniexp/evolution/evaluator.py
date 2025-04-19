from typing import Dict, Any, List, Tuple
import numpy as np
from datetime import datetime

class EvolutionEvaluator:
    """
    进化评估器
    评估智能体的表现和进化进度
    """
    
    def __init__(self):
        """
        初始化评估器
        """
        self.history = []
        self.current_episode = 0
        self.metrics = {
            'episode': [],
            'steps': [],
            'rewards': [],
            'perception_level': [],
            'decision_level': [],
            'survival_time': [],
            'target_reached': [],
            'obstacles_avoided': [],
            'predators_avoided': []
        }
        
    def start_episode(self):
        """
        开始新的评估周期
        """
        self.current_episode += 1
        self.current_metrics = {
            'steps': 0,
            'rewards': 0,
            'target_reached': False,
            'obstacles_avoided': 0,
            'predators_avoided': 0,
            'start_time': datetime.now()
        }
        
    def update(self, step_info: Dict[str, Any]):
        """
        更新评估指标
        
        Args:
            step_info: 步骤信息
        """
        self.current_metrics['steps'] += 1
        self.current_metrics['rewards'] += step_info.get('reward', 0)
        
        # 更新避障统计
        if step_info.get('obstacle_avoided', False):
            self.current_metrics['obstacles_avoided'] += 1
            
        # 更新避捕食者统计
        if step_info.get('predator_avoided', False):
            self.current_metrics['predators_avoided'] += 1
            
        # 更新目标达成状态
        if step_info.get('target_reached', False):
            self.current_metrics['target_reached'] = True
            
    def end_episode(self, agent_info: Dict[str, Any]):
        """
        结束评估周期
        
        Args:
            agent_info: 智能体信息
        """
        # 计算生存时间
        end_time = datetime.now()
        survival_time = (end_time - self.current_metrics['start_time']).total_seconds()
        
        # 记录指标
        self.metrics['episode'].append(self.current_episode)
        self.metrics['steps'].append(self.current_metrics['steps'])
        self.metrics['rewards'].append(self.current_metrics['rewards'])
        self.metrics['perception_level'].append(agent_info.get('perception_level', 1))
        self.metrics['decision_level'].append(agent_info.get('decision_level', 1))
        self.metrics['survival_time'].append(survival_time)
        self.metrics['target_reached'].append(self.current_metrics['target_reached'])
        self.metrics['obstacles_avoided'].append(self.current_metrics['obstacles_avoided'])
        self.metrics['predators_avoided'].append(self.current_metrics['predators_avoided'])
        
        # 保存历史记录
        self.history.append({
            'episode': self.current_episode,
            'metrics': self.current_metrics.copy(),
            'agent_info': agent_info.copy(),
            'timestamp': end_time
        })
        
    def evaluate_progress(self) -> Dict[str, float]:
        """
        评估进化进度
        
        Returns:
            Dict[str, float]: 进化进度指标
        """
        if not self.history:
            return {}
            
        # 计算最近10个周期的平均表现
        recent_history = self.history[-10:]
        
        # 计算各项指标
        avg_steps = np.mean([h['metrics']['steps'] for h in recent_history])
        avg_rewards = np.mean([h['metrics']['rewards'] for h in recent_history])
        avg_survival = np.mean([h['metrics']['survival_time'] for h in recent_history])
        success_rate = np.mean([h['metrics']['target_reached'] for h in recent_history])
        
        # 计算能力提升
        perception_growth = self.metrics['perception_level'][-1] - self.metrics['perception_level'][0]
        decision_growth = self.metrics['decision_level'][-1] - self.metrics['decision_level'][0]
        
        return {
            'avg_steps': avg_steps,
            'avg_rewards': avg_rewards,
            'avg_survival': avg_survival,
            'success_rate': success_rate,
            'perception_growth': perception_growth,
            'decision_growth': decision_growth
        }
        
    def should_evolve(self) -> Tuple[bool, str]:
        """
        判断是否应该进化
        
        Returns:
            Tuple[bool, str]: (是否应该进化, 进化原因)
        """
        if len(self.history) < 10:
            return False, "需要更多数据"
            
        progress = self.evaluate_progress()
        
        # 检查是否达到进化条件
        if progress['success_rate'] > 0.8:  # 成功率超过80%
            return True, "高成功率"
            
        if progress['avg_rewards'] > 50:  # 平均奖励超过50
            return True, "高奖励"
            
        if progress['avg_survival'] > 60:  # 平均生存时间超过60秒
            return True, "长生存时间"
            
        return False, "未达到进化条件"
        
    def get_metrics(self) -> Dict[str, List[float]]:
        """
        获取所有评估指标
        
        Returns:
            Dict[str, List[float]]: 评估指标
        """
        return self.metrics.copy()
        
    def get_history(self) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        Returns:
            List[Dict[str, Any]]: 历史记录
        """
        return self.history.copy() 