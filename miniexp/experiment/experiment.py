from typing import Dict, Any, Optional
import time
from datetime import datetime
import os

from miniexp.config import Config
from miniexp.env.advanced import AdvancedEnvironment
from miniexp.agent.abilities.perception import PerceptionAbility
from miniexp.agent.abilities.decision import DecisionAbility
from miniexp.evolution.evaluator import EvolutionEvaluator
from miniexp.visualization.visualizer import StateVisualizer, EvolutionVisualizer, MetricsVisualizer
from miniexp.experiment.metrics import ExperimentMetrics

class Experiment:
    """
    主实验类
    整合所有模块并实现实验运行逻辑
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化环境
        env_config = self.config.get_env_config()
        self.env = AdvancedEnvironment(
            width=env_config['width'],
            height=env_config['height'],
            obstacle_density=env_config['obstacle_density'],
            num_predators=env_config['num_predators'],
            num_moving_obstacles=env_config['num_moving_obstacles']
        )
        
        # 初始化智能体能力
        agent_config = self.config.get_agent_config()
        self.perception = PerceptionAbility(level=agent_config['initial_perception_level'])
        self.decision = DecisionAbility(level=agent_config['initial_decision_level'])
        
        # 初始化评估器
        self.evaluator = EvolutionEvaluator()
        self.metrics = ExperimentMetrics()
        
        # 初始化可视化器
        self.state_viz = StateVisualizer(env_config['width'], env_config['height'])
        self.evolution_viz = EvolutionVisualizer()
        self.metrics_viz = MetricsVisualizer()
        
        # 实验状态
        self.current_episode = 0
        self.running = False
        
    def run_episode(self) -> Dict[str, Any]:
        """
        运行一个实验周期
        
        Returns:
            Dict[str, Any]: 周期结果
        """
        # 重置环境
        state = self.env.reset()
        self.evaluator.start_episode()
        
        # 初始化周期状态
        episode_info = {
            'episode': self.current_episode,
            'steps': 0,
            'rewards': 0,
            'target_reached': False,
            'obstacles_avoided': 0,
            'predators_avoided': 0,
            'perception_level': self.perception.level,
            'decision_level': self.decision.level
        }
        
        # 运行周期
        while True:
            # 获取观察
            observation = self.perception.get_observation(state['agent_pos'], state)
            
            # 做出决策
            action = self.decision.make_decision(observation, [0, 1, 2, 3])
            
            # 执行动作
            next_state, reward, done, info = self.env.step(action)
            
            # 更新状态
            state = next_state
            episode_info['steps'] += 1
            episode_info['rewards'] += reward
            
            # 更新评估器
            self.evaluator.update(info)
            
            # 更新可视化
            if self.config.get('experiment.visualization'):
                self.state_viz.update(state)
                
            # 检查是否结束
            if done or episode_info['steps'] >= self.config.get('experiment.max_steps_per_episode'):
                break
                
        # 结束周期
        self.evaluator.end_episode({
            'perception_level': self.perception.level,
            'decision_level': self.decision.level
        })
        
        # 更新指标
        self.metrics.update(episode_info)
        
        return episode_info
        
    def run(self):
        """
        运行实验
        """
        self.running = True
        self.current_episode = 0
        
        try:
            while self.running and self.current_episode < self.config.get('experiment.max_episodes'):
                # 运行周期
                episode_info = self.run_episode()
                self.current_episode += 1
                
                # 更新可视化
                if self.config.get('experiment.visualization'):
                    self.evolution_viz.update(self.metrics.get_metrics())
                    self.metrics_viz.update(self.metrics.get_metrics())
                    
                # 检查进化
                should_evolve, reason = self.evaluator.should_evolve()
                if should_evolve:
                    # 进化感知能力
                    if self.perception.level < self.config.get('agent.max_perception_level'):
                        self.perception.upgrade()
                        
                    # 进化决策能力
                    if self.decision.level < self.config.get('agent.max_decision_level'):
                        self.decision.upgrade()
                        
                # 保存结果
                if self.current_episode % self.config.get('experiment.save_interval') == 0:
                    self.save_results()
                    
        except KeyboardInterrupt:
            print("实验被中断")
        finally:
            self.running = False
            self.save_results()
            
    def save_results(self):
        """
        保存实验结果
        """
        # 创建结果目录
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'experiment_{timestamp}.json'
        filepath = os.path.join(results_dir, filename)
        
        # 保存结果
        self.metrics.save_results(filepath)
        print(f"结果已保存到: {filepath}")
        
    def stop(self):
        """
        停止实验
        """
        self.running = False 