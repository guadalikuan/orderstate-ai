from typing import Dict, Any, Optional
import time
from datetime import datetime
import os
import numpy as np

from miniexp.config import Config
from miniexp.env.advanced import AdvancedEnvironment
from miniexp.env.simple import SimpleEnvironment
from miniexp.env.medium import MediumEnvironment
from miniexp.agent.abilities.perception import PerceptionAbility
from miniexp.agent.abilities.decision import DecisionAbility
from miniexp.evolution.evaluator import EvolutionEvaluator
from miniexp.visualization.visualizer import StateVisualizer, EvolutionVisualizer, MetricsVisualizer
from miniexp.experiment.metrics import ExperimentMetrics
from miniexp.agent.baseline import BaselineAgent
from miniexp.agent.energy import EnergyAgent
from miniexp.experiment.metrics_recorder import MetricsRecorder
from miniexp.experiment.experiment_report import ExperimentReport

class Experiment:
    """
    实验管理类
    负责运行实验、记录数据和生成报告
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化实验
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = Config(config_path)
        
        # 初始化环境
        self.env_type = self.config.get('experiment.env_type', 'advanced')
        self.env = self._create_environment()
        
        # 初始化智能体
        self.agents = self._create_agents()
        
        # 初始化指标记录器
        self.metrics_recorder = MetricsRecorder()
        
        # 初始化可视化器
        self.visualizer = None
        if self.config.get('experiment.visualization', False):
            self.visualizer = StateVisualizer(
                width=self.env.width,
                height=self.env.height,
                env_type=self.env_type
            )
            
    def _create_environment(self):
        """
        创建环境实例
        """
        width = self.config.get('environment.width', 10)
        height = self.config.get('environment.height', 10)
        
        if self.env_type == 'simple':
            return SimpleEnvironment(width=width, height=height)
        elif self.env_type == 'medium':
            return MediumEnvironment(width=width, height=height)
        else:
            return AdvancedEnvironment(width=width, height=height)
            
    def _create_agents(self):
        """
        创建智能体实例
        """
        agents = []
        
        # 创建基线智能体
        if self.config.get('agents.baseline.enabled', True):
            agents.append(BaselineAgent(
                env=self.env,
                name="BaselineAgent"
            ))
            
        # 创建能量管理智能体
        if self.config.get('agents.energy.enabled', True):
            agents.append(EnergyAgent(
                env=self.env,
                init_energy=self.config.get('agents.energy.init_energy', 100),
                threshold=self.config.get('agents.energy.threshold', 20),
                name="EnergyAgent"
            ))
            
        return agents
        
    def change_environment(self, env_type: str):
        """
        切换环境类型
        
        Args:
            env_type: 新的环境类型
        """
        self.env_type = env_type
        self.env = self._create_environment()
        
        # 更新智能体的环境
        for agent in self.agents:
            agent.env = self.env
            
        # 更新可视化器
        if self.visualizer:
            self.visualizer.change_environment(env_type)
            
    def run(self):
        """
        运行实验
        """
        try:
            # 获取实验参数
            num_episodes = self.config.get('experiment.num_episodes', 100)
            max_steps = self.config.get('experiment.max_steps', 1000)
            
            # 运行每个智能体的实验
            for agent in self.agents:
                print(f"\n开始测试智能体: {agent.name}")
                
                for episode in range(num_episodes):
                    # 重置环境和智能体
                    state = self.env.reset()
                    agent.reset()
                    
                    # 回合状态变量
                    done = False
                    steps = 0
                    total_reward = 0
                    
                    # 运行单个回合
                    while not done and steps < max_steps:
                        # 执行一步
                        action = agent.act(state)
                        next_state, reward, done, info = self.env.step(action)
                        
                        # 更新状态
                        state = next_state
                        steps += 1
                        total_reward += reward
                        
                        # 更新可视化
                        if self.visualizer:
                            self.visualizer.update({
                                'agent_pos': state['agent_pos'],
                                'target_pos': state['target_pos'],
                                'obstacles': state.get('obstacles', []),
                                'predators': state.get('predators', []),
                                'steps': steps,
                                'reward': total_reward,
                                'distance_to_target': info.get('distance_to_target', 0),
                                'obstacles_avoided': info.get('obstacles_avoided', 0),
                                'predators_avoided': info.get('predators_avoided', 0),
                                'perception_level': agent.perception.level,
                                'perception_range': agent.perception.range,
                                'perception_precision': agent.perception.precision,
                                'decision_level': agent.decision.level,
                                'decision_quality': agent.decision.quality,
                                'decision_speed': agent.decision.speed
                            })
                    
                    # 记录实验结果
                    self.metrics_recorder.record(
                        agent_name=agent.name,
                        success=info.get('reached_target', False),
                        steps=steps,
                        reward=total_reward,
                        energy_left=agent.get_remaining_energy() if isinstance(agent, EnergyAgent) else np.nan,
                        energy_exhausted=agent.is_energy_exhausted() if isinstance(agent, EnergyAgent) else False
                    )
                    
                    # 定期打印进度
                    if (episode + 1) % (num_episodes // 10) == 0 or episode == 0:
                        print(f"  完成 {episode + 1}/{num_episodes} 回合")
                
                print(f"智能体 {agent.name} 测试完成")
                
            # 生成实验报告
            self._generate_report()
            
        except KeyboardInterrupt:
            print("\n实验被中断")
        finally:
            self.stop()
            
    def _generate_report(self):
        """
        生成实验报告
        """
        # 汇总实验结果
        summary_df = self.metrics_recorder.summarize(plot=True)
        
        # 创建报告生成器
        report_generator = ExperimentReport(
            env_type=self.env_type,
            config=self.config,
            metrics=self.metrics_recorder.metrics
        )
        
        # 生成报告
        report_path = report_generator.generate()
        print(f"\n实验报告已生成: {report_path}")
        
    def stop(self):
        """
        停止实验
        """
        if self.visualizer:
            self.visualizer.show() 