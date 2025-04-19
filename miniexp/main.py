import numpy as np
import time
import sys
import os
import argparse
import json

# 导入项目模块
from miniexp.env.simple import SimpleEnvironment
from miniexp.env.medium import MediumEnvironment
from miniexp.env.advanced import AdvancedEnvironment
from miniexp.agent.baseline import BaselineAgent
from miniexp.agent.energy import EnergyAgent
from miniexp.experiment.metrics_recorder import MetricsRecorder
from miniexp.experiment.experiment import Experiment
from miniexp.experiment.config import ExperimentConfig, EnvironmentType, AgentType
from miniexp.experiment.evaluator import ExperimentEvaluator

# 实验参数
GRID_WIDTH = 10
GRID_HEIGHT = 10
TARGET_POS = (GRID_HEIGHT - 1, GRID_WIDTH - 1)  # 右下角
START_POS = (0, 0)                               # 左上角
NUM_EPISODES = 100                               # 每个智能体运行的实验次数
MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 2  # 每回合最大步数
INIT_ENERGY = (GRID_WIDTH + GRID_HEIGHT) * 1.5   # 初始能量，略多于曼哈顿距离
ENERGY_THRESHOLD = INIT_ENERGY * 0.2             # 能量阈值，初始能量的20%

def run_experiment(config: ExperimentConfig):
    """
    运行完整实验并报告结果。
    """
    print(f"启动实验 - 网格大小: {config.environment.width}x{config.environment.height}, 回合数: {config.experiment['max_episodes']}")
    print(f"初始能量: {config.agent.initial_energy}, 能量阈值: {config.agent.energy_threshold}")
    
    # 创建环境和智能体
    if config.environment.type == EnvironmentType.SIMPLE:
        env = SimpleEnvironment(width=config.environment.width, height=config.environment.height)
    elif config.environment.type == EnvironmentType.MEDIUM:
        env = MediumEnvironment(width=config.environment.width, height=config.environment.height)
    else:
        env = AdvancedEnvironment(
            width=config.environment.width,
            height=config.environment.height,
            obstacle_density=config.environment.obstacle_density,
            num_predators=config.environment.num_predators,
            num_moving_obstacles=config.environment.num_moving_obstacles
        )
    
    baseline_agent = BaselineAgent(env, name="BaselineAgent")
    energy_agent = EnergyAgent(env, 
                              init_energy=config.agent.initial_energy, 
                              threshold=config.agent.energy_threshold, 
                              name="EnergyAgent")
    
    # 创建指标记录器和评估器
    metrics_recorder = MetricsRecorder()
    evaluator = ExperimentEvaluator(config.to_dict())
    
    # 测试两种智能体
    agents = [baseline_agent, energy_agent]
    
    # 记录实验开始时间
    start_time = time.time()
    
    # 对每个智能体运行实验
    for agent in agents:
        print(f"\n开始测试智能体: {agent.name}")
        
        for episode in range(config.experiment['max_episodes']):
            # 重置环境和智能体
            state = env.reset()
            agent.reset()
            
            # 回合状态变量
            done = False
            steps = 0
            is_success = False
            energy_exhausted_flag = False
            total_reward = 0
            
            # 运行单个回合
            while not done and steps < config.experiment['max_steps_per_episode']:
                # 执行一步
                next_state, reward, done, info = env.step(agent.act(state))
                state = next_state
                steps += 1
                total_reward += reward
                
                # 检查结束条件
                if done:
                    if isinstance(agent, EnergyAgent) and agent.is_energy_exhausted():
                        is_success = False
                        energy_exhausted_flag = True
                    elif info.get('reached_target', False):
                        is_success = True
                    else:
                        is_success = False
            
            # 检查是否超时
            if not done and steps >= config.experiment['max_steps_per_episode']:
                is_success = False
                done = True
            
            # 记录实验结果
            energy_left = np.nan
            if isinstance(agent, EnergyAgent):
                energy_left = agent.get_remaining_energy()
                
            metrics_recorder.record(
                agent_name=agent.name,
                success=is_success,
                steps=steps,
                reward=total_reward,
                energy_left=energy_left,
                energy_exhausted=energy_exhausted_flag
            )
            
            # 更新评估器
            perception_level = 0
            decision_level = 0
            attention_level = 0
            
            # 获取智能体能力等级
            if hasattr(agent, 'perception') and agent.perception:
                perception_level = agent.perception.level if hasattr(agent.perception, 'level') else 0
                
            if hasattr(agent, 'decision') and agent.decision:
                decision_level = agent.decision.level if hasattr(agent.decision, 'level') else 0
                
            if hasattr(agent, 'attention') and agent.attention:
                attention_level = agent.attention.get_attention_level() if hasattr(agent.attention, 'get_attention_level') else 0
            
            evaluator.update_episode({
                'total_reward': total_reward,
                'success': is_success,
                'survival_time': steps,
                'avg_energy': energy_left if not np.isnan(energy_left) else 0,
                'avg_attention': attention_level,
                'avg_perception': perception_level,
                'avg_decision': decision_level
            })
            
            # 定期打印进度
            if config.experiment['max_episodes'] <= 10 or (episode + 1) % (max(1, config.experiment['max_episodes'] // 10)) == 0 or episode == 0:
                print(f"  完成 {episode + 1}/{config.experiment['max_episodes']} 回合")
                
                # 评估进展
                progress = evaluator.evaluate_progress()
                print(f"  当前评估: 平均奖励={progress['avg_reward']:.2f}, 成功率={progress['success_rate']:.2f}")
                if progress['evolution_status']['needs_evolution']:
                    print(f"  检测到进化需求: {progress['evolution_status']['current_stage']}")
        
        print(f"智能体 {agent.name} 测试完成")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n实验完成 - 总耗时: {total_time:.2f} 秒")
    
    # 汇总并显示结果
    summary_df = metrics_recorder.summarize(plot=config.experiment['visualization'])
    
    # 返回汇总数据，以便进一步分析
    return summary_df

def main():
    """
    主程序入口
    """
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行智能体进化实验')
    parser.add_argument('--config', type=str, help='配置文件路径')
    parser.add_argument('--env', type=str, choices=['simple', 'medium', 'advanced'], 
                       default='advanced', help='环境类型')
    parser.add_argument('--episodes', type=int, help='实验周期数')
    parser.add_argument('--visualization', action='store_true', help='启用可视化')
    parser.add_argument('--save-interval', type=int, help='保存间隔')
    args = parser.parse_args()
    
    # 创建默认配置
    config = ExperimentConfig()
    
    # 如果提供了配置文件，则加载
    if args.config:
        config = ExperimentConfig.load(args.config)
    
    # 更新配置
    if args.env:
        config.environment.type = EnvironmentType[args.env.upper()]
    if args.episodes:
        config.experiment['max_episodes'] = args.episodes
    if args.visualization:
        config.experiment['visualization'] = True
    if args.save_interval:
        config.experiment['save_interval'] = args.save_interval
        
    # 运行实验
    try:
        run_experiment(config)
    except KeyboardInterrupt:
        print("\n实验被中断")
    finally:
        # 保存最终配置
        config.save('experiment_config.json')
        print("实验配置已保存到 experiment_config.json")

if __name__ == '__main__':
    main() 