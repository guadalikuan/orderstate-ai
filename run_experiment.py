#!/usr/bin/env python
"""
八阶段闭环与生存焦虑驱动注意力实验 - 命令行启动脚本
"""
import os
import sys
import argparse

# 确保可以找到miniexp包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 从miniexp导入必要的模块
from miniexp.env.simple import SimpleEnvironment
from miniexp.env.medium import MediumEnvironment 
from miniexp.env.advanced import AdvancedEnvironment
from miniexp.agent.baseline import BaselineAgent
from miniexp.agent.energy import EnergyAgent
from miniexp.experiment.experiment import Experiment
from miniexp.experiment.config import ExperimentConfig, EnvironmentType

def main():
    """主程序入口"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行智能体进化实验')
    parser.add_argument('--env', type=str, choices=['simple', 'medium', 'advanced'], 
                       default='medium', help='环境类型')
    parser.add_argument('--episodes', type=int, default=20, help='实验周期数')
    parser.add_argument('--viz', action='store_true', help='启用可视化')
    args = parser.parse_args()
    
    print("=" * 60)
    print("八阶段闭环与生存焦虑驱动注意力实验")
    print("=" * 60)
    
    # 创建配置
    config = ExperimentConfig()
    config.environment.type = EnvironmentType[args.env.upper()]
    config.experiment['max_episodes'] = args.episodes
    config.experiment['visualization'] = args.viz
    
    print(f"启动环境: {args.env}, 回合数: {args.episodes}")
    
    # 启动实验
    try:
        # 使用miniexp.main中的run_experiment函数
        from miniexp.main import run_experiment
        summary = run_experiment(config)
        
        # 打印汇总结果
        print("\n实验结果汇总:")
        print(summary)
        
    except KeyboardInterrupt:
        print("\n实验被中断")
    except Exception as e:
        print(f"\n实验出错: {e}")
    finally:
        print("\n实验结束")

if __name__ == '__main__':
    main() 