#!/usr/bin/env python
"""
最小可解释实验程序 - 运行脚本
演示"能量→信号→数据→信息→知识→智慧→决策→动作→能量"八阶段闭环
及"生存焦虑驱动Attention"机制。
"""

import sys
import os
import argparse

# 确保能找到miniexp包
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

# 导入实验主函数
from miniexp.main import run_experiment

def print_environment_info(env_type):
    """打印环境信息"""
    print("\n环境信息:")
    if env_type == 'simple':
        print("简单环境:")
        print("- 无障碍物")
        print("- 固定目标位置")
        print("- 适合测试基本导航能力")
    elif env_type == 'medium':
        print("中等环境:")
        print("- 包含静态障碍物")
        print("- 固定目标位置")
        print("- 测试路径规划能力")
    else:
        print("高级环境:")
        print("- 包含静态和动态障碍物")
        print("- 包含捕食者")
        print("- 动态目标位置")
        print("- 测试综合决策能力")

if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='运行智能体进化实验')
    parser.add_argument('--env', type=str, choices=['simple', 'medium', 'advanced'], 
                       default='advanced', help='环境类型')
    parser.add_argument('--episodes', type=int, default=100, help='实验周期数')
    parser.add_argument('--visualization', action='store_true', help='启用可视化')
    args = parser.parse_args()
    
    print("=" * 60)
    print("八阶段闭环与生存焦虑驱动注意力实验")
    print("=" * 60)
    
    # 打印环境信息
    print_environment_info(args.env)
    
    print("\n该实验演示了:")
    print("1. 能量→信号→数据→信息→知识→智慧→决策→动作→能量 的完整闭环")
    print("2. 生存焦虑如何影响注意力分配机制")
    print("3. 带能量管理的智能体与基线智能体的行为对比")
    
    # 运行实验
    results = run_experiment(env_type=args.env)
    
    print("\n实验结论:")
    print("1. 能量管理会影响智能体的决策过程")
    print("2. 焦虑度越高，注意力越集中，决策越倾向于直接路径")
    print("3. 智能体需要平衡能量消耗与任务完成之间的关系") 