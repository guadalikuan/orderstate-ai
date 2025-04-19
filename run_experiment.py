#!/usr/bin/env python
"""
最小可解释实验程序 - 运行脚本
演示"能量→信号→数据→信息→知识→智慧→决策→动作→能量"八阶段闭环
及"生存焦虑驱动Attention"机制。
"""

import sys
import os

# 确保能找到miniexp包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入实验主函数
from miniexp.main import run_experiment

if __name__ == "__main__":
    print("=" * 60)
    print("八阶段闭环与生存焦虑驱动注意力实验")
    print("=" * 60)
    print("\n该实验演示了:")
    print("1. 能量→信号→数据→信息→知识→智慧→决策→动作→能量 的完整闭环")
    print("2. 生存焦虑如何影响注意力分配机制")
    print("3. 带能量管理的智能体与基线智能体的行为对比")
    
    # 运行实验
    results = run_experiment()
    
    print("\n实验结论:")
    print("1. 能量管理会影响智能体的决策过程")
    print("2. 焦虑度越高，注意力越集中，决策越倾向于直接路径")
    print("3. 智能体需要平衡能量消耗与任务完成之间的关系") 