import numpy as np
import time
import sys
import os

# 导入项目模块
from miniexp.env import GridWorld
from miniexp.agent import BaselineAgent, EnergyAgent
from miniexp.metrics import MetricsRecorder, matplotlib_available

# 实验参数
GRID_WIDTH = 10
GRID_HEIGHT = 10
TARGET_POS = (GRID_HEIGHT - 1, GRID_WIDTH - 1)  # 右下角
START_POS = (0, 0)                               # 左上角
NUM_EPISODES = 100                               # 每个智能体运行的实验次数
MAX_STEPS_PER_EPISODE = GRID_WIDTH * GRID_HEIGHT * 2  # 每回合最大步数
INIT_ENERGY = (GRID_WIDTH + GRID_HEIGHT) * 1.5   # 初始能量，略多于曼哈顿距离
ENERGY_THRESHOLD = INIT_ENERGY * 0.2             # 能量阈值，初始能量的20%

def run_experiment():
    """
    运行完整实验并报告结果。
    """
    print(f"启动实验 - 网格大小: {GRID_WIDTH}x{GRID_HEIGHT}, 回合数: {NUM_EPISODES}")
    print(f"初始能量: {INIT_ENERGY}, 能量阈值: {ENERGY_THRESHOLD}")
    
    # 创建环境和智能体
    env = GridWorld(width=GRID_WIDTH, height=GRID_HEIGHT, 
                   target_pos=TARGET_POS, start_pos=START_POS)
    
    baseline_agent = BaselineAgent(env, name="BaselineAgent")
    energy_agent = EnergyAgent(env, 
                              init_energy=INIT_ENERGY, 
                              threshold=ENERGY_THRESHOLD, 
                              name="EnergyAgent")
    
    # 创建指标记录器
    metrics_recorder = MetricsRecorder()
    
    # 测试两种智能体
    agents = [baseline_agent, energy_agent]
    
    # 记录实验开始时间
    start_time = time.time()
    
    # 对每个智能体运行实验
    for agent in agents:
        print(f"\n开始测试智能体: {agent.name}")
        
        for episode in range(NUM_EPISODES):
            # 重置环境和智能体
            state = env.reset()
            agent.reset()
            
            # 回合状态变量
            done = False
            steps = 0
            is_success = False
            energy_exhausted_flag = False
            
            # 运行单个回合
            while not done and steps < MAX_STEPS_PER_EPISODE:
                # --- 八阶段闭环 ---
                # [阶段 8 -> 1: 动作 -> 能量] (EnergyAgent内部处理)
                # [阶段 1 -> 2: 能量/环境状态 -> 信号]
                # [阶段 2 -> 3: 信号 -> 数据] (Agent.perceive)
                # [阶段 3 -> 4: 数据 -> 信息] (AttentionModule.compute_attention)
                # [阶段 4 -> 7: 信息 -> 知识 -> 智慧 -> 决策] (Agent.decide)
                # [阶段 7 -> 8: 决策 -> 动作] (env.step)
                
                # 执行一步
                next_state, reward, done = agent.step(state)
                state = next_state
                steps += 1
                
                # 检查结束条件
                if done:
                    if isinstance(agent, EnergyAgent) and agent.is_energy_exhausted():
                        is_success = False
                        energy_exhausted_flag = True
                    elif state == env.target_pos:
                        is_success = True
                    else:
                        is_success = False
            
            # 检查是否超时
            if not done and steps >= MAX_STEPS_PER_EPISODE:
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
                energy_left=energy_left,
                energy_exhausted=energy_exhausted_flag
            )
            
            # 定期打印进度
            if (episode + 1) % (NUM_EPISODES // 10) == 0 or episode == 0:
                print(f"  完成 {episode + 1}/{NUM_EPISODES} 回合")
        
        print(f"智能体 {agent.name} 测试完成")
    
    # 计算总耗时
    total_time = time.time() - start_time
    print(f"\n实验完成 - 总耗时: {total_time:.2f} 秒")
    
    # 汇总并显示结果
    summary_df = metrics_recorder.summarize(plot=matplotlib_available)
    
    # 返回汇总数据，以便进一步分析
    return summary_df

# 主函数
if __name__ == "__main__":
    # 运行实验
    results = run_experiment()
    print("\n实验结束!") 