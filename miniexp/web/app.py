#!/usr/bin/env python
"""
八阶段闭环与生存焦虑驱动注意力实验 - Web可视化
基于Flask和Socket.IO的Web应用，用于可视化和监控实验
"""
import os
import sys
import time
import json
import threading
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# 确保能导入miniexp包
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# 导入实验相关模块
from miniexp.env import GridWorld
from miniexp.agent import BaselineAgent, EnergyAgent
from miniexp.metrics import MetricsRecorder

# 创建Flask应用
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# 配置Socket.IO，启用CORS，并明确指定传输方式
socketio = SocketIO(
    app, 
    cors_allowed_origins="*", 
    async_mode='eventlet',  # 使用eventlet作为异步模式
    logger=True,            # 启用日志记录
    engineio_logger=True    # 启用引擎日志记录
)

# 保存当前实验状态的全局变量
experiment_status = {
    'running': False,
    'current_agent': None,
    'episode': 0,
    'step': 0,
    'total_episodes': 0,
    'grid_width': 10,
    'grid_height': 10,
    'env': None,
    'agents': {},
    'metrics': None,
    'current_state': None,
    'target_pos': None,
    'thread': None
}

# 默认实验配置
default_config = {
    'grid_width': 10,
    'grid_height': 10,
    'episodes': 20,
    'max_steps': 300,
    'init_energy': 30,
    'energy_threshold': 6,
    'display_interval': 0.3  # 每步延迟秒数，用于可视化
}

# 路由：主页
@app.route('/')
def index():
    return render_template('index.html', config=default_config)

# API：开始新实验
@app.route('/api/start_experiment', methods=['POST'])
def start_experiment():
    global experiment_status
    
    # 如果已经有实验在运行，先停止
    if experiment_status['running']:
        stop_experiment()
    
    # 获取配置参数
    config = request.json or default_config
    grid_width = int(config.get('grid_width', default_config['grid_width']))
    grid_height = int(config.get('grid_height', default_config['grid_height']))
    episodes = int(config.get('episodes', default_config['episodes']))
    max_steps = int(config.get('max_steps', default_config['max_steps']))
    init_energy = float(config.get('init_energy', default_config['init_energy']))
    energy_threshold = float(config.get('energy_threshold', default_config['energy_threshold']))
    display_interval = float(config.get('display_interval', default_config['display_interval']))
    
    # 设置目标位置为右下角
    target_pos = (grid_height - 1, grid_width - 1)
    # 设置起始位置为左上角
    start_pos = (0, 0)
    
    # 创建环境和智能体
    env = GridWorld(width=grid_width, height=grid_height, 
                    target_pos=target_pos, start_pos=start_pos)
    
    baseline_agent = BaselineAgent(env, name="BaselineAgent")
    energy_agent = EnergyAgent(env, 
                              init_energy=init_energy, 
                              threshold=energy_threshold, 
                              name="EnergyAgent")
    
    # 创建指标记录器
    metrics_recorder = MetricsRecorder()
    
    # 更新实验状态
    experiment_status['running'] = True
    experiment_status['episode'] = 0
    experiment_status['step'] = 0
    experiment_status['total_episodes'] = episodes * 2  # 两个智能体
    experiment_status['grid_width'] = grid_width
    experiment_status['grid_height'] = grid_height
    experiment_status['env'] = env
    experiment_status['agents'] = {
        'BaselineAgent': baseline_agent,
        'EnergyAgent': energy_agent
    }
    experiment_status['metrics'] = metrics_recorder
    experiment_status['current_state'] = start_pos
    experiment_status['target_pos'] = target_pos
    
    # 在新线程中运行实验
    experiment_status['thread'] = threading.Thread(
        target=run_experiment_thread, 
        args=(episodes, max_steps, display_interval)
    )
    experiment_status['thread'].daemon = True
    experiment_status['thread'].start()
    
    return jsonify({
        'status': 'success',
        'message': '实验已开始',
        'config': {
            'grid_width': grid_width,
            'grid_height': grid_height,
            'episodes': episodes,
            'max_steps': max_steps,
            'init_energy': init_energy,
            'energy_threshold': energy_threshold,
            'target_pos': target_pos,
            'start_pos': start_pos
        }
    })

# API：停止实验
@app.route('/api/stop_experiment', methods=['POST'])
def stop_experiment():
    global experiment_status
    experiment_status['running'] = False
    
    # 等待线程结束
    if experiment_status['thread'] and experiment_status['thread'].is_alive():
        experiment_status['thread'].join(timeout=2.0)
    
    return jsonify({
        'status': 'success',
        'message': '实验已停止'
    })

# API：获取实验状态
@app.route('/api/experiment_status', methods=['GET'])
def get_experiment_status():
    return jsonify({
        'running': experiment_status['running'],
        'current_agent': experiment_status['current_agent'],
        'episode': experiment_status['episode'],
        'step': experiment_status['step'],
        'total_episodes': experiment_status['total_episodes']
    })

# API：获取结果汇总
@app.route('/api/experiment_results', methods=['GET'])
def get_experiment_results():
    if experiment_status['metrics'] is None:
        return jsonify({'status': 'error', 'message': '没有可用的实验结果'})
    
    # 将DataFrame转换为JSON
    summary_df = experiment_status['metrics'].summarize(plot=False)
    if summary_df is None:
        return jsonify({'status': 'error', 'message': '没有可用的实验结果'})
    
    results = summary_df.to_dict('index')
    
    return jsonify({
        'status': 'success',
        'results': results
    })

# Socket.IO：当客户端连接时
@socketio.on('connect')
def handle_connect():
    print('客户端已连接')
    emit('server_message', {'message': '已连接到服务器'})

# Socket.IO：当客户端断开连接时
@socketio.on('disconnect')
def handle_disconnect():
    print('客户端已断开连接')

# 实验线程函数
def run_experiment_thread(episodes, max_steps, display_interval):
    global experiment_status
    
    try:
        env = experiment_status['env']
        agents = experiment_status['agents']
        metrics_recorder = experiment_status['metrics']
        
        # 为每个智能体运行实验
        overall_episode = 0
        for agent_name, agent in agents.items():
            experiment_status['current_agent'] = agent_name
            print(f"开始运行智能体: {agent_name}")
            
            for episode in range(episodes):
                if not experiment_status['running']:
                    break
                
                experiment_status['episode'] = overall_episode + 1
                overall_episode += 1
                print(f"开始回合 {episode+1}/{episodes} - 智能体: {agent_name}")
                
                # 重置环境和智能体
                state = env.reset()
                agent.reset()
                experiment_status['current_state'] = state
                experiment_status['step'] = 0
                
                # 发送初始状态
                emit_state_update(agent_name, state, 0, 0.0 if isinstance(agent, EnergyAgent) else None, 0.0 if isinstance(agent, EnergyAgent) else None)
                
                # 回合状态变量
                done = False
                steps = 0
                is_success = False
                energy_exhausted_flag = False
                
                # 运行单个回合
                while not done and steps < max_steps and experiment_status['running']:
                    try:
                        experiment_status['step'] = steps + 1
                        
                        # 执行一步
                        next_state, reward, done = agent.step(state)
                        
                        # 获取能量信息（如果是EnergyAgent）
                        energy_left = None
                        anxiety = None
                        if isinstance(agent, EnergyAgent):
                            energy_left = agent.get_remaining_energy()
                            anxiety = agent.energy_module.get_anxiety()
                        
                        # 更新状态
                        experiment_status['current_state'] = next_state
                        
                        # 发送状态更新
                        emit_state_update(agent_name, next_state, steps + 1, energy_left, anxiety)
                        
                        state = next_state
                        steps += 1
                        
                        # 检查结束条件
                        if done:
                            if isinstance(agent, EnergyAgent) and agent.is_energy_exhausted():
                                is_success = False
                                energy_exhausted_flag = True
                                emit_message(f"{agent_name} 能量耗尽，失败于步骤 {steps}")
                            elif state == env.target_pos:
                                is_success = True
                                emit_message(f"{agent_name} 成功到达目标，步数: {steps}")
                            else:
                                is_success = False
                        
                        # 短暂延迟，以便于可视化
                        time.sleep(display_interval)
                    except Exception as e:
                        print(f"步骤执行出错: {e}")
                        time.sleep(display_interval)
                
                # 检查是否超时
                if not done and steps >= max_steps:
                    is_success = False
                    emit_message(f"{agent_name} 超时，步数: {steps}")
                
                # 记录实验结果
                energy_left = np.nan
                if isinstance(agent, EnergyAgent):
                    energy_left = agent.get_remaining_energy()
                    
                metrics_recorder.record(
                    agent_name=agent_name,
                    success=is_success,
                    steps=steps,
                    energy_left=energy_left,
                    energy_exhausted=energy_exhausted_flag
                )
                
                print(f"回合 {episode+1} 完成 - 智能体: {agent_name}, 成功: {is_success}, 步数: {steps}")
        
        # 实验完成，发送结果
        if experiment_status['running']:
            experiment_status['running'] = False
            summary_df = metrics_recorder.summarize(plot=False)
            results = summary_df.to_dict('index') if summary_df is not None else {}
            
            socketio.emit('experiment_complete', {
                'message': '实验已完成',
                'results': results
            })
            print("实验完成，结果已发送")
    except Exception as e:
        print(f"实验线程发生错误: {e}")
        experiment_status['running'] = False
        socketio.emit('server_message', {'message': f'实验出错: {str(e)}'})

# 发送状态更新到客户端
def emit_state_update(agent_name, state, step, energy=None, anxiety=None):
    data = {
        'agent': agent_name,
        'state': state,
        'step': step,
        'grid_width': experiment_status['grid_width'],
        'grid_height': experiment_status['grid_height'],
        'target_pos': experiment_status['target_pos']
    }
    
    if energy is not None:
        data['energy'] = energy
    if anxiety is not None:
        data['anxiety'] = anxiety
    
    # 添加日志输出，帮助调试
    print(f"发送状态更新: agent={agent_name}, state={state}, step={step}")
    
    # 使用带命名空间的emit，确保消息发送到所有客户端
    socketio.emit('state_update', data, namespace='/', broadcast=True)

# 发送消息到客户端
def emit_message(message):
    socketio.emit('server_message', {'message': message})

# 主函数
if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
    
# 修复网格世界实时更新问题 