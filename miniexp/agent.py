import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional
import time
import copy

from miniexp.env import GridWorld
from miniexp.modules.energy import EnergyModule
from miniexp.modules.signal import SignalModule
from miniexp.modules.data import DataModule
from miniexp.modules.information import InformationModule
from miniexp.modules.knowledge import KnowledgeModule
from miniexp.modules.wisdom import WisdomModule
from miniexp.modules.decision import DecisionModule
from miniexp.modules.action import ActionModule
from miniexp.orderstate import OrderStateManager
from miniexp.modules.attention import AttentionModule

class StateTracker:
    """
    状态追踪器：记录八阶段序态循环的状态变化
    """
    
    def __init__(self):
        # 存储八个阶段的当前状态
        self.states = {
            'energy': {'value': None, 'timestamp': None},
            'signal': {'value': None, 'timestamp': None},
            'data': {'value': None, 'timestamp': None},
            'information': {'value': None, 'timestamp': None},
            'knowledge': {'value': None, 'timestamp': None},
            'wisdom': {'value': None, 'timestamp': None},
            'decision': {'value': None, 'timestamp': None},
            'action': {'value': None, 'timestamp': None}
        }
        
        # 存储历史状态变化
        self.history = []
        
        # 当前活跃阶段
        self.current_stage = None
        
        # 完整循环计数
        self.cycle_count = 0
    
    def reset(self):
        """
        重置状态追踪器到初始状态
        """
        # 重置所有阶段的状态
        for stage in self.states:
            self.states[stage]['value'] = None
            self.states[stage]['timestamp'] = None
        
        # 清空历史记录
        self.history = []
        
        # 重置当前阶段
        self.current_stage = None
        
        # 重置循环计数
        self.cycle_count = 0
    
    def update_state(self, stage: str, value: Any):
        """更新指定阶段的状态"""
        if stage not in self.states:
            raise ValueError(f"未知阶段: {stage}")
            
        self.states[stage]['value'] = value
        self.states[stage]['timestamp'] = time.time()
        self.current_stage = stage
        
        # 记录历史
        self.history.append({
            'stage': stage,
            'value': value,
            'timestamp': time.time()
        })
        
        # 检查是否完成一个循环
        if stage == 'action':
            self.complete_cycle()
            
    def complete_cycle(self):
        """完成一个完整的八阶段循环"""
        # 检查是否所有阶段都有值
        if all(self.states[stage]['value'] is not None for stage in self.states):
            self.cycle_count += 1
            print(f"完成第 {self.cycle_count} 个序态循环")
            
            # 这里可以添加完成循环后的回调或其他处理
    
    def get_current_state(self) -> Dict[str, Any]:
        """获取当前所有阶段的状态"""
        result = {
            'current': {stage: {'value': self.states[stage]['value']} 
                      for stage in self.states},
            'current_stage': self.current_stage,
            'cycle_count': self.cycle_count,
            'status': 'success'
        }
        return result
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取历史状态记录"""
        return self.history[-limit:]

class BaseAgent(ABC):
    """
    基础智能体抽象类。
    定义了所有智能体必须实现的基本接口。
    """
    def __init__(self, env: GridWorld, name: str = "BaseAgent"):
        """
        初始化基础智能体。

        Args:
            env (GridWorld): 环境实例。
            name (str, optional): 智能体名称。
        """
        self.env = env
        self.name = name
        
        # 初始化注意力模块
        self.attention = AttentionModule(
            perception_capacity=20,  # 感知容量
            decision_capacity=10,    # 决策容量
            anxiety_influence_rate=0.0,  # 默认不考虑焦虑
            recovery_rate=0.2
        )
        
    @abstractmethod
    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，返回特征向量。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: 特征向量。
        """
        pass

    @abstractmethod
    def decide(self, features: np.ndarray) -> int:
        """
        基于特征向量决策下一步动作。

        Args:
            features (np.ndarray): 特征向量。

        Returns:
            int: 所选动作。
        """
        pass

    @abstractmethod
    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        pass

    @abstractmethod
    def reset(self):
        """
        重置智能体状态。
        """
        pass

class BaselineAgent(BaseAgent):
    """
    基线智能体实现。使用注意力机制但不考虑能量管理。
    """
    def __init__(self, env: GridWorld, name: str = "BaselineAgent"):
        """
        初始化基线智能体。

        Args:
            env (GridWorld): 环境实例
            name (str, optional): 智能体名称
        """
        super().__init__(env, name)
        
        # 配置注意力模块参数
        self.attention.perception_capacity = 20  # 感知容量
        self.attention.decision_capacity = 10    # 决策容量
        self.attention.anxiety_influence_rate = 0.0  # 基线智能体不考虑焦虑
        self.attention.recovery_rate = 0.2
        
        # 创建状态追踪器
        self.state_tracker = StateTracker()
        
        # 序态循环当前阶段
        self.current_stage = None
        
        print(f"创建基线智能体: {name}")

    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，返回特征向量。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: 特征向量。
        """
        # [阶段 2 -> 3: 信号 -> 数据]
        self.current_stage = 'signal'
        signal_value = {
            'agent_pos': state,
            'target_pos': self.env.target_pos
        }
        self.state_tracker.update_state('signal', signal_value)
        
        # 提取原始特征
        # 1-2: 智能体位置(归一化)
        # 3-4: 目标位置(归一化)
        # 5: 与目标的距离(曼哈顿距离)
        r, c = state
        tr, tc = self.env.target_pos
        
        # 提取原始特征
        raw_features = np.array([
            r / max(1, self.env.height - 1),  # 行归一化
            c / max(1, self.env.width - 1),   # 列归一化
            tr / max(1, self.env.height - 1), # 目标行归一化
            tc / max(1, self.env.width - 1),  # 目标列归一化
            abs(r - tr) + abs(c - tc)        # 曼哈顿距离
        ])
        
        # 记录数据阶段
        self.current_stage = 'data'
        data_value = {'raw_features': raw_features.tolist()}
        self.state_tracker.update_state('data', data_value)
        
        # [阶段 3 -> 4: 数据 -> 信息]
        # 通过注意力模块过滤感知
        features = self.attention.filter_perception(raw_features, state)
        
        # 记录信息阶段
        self.current_stage = 'information'
        info_value = {'filtered_features': features.tolist()}
        self.state_tracker.update_state('information', info_value)
        
        return features

    def decide(self, features: np.ndarray) -> int:
        """
        基于特征向量决策下一步动作。

        Args:
            features (np.ndarray): 特征向量。

        Returns:
            int: 所选动作。
        """
        # [阶段 4 -> 5/6/7: 信息 -> 知识/智慧/决策]
        
        # 根据特征计算每个动作的值
        action_values = np.zeros(4)  # 上、下、左、右
        
        # 当前位置在特征向量中的索引
        r_norm = features[0]  # 归一化后的行
        c_norm = features[1]  # 归一化后的列
        tr_norm = features[2]  # 归一化后的目标行
        tc_norm = features[3]  # 归一化后的目标列
        
        # 简单启发式：向目标方向移动有更高的值
        r_diff = tr_norm - r_norm
        c_diff = tc_norm - c_norm
        
        # 计算动作值
        # 上(0)
        action_values[0] = 0.5 if r_diff < 0 else -0.5
        # 下(1)
        action_values[1] = 0.5 if r_diff > 0 else -0.5
        # 左(2)
        action_values[2] = 0.5 if c_diff < 0 else -0.5
        # 右(3)
        action_values[3] = 0.5 if c_diff > 0 else -0.5
        
        # 记录知识阶段
        self.current_stage = 'knowledge'
        knowledge_value = {'action_values': action_values.tolist()}
        self.state_tracker.update_state('knowledge', knowledge_value)
        
        # 使用注意力过滤决策
        # 基线智能体焦虑度始终为0
        anxiety = 0.0
        filtered_values = self.attention.filter_action_values(action_values, anxiety)
        
        # 记录智慧阶段
        self.current_stage = 'wisdom' 
        wisdom_value = {'filtered_values': filtered_values.tolist()}
        self.state_tracker.update_state('wisdom', wisdom_value)
        
        # 选择值最高的动作
        action = np.argmax(filtered_values)
        
        # 记录决策阶段
        self.current_stage = 'decision'
        decision_value = {'selected_action': int(action)}
        self.state_tracker.update_state('decision', decision_value)
        
        return action

    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        # [完整的八阶段闭环]
        
        # 感知并决策
        features = self.perceive(state)
        action = self.decide(features)
        
        # [阶段 7 -> 8: 决策 -> 动作]
        # 执行选择的动作
        next_state, reward, done = self.env.step(action)
        
        # 记录动作阶段
        self.current_stage = 'action'
        action_value = {
            'action_taken': int(action),
            'next_state': next_state,
            'reward': float(reward)
        }
        self.state_tracker.update_state('action', action_value)
        
        # 更新注意力模块
        self.attention.update(0.0, reward)  # 基线智能体焦虑度始终为0
        
        return next_state, reward, done

    def reset(self):
        """
        重置智能体状态。
        """
        self.attention.reset()
        self.state_tracker.reset()

class EnergyAgent(BaseAgent):
    """
    能量智能体实现。使用注意力机制并考虑能量管理。
    """
    def __init__(self, env: GridWorld, init_energy: float, threshold: float, name: str = "EnergyAgent"):
        """
        初始化能量智能体。

        Args:
            env (GridWorld): 环境实例
            init_energy (float): 初始能量值
            threshold (float): 能量阈值，低于此值会产生焦虑
            name (str, optional): 智能体名称
        """
        super().__init__(env, name)
        
        # 配置注意力模块参数
        self.attention.perception_capacity = 20  # 感知容量
        self.attention.decision_capacity = 10    # 决策容量
        self.attention.anxiety_influence_rate = 1.5  # 焦虑对注意力的影响率
        self.attention.recovery_rate = 0.1
        
        # 创建能量模块
        self.energy_module = EnergyModule(
            initial_energy=init_energy,
            max_energy=init_energy,
            min_energy=0.0,
            energy_decay_rate=0.5,   # 能量衰减率
            energy_recovery_rate=0.2,  # 能量恢复率
            anxiety_threshold=threshold,  # 焦虑阈值
            anxiety_sensitivity=2.0,  # 焦虑敏感度
            anxiety_recovery_rate=0.1  # 焦虑恢复率
        )
        
        # 创建状态追踪器
        self.state_tracker = StateTracker()
        
        # 序态循环当前阶段
        self.current_stage = None
        
        print(f"创建能量智能体: {name}, 初始能量: {init_energy}, 阈值: {threshold}")

    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，返回特征向量。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: 特征向量。
        """
        # [阶段 1 -> 2: 能量 -> 信号]
        # 更新并记录能量阶段
        self.current_stage = 'energy'
        energy_value = {
            'energy': float(self.energy_module.energy),
            'anxiety': float(self.energy_module.anxiety)
        }
        self.state_tracker.update_state('energy', energy_value)
        
        # [阶段 2 -> 3: 信号 -> 数据]
        self.current_stage = 'signal'
        signal_value = {
            'agent_pos': state,
            'target_pos': self.env.target_pos
        }
        self.state_tracker.update_state('signal', signal_value)
        
        # 提取原始特征
        # 1-2: 智能体位置(归一化)
        # 3-4: 目标位置(归一化)
        # 5: 与目标的距离(曼哈顿距离)
        # 6: 当前能量水平(归一化)
        # 7: 当前焦虑水平
        r, c = state
        tr, tc = self.env.target_pos
        
        raw_features = np.array([
            r / max(1, self.env.height - 1),  # 行归一化
            c / max(1, self.env.width - 1),   # 列归一化
            tr / max(1, self.env.height - 1), # 目标行归一化
            tc / max(1, self.env.width - 1),  # 目标列归一化
            abs(r - tr) + abs(c - tc),       # 曼哈顿距离
            self.energy_module.energy / self.energy_module.max_energy,  # 能量归一化
            self.energy_module.anxiety        # 焦虑水平
        ])
        
        # 记录数据阶段
        self.current_stage = 'data'
        data_value = {'raw_features': raw_features.tolist()}
        self.state_tracker.update_state('data', data_value)
        
        # [阶段 3 -> 4: 数据 -> 信息]
        # 通过注意力模块过滤感知，考虑焦虑影响
        features = self.attention.filter_perception(raw_features, state)
        
        # 记录信息阶段
        self.current_stage = 'information'
        info_value = {'filtered_features': features.tolist()}
        self.state_tracker.update_state('information', info_value)
        
        return features

    def decide(self, features: np.ndarray) -> int:
        """
        基于特征向量决策下一步动作，考虑能量因素。

        Args:
            features (np.ndarray): 特征向量。

        Returns:
            int: 所选动作。
        """
        # [阶段 4 -> 5/6/7: 信息 -> 知识/智慧/决策]
        
        # 根据特征计算每个动作的值
        action_values = np.zeros(4)  # 上、下、左、右
        
        # 解析特征
        r_norm = features[0]  # 归一化后的行
        c_norm = features[1]  # 归一化后的列
        tr_norm = features[2]  # 归一化后的目标行
        tc_norm = features[3]  # 归一化后的目标列
        energy_norm = features[5] if len(features) > 5 else 1.0  # 归一化后的能量
        
        # 计算动作值，考虑到目标和能量
        r_diff = tr_norm - r_norm
        c_diff = tc_norm - c_norm
        
        # 基本动作值
        # 上(0)
        action_values[0] = 0.5 if r_diff < 0 else -0.5
        # 下(1)
        action_values[1] = 0.5 if r_diff > 0 else -0.5
        # 左(2)
        action_values[2] = 0.5 if c_diff < 0 else -0.5
        # 右(3)
        action_values[3] = 0.5 if c_diff > 0 else -0.5
        
        # 记录知识阶段
        self.current_stage = 'knowledge'
        knowledge_value = {'action_values': action_values.tolist()}
        self.state_tracker.update_state('knowledge', knowledge_value)
        
        # 应用能量优先级
        survival_priority = self.energy_module.get_survival_priority()
        adjusted_values = self.energy_module.compute_action_priority(action_values)
        
        # 使用注意力过滤决策，考虑焦虑影响
        anxiety = self.energy_module.anxiety
        final_values = self.attention.filter_action_values(adjusted_values, anxiety)
        
        # 记录智慧阶段
        self.current_stage = 'wisdom'
        wisdom_value = {
            'raw_values': action_values.tolist(),
            'energy_adjusted': adjusted_values.tolist(),
            'final_values': final_values.tolist(),
            'survival_priority': float(survival_priority)
        }
        self.state_tracker.update_state('wisdom', wisdom_value)
        
        # 选择值最高的动作
        action = np.argmax(final_values)
        
        # 记录决策阶段
        self.current_stage = 'decision'
        decision_value = {'selected_action': int(action)}
        self.state_tracker.update_state('decision', decision_value)
        
        return action

    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互，考虑能量管理。

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束或能量耗尽)。
        """
        # 检查能量是否耗尽
        if self.is_energy_exhausted():
            return state, -1.0, True
        
        # 感知并决策
        features = self.perceive(state)
        action = self.decide(features)
        
        # [阶段 7 -> 8: 决策 -> 动作]
        # 执行选择的动作，消耗能量
        next_state, reward, done = self.env.step(action)
        
        # 记录动作阶段
        self.current_stage = 'action'
        action_value = {
            'action_taken': int(action),
            'next_state': next_state,
            'reward': float(reward),
            'energy_cost': 1.0  # 基本能量消耗
        }
        self.state_tracker.update_state('action', action_value)
        
        # [阶段 8 -> 1: 动作 -> 能量]
        # 更新能量和焦虑
        self.energy_module.update(reward, action_cost=1.0)
        
        # 更新注意力模块
        self.attention.update(self.energy_module.anxiety, reward)
        
        # 检查是否能量耗尽
        if self.is_energy_exhausted():
            return next_state, -1.0, True
            
        return next_state, reward, done

    def reset(self):
        """
        重置智能体状态。
        """
        self.attention.reset()
        self.energy_module.reset()
        self.state_tracker.reset()

    def get_remaining_energy(self) -> float:
        """
        获取剩余能量。

        Returns:
            float: 剩余能量值。
        """
        return self.energy_module.energy

    def is_energy_exhausted(self) -> bool:
        """
        检查能量是否耗尽。

        Returns:
            bool: 如果能量耗尽则为True。
        """
        return self.energy_module.energy <= 0

# 测试代码
if __name__ == "__main__":
    # 创建环境和两种智能体
    env = GridWorld(width=5, height=5, target_pos=(4, 4), start_pos=(0, 0))
    
    # 测试基线智能体
    print("\n测试基线智能体:")
    baseline_agent = BaselineAgent(env)
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        next_state, reward, done = baseline_agent.step(state)
        state = next_state
        env.render()
        print(f"奖励: {reward}, 完成: {done}")
    
    # 测试能量智能体
    print("\n测试能量智能体:")
    energy_agent = EnergyAgent(env, init_energy=20.0, threshold=5.0)
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        next_state, reward, done = energy_agent.step(state)
        state = next_state
        env.render()
        print(f"奖励: {reward}, 完成: {done}, 能量: {energy_agent.get_remaining_energy():.1f}, 焦虑: {energy_agent.energy_module.anxiety:.2f}") 