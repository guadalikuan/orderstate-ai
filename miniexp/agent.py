import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, List, Optional

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

class StateTracker:
    """
    状态跟踪器，用于记录八阶段状态
    """
    
    def __init__(self):
        """
        初始化状态跟踪器
        """
        # 默认使用全局OrderStateManager
        from miniexp.orderstate import order_state_manager
        self.order_state_manager = order_state_manager
    
    def update_state(self, stage: str, value: Any):
        """
        更新指定阶段的状态
        
        Args:
            stage: 阶段名称
            value: 状态值
        """
        self.order_state_manager.update_state(stage, value)
    
    def complete_cycle(self):
        """
        完成一个周期
        """
        # 在OrderStateManager中此功能已自动处理
        pass
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前状态
        
        Returns:
            Dict: 当前状态
        """
        return self.order_state_manager.get_current_state()
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        Args:
            limit: 最大记录数，默认为10
            
        Returns:
            List[Dict]: 历史记录
        """
        return self.order_state_manager.get_history(limit)

class BaseAgent(ABC):
    """
    智能体基类，定义了智能体的基本接口。
    """
    
    def __init__(self, env: GridWorld, name: str = "BaseAgent"):
        """
        初始化智能体。
        
        Args:
            env (GridWorld): 环境实例。
            name (str): 智能体名称。
        """
        self.env = env
        self.name = name
        self.state_tracker = StateTracker()
    
    @abstractmethod
    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，提取特征。
        
        Args:
            state (Tuple[int, int]): 环境状态，通常是智能体的位置坐标。
            
        Returns:
            np.ndarray: 提取的特征向量。
        """
        pass
    
    @abstractmethod
    def decide(self, features: np.ndarray) -> int:
        """
        根据特征向量做出决策。
        
        Args:
            features (np.ndarray): 特征向量。
            
        Returns:
            int: 决策的动作。
        """
        pass
    
    @abstractmethod
    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互。
        
        Args:
            state (Tuple[int, int]): 当前状态。
            
        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        pass
    
    @abstractmethod
    def reset(self):
        """
        重置智能体的状态。
        """
        pass

class BaselineAgent(BaseAgent):
    """
    基线智能体，简单地使用注意力机制而不考虑能量管理。
    """
    
    def __init__(self, env: GridWorld, name: str = "BaselineAgent"):
        """
        初始化基线智能体。
        
        Args:
            env (GridWorld): 环境实例。
            name (str): 智能体名称。
        """
        super().__init__(env, name)
        
        # 使用注意力模块
        from miniexp.modules.attention import AttentionModule
        self.attention = AttentionModule()
        
        # 创建各阶段模块
        self.signal_module = SignalModule()
        self.data_module = DataModule()
        self.information_module = InformationModule()
        self.knowledge_module = KnowledgeModule()
        self.wisdom_module = WisdomModule()
        self.decision_module = DecisionModule()
        self.action_module = ActionModule()
    
    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，提取特征。
        
        Args:
            state (Tuple[int, int]): 环境状态，通常是智能体的位置坐标。
            
        Returns:
            np.ndarray: 提取的特征向量。
        """
        # [阶段 1->2: 能量/环境状态 -> 信号]
        # 基线智能体不使用能量阶段，直接从信号开始
        signal_data = self.signal_module.process_environment_signal(
            env_state=state,
            target_pos=self.env.target_pos,
            energy_level=100.0  # 固定能量水平
        )
        self.state_tracker.update_state('signal', signal_data)
        
        # [阶段 2->3: 信号 -> 数据]
        features = self.data_module.extract_features(
            signal_data=signal_data,
            grid_size=(self.env.height, self.env.width)
        )
        self.state_tracker.update_state('data', features)
        
        return features
    
    def decide(self, features: np.ndarray) -> int:
        """
        根据特征向量做出决策。
        
        Args:
            features (np.ndarray): 特征向量。
            
        Returns:
            int: 决策的动作。
        """
        # [阶段 3->4: 数据 -> 信息]
        # 使用注意力机制处理特征，但不考虑焦虑影响
        attention_state = self.attention.update(anxiety=0.0, reward=0.0)
        weighted_features = self.attention.filter_perception(features, self.env.agent_pos)
        
        information_data = self.information_module.process_data(
            features=features, 
            anxiety=0.0,  # 基线智能体没有焦虑
            raw_data={"attention_state": attention_state}
        )
        self.state_tracker.update_state('information', information_data)
        
        # [阶段 4->5: 信息 -> 知识]
        knowledge_data = self.knowledge_module.process_information(
            information_data=information_data,
            current_position=self.env.agent_pos,
            target_position=self.env.target_pos,
            grid_size=(self.env.height, self.env.width)
        )
        self.state_tracker.update_state('knowledge', knowledge_data)
        
        # [阶段 5->6: 知识 -> 智慧]
        wisdom_data = self.wisdom_module.process_knowledge(
            knowledge_data=knowledge_data,
            anxiety=0.0,
            energy_level=100.0  # 固定能量水平
        )
        self.state_tracker.update_state('wisdom', wisdom_data)
        
        # [阶段 6->7: 智慧 -> 决策]
        action = self.decision_module.make_decision(wisdom_data)
        self.state_tracker.update_state('decision', action)
        
        return action
    
    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互。
        
        Args:
            state (Tuple[int, int]): 当前状态。
            
        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        # [阶段 2->3->4: 信号->数据->信息] + [阶段 4->5->6->7: 信息->知识->智慧->决策]
        features = self.perceive(state)
        action = self.decide(features)
        
        # [阶段 7->8: 决策 -> 动作]
        action_result = self.action_module.execute_action(
            action=action,
            env=self.env,
            anxiety=0.0  # 基线智能体没有焦虑
        )
        self.state_tracker.update_state('action', action_result)
        
        # 从执行结果中获取下一状态、奖励和是否结束
        next_state = action_result["next_state"]
        reward = action_result["reward"]
        done = action_result["done"]
        
        # 基线智能体没有能量消耗，所以不用处理 action->energy 阶段
        
        return next_state, reward, done
    
    def reset(self):
        """
        重置智能体的状态。
        """
        self.attention.reset()
        
        # 重置各阶段模块
        self.signal_module.reset()
        self.data_module.reset()
        self.information_module.reset()
        self.knowledge_module.reset()
        self.wisdom_module.reset()
        self.decision_module.reset()
        self.action_module.reset()

class EnergyAgent(BaseAgent):
    """
    能量智能体，考虑能量管理，生存焦虑会影响注意力分配。
    """
    
    def __init__(self, env: GridWorld, init_energy: float, threshold: float, name: str = "EnergyAgent"):
        """
        初始化能量智能体。
        
        Args:
            env (GridWorld): 环境实例。
            init_energy (float): 初始能量。
            threshold (float): 能量阈值，低于此值会产生焦虑。
            name (str): 智能体名称。
        """
        super().__init__(env, name)
        
        # 初始化能量模块
        self.energy_module = EnergyModule(
            initial_energy=init_energy,
            max_energy=init_energy,
            min_energy=0.0,
            energy_decay_rate=0.5,
            energy_recovery_rate=2.0,
            anxiety_threshold=threshold
        )
        
        # 初始化注意力模块
        from miniexp.modules.attention import AttentionModule
        self.attention = AttentionModule()
        
        # 创建各阶段模块
        self.signal_module = SignalModule()
        self.data_module = DataModule()
        self.information_module = InformationModule()
        self.knowledge_module = KnowledgeModule()
        self.wisdom_module = WisdomModule()
        self.decision_module = DecisionModule()
        self.action_module = ActionModule()
    
    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        感知环境状态，提取特征。考虑能量的影响。
        
        Args:
            state (Tuple[int, int]): 环境状态，通常是智能体的位置坐标。
            
        Returns:
            np.ndarray: 提取的特征向量。
        """
        # 首先更新能量状态，以便传递到信号模块
        # [阶段 8->1: 动作 -> 能量]
        energy_state = self.energy_module.get_metrics()
        self.state_tracker.update_state('energy', energy_state["current_energy"])
        
        # [阶段 1->2: 能量/环境状态 -> 信号]
        signal_data = self.signal_module.process_environment_signal(
            env_state=state,
            target_pos=self.env.target_pos,
            energy_level=energy_state["current_energy"]
        )
        self.state_tracker.update_state('signal', signal_data)
        
        # [阶段 2->3: 信号 -> 数据]
        # 提取特征，并将能量相关特征添加到特征向量中
        features = self.data_module.extract_features(
            signal_data=signal_data,
            grid_size=(self.env.height, self.env.width)
        )
        
        # 将能量模块的状态特征合并到感知特征中
        energy_features = self.energy_module.get_state_features()
        
        # 合并特征（这里简单地将两个特征向量连接）
        combined_features = np.hstack([features, energy_features])
        
        self.state_tracker.update_state('data', combined_features)
        
        return combined_features
    
    def decide(self, features: np.ndarray) -> int:
        """
        根据特征向量做出决策。考虑焦虑对注意力的影响。
        
        Args:
            features (np.ndarray): 特征向量。
            
        Returns:
            int: 决策的动作。
        """
        # 获取当前焦虑水平
        anxiety = self.energy_module.anxiety
        
        # [阶段 3->4: 数据 -> 信息]
        # 更新注意力状态，考虑焦虑的影响
        attention_state = self.attention.update(anxiety=anxiety, reward=0.0)
        
        # 使用注意力过滤感知
        weighted_features = self.attention.filter_perception(features, self.env.agent_pos)
        
        information_data = self.information_module.process_data(
            features=features, 
            anxiety=anxiety,
            raw_data={"attention_state": attention_state}
        )
        self.state_tracker.update_state('information', information_data)
        
        # [阶段 4->5: 信息 -> 知识]
        knowledge_data = self.knowledge_module.process_information(
            information_data=information_data,
            current_position=self.env.agent_pos,
            target_position=self.env.target_pos,
            grid_size=(self.env.height, self.env.width)
        )
        self.state_tracker.update_state('knowledge', knowledge_data)
        
        # [阶段 5->6: 知识 -> 智慧]
        wisdom_data = self.wisdom_module.process_knowledge(
            knowledge_data=knowledge_data,
            anxiety=anxiety,
            energy_level=self.energy_module.energy
        )
        self.state_tracker.update_state('wisdom', wisdom_data)
        
        # [阶段 6->7: 智慧 -> 决策]
        action = self.decision_module.make_decision(wisdom_data)
        self.state_tracker.update_state('decision', action)
        
        return action
    
    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步交互。考虑能量消耗和焦虑影响。
        
        Args:
            state (Tuple[int, int]): 当前状态。
            
        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        # [阶段 2->3->4: 信号->数据->信息] + [阶段 4->5->6->7: 信息->知识->智慧->决策]
        features = self.perceive(state)
        action = self.decide(features)
        
        # 获取当前焦虑水平
        anxiety = self.energy_module.anxiety
        
        # [阶段 7->8: 决策 -> 动作]
        action_result = self.action_module.execute_action(
            action=action,
            env=self.env,
            anxiety=anxiety
        )
        self.state_tracker.update_state('action', action_result)
        
        # 从执行结果中获取下一状态、奖励和是否结束
        next_state = action_result["next_state"]
        reward = action_result["reward"]
        done = action_result["done"]
        
        # 更新能量状态，消耗动作成本
        # [阶段 8->1: 动作 -> 能量]
        energy_cost = action_result["energy_cost"]
        energy_update = self.energy_module.update(reward=reward, action_cost=energy_cost)
        
        # 能量耗尽检查
        if self.is_energy_exhausted():
            done = True
        
        return next_state, reward, done
    
    def reset(self):
        """
        重置智能体的状态。
        """
        self.energy_module.reset()
        self.attention.reset()
        
        # 重置各阶段模块
        self.signal_module.reset()
        self.data_module.reset()
        self.information_module.reset()
        self.knowledge_module.reset()
        self.wisdom_module.reset()
        self.decision_module.reset()
        self.action_module.reset()
    
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
            bool: 如果能量耗尽返回True，否则返回False。
        """
        return self.energy_module.energy <= 0

# 测试代码
if __name__ == '__main__':
    # 创建一个简单的网格世界环境
    env = GridWorld(width=5, height=4, target_pos=(3, 4))
    
    # 测试BaselineAgent
    print("测试 BaselineAgent:")
    baseline_agent = BaselineAgent(env)
    state = env.reset()
    done = False
    while not done:
        next_state, reward, done = baseline_agent.step(state)
        print(f"状态: {state} -> {next_state}, 动作: {baseline_agent.decision_module.get_current_decision().get('action')}, 奖励: {reward}")
        state = next_state
    print("完成!")
    
    # 测试EnergyAgent
    print("\n测试 EnergyAgent:")
    energy_agent = EnergyAgent(env, init_energy=10.0, threshold=3.0)
    state = env.reset()
    done = False
    while not done:
        next_state, reward, done = energy_agent.step(state)
        print(f"状态: {state} -> {next_state}, 动作: {energy_agent.decision_module.get_current_decision().get('action')}, 奖励: {reward}, 能量: {energy_agent.get_remaining_energy():.1f}, 焦虑: {energy_agent.energy_module.anxiety:.2f}")
        state = next_state
    print("完成!") 