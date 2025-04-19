from typing import Dict, Any, List, Tuple
import numpy as np
import random
from abc import ABC, abstractmethod

from miniexp.env import BaseEnvironment, SimpleEnvironment, MediumEnvironment, AdvancedEnvironment
from miniexp.modules.attention import AttentionModule
from miniexp.modules.information import InformationModule
from miniexp.modules.knowledge import KnowledgeModule
from miniexp.modules.wisdom import WisdomModule
from miniexp.modules.decision import DecisionModule
from miniexp.modules.action import ActionModule
from miniexp.modules.energy import EnergyModule

class BaseAgent(ABC):
    """
    智能体基类
    实现八阶段闭环的基本框架
    """
    
    def __init__(self, env: BaseEnvironment, name: str = "Agent"):
        """
        初始化智能体
        
        Args:
            env: 环境实例
            name: 智能体名称
        """
        self.env = env
        self.name = name
        
        # 初始化各个模块
        self.attention = AttentionModule()
        self.information = InformationModule()
        self.knowledge = KnowledgeModule()
        self.wisdom = WisdomModule()
        self.decision = DecisionModule()
        self.action = ActionModule()
        
    @abstractmethod
    def act(self, state: Dict[str, Any]) -> int:
        """
        执行动作
        
        Args:
            state: 环境状态
            
        Returns:
            int: 动作编号
        """
        pass
        
    def reset(self):
        """
        重置智能体状态
        """
        self.attention.reset()
        self.information.reset()
        self.knowledge.reset()
        self.wisdom.reset()
        self.decision.reset()
        self.action.reset()
        
class BaselineAgent(BaseAgent):
    """
    基线智能体
    实现基本的八阶段闭环
    """
    
    def __init__(self, env: BaseEnvironment, name: str = "BaselineAgent"):
        """
        初始化基线智能体
        
        Args:
            env: 环境实例
            name: 智能体名称
        """
        super().__init__(env, name)
        
        # 配置注意力参数
        self.attention.perception_capacity = 20
        self.attention.decision_capacity = 10
        self.attention.anxiety_influence_rate = 0.0
        self.attention.recovery_rate = 0.2
        
    def act(self, state: Dict[str, Any]) -> int:
        """
        执行动作
        
        Args:
            state: 环境状态
            
        Returns:
            int: 动作编号
        """
        # 1. 获取观察
        observation = self.env.get_observation(state['agent_pos'])
        
        # 2. 计算注意力
        attention = self.attention.compute_attention(observation)
        
        # 3. 处理信息
        information = self.information.process(observation, attention)
        
        # 4. 更新知识
        self.knowledge.update(information)
        
        # 5. 生成智慧
        wisdom = self.wisdom.generate(self.knowledge)
        
        # 6. 做出决策
        action = self.decision.make_decision(wisdom)
        
        # 7. 执行动作
        return self.action.execute(action)
        
class EnergyAgent(BaseAgent):
    """
    能量管理智能体
    在基线智能体的基础上添加能量管理
    """
    
    def __init__(self, env: BaseEnvironment, init_energy: float = 100.0, 
                 threshold: float = 20.0, name: str = "EnergyAgent"):
        """
        初始化能量管理智能体
        
        Args:
            env: 环境实例
            init_energy: 初始能量
            threshold: 能量阈值
            name: 智能体名称
        """
        super().__init__(env, name)
        
        # 初始化能量模块
        self.energy = EnergyModule(init_energy, threshold)
        
        # 配置注意力参数
        self.attention.perception_capacity = 20
        self.attention.decision_capacity = 10
        self.attention.anxiety_influence_rate = 0.5
        self.attention.recovery_rate = 0.2
        
    def act(self, state: Dict[str, Any]) -> int:
        """
        执行动作
        
        Args:
            state: 环境状态
            
        Returns:
            int: 动作编号
        """
        # 1. 更新能量状态
        self.energy.update()
        
        # 2. 获取观察
        observation = self.env.get_observation(state['agent_pos'])
        
        # 3. 计算注意力（受能量状态影响）
        anxiety = self.energy.get_anxiety()
        attention = self.attention.compute_attention(observation, anxiety)
        
        # 4. 处理信息
        information = self.information.process(observation, attention)
        
        # 5. 更新知识
        self.knowledge.update(information)
        
        # 6. 生成智慧
        wisdom = self.wisdom.generate(self.knowledge)
        
        # 7. 做出决策
        action = self.decision.make_decision(wisdom)
        
        # 8. 执行动作
        action_id = self.action.execute(action)
        
        # 9. 消耗能量
        self.energy.consume(1.0)
        
        return action_id
        
    def get_remaining_energy(self) -> float:
        """
        获取剩余能量
        
        Returns:
            float: 剩余能量
        """
        return self.energy.get_remaining()
        
    def is_energy_exhausted(self) -> bool:
        """
        检查能量是否耗尽
        
        Returns:
            bool: 是否耗尽
        """
        return self.energy.is_exhausted()
        
    def reset(self):
        """
        重置智能体状态
        """
        super().reset()
        self.energy.reset()

# 测试代码
if __name__ == "__main__":
    # 创建环境和两种智能体
    env = SimpleEnvironment(width=5, height=5, target_pos=(4, 4), start_pos=(0, 0))
    
    # 测试基线智能体
    print("\n测试基线智能体:")
    baseline_agent = BaselineAgent(env)
    state = env.reset()
    env.render()
    
    done = False
    while not done:
        next_state, reward, done = baseline_agent.act(state), 0.0, False
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
        next_state, reward, done = energy_agent.act(state), 0.0, False
        state = next_state
        env.render()
        print(f"奖励: {reward}, 完成: {done}, 能量: {energy_agent.get_remaining_energy():.1f}") 