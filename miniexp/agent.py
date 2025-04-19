import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List

from .env import GridWorld
from .modules.attention import AttentionModule
from .modules.energy import EnergyModule

class BaseAgent(ABC):
    """
    智能体基类，定义了智能体的基本接口。
    实现了感知-决策-行动的基本循环。
    """
    def __init__(self, env: GridWorld, name: str = "BaseAgent"):
        """
        初始化基础智能体。

        Args:
            env (GridWorld): 环境实例。
            name (str, optional): 智能体名称。默认为"BaseAgent"。
        """
        self.env = env
        self.name = name

    @abstractmethod
    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        从状态中提取特征。
        [阶段 2 -> 3: 信号 -> 数据]

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: 提取的特征。
        """
        pass

    @abstractmethod
    def decide(self, features: np.ndarray) -> int:
        """
        根据特征决定采取的动作。
        [阶段 4 -> 7: 信息 -> 知识 -> 智慧 -> 决策]

        Args:
            features (np.ndarray): 特征数组。

        Returns:
            int: 选择的动作。
        """
        pass

    @abstractmethod
    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步：感知-决策-行动。
        [完整的八阶段闭环]

        Args:
            state (Tuple[int, int]): 当前状态。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        pass

    @abstractmethod
    def reset(self):
        """重置智能体的内部状态。"""
        pass


class BaselineAgent(BaseAgent):
    """
    基准智能体，使用注意力机制但不考虑能量。
    """
    def __init__(self, env: GridWorld, name: str = "BaselineAgent"):
        """
        初始化基准智能体。

        Args:
            env (GridWorld): 环境实例。
            name (str, optional): 智能体名称。默认为"BaselineAgent"。
        """
        super().__init__(env, name)
        self.feature_dim = 2  # (dx, dy) 特征
        self.attention_module = AttentionModule(feature_dim=self.feature_dim)

    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        从状态中提取特征：到目标的相对距离。
        [阶段 2 -> 3: 信号 -> 数据]

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: [dx, dy] 到目标的相对距离。
        """
        agent_row, agent_col = state
        target_row, target_col = self.env.target_pos
        
        # 计算相对距离
        dx = target_col - agent_col
        dy = target_row - agent_row
        
        return np.array([dx, dy])

    def decide(self, features: np.ndarray) -> int:
        """
        根据特征和注意力权重决定行动。
        [阶段 4 -> 7: 信息 -> 知识 -> 智慧 -> 决策]

        Args:
            features (np.ndarray): [dx, dy] 特征。

        Returns:
            int: 选择的动作 (0:上, 1:下, 2:左, 3:右)。
        """
        # 使用注意力模块计算每个动作的权重
        # 由于BaselineAgent不考虑焦虑，所以anxiety=0
        attention_weights = self.attention_module.compute_attention(features, anxiety=0.0)
        
        # 选择权重最高的动作
        action = np.argmax(attention_weights)
        return int(action)

    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步完整的感知-决策-行动循环。
        [完整的八阶段闭环，但没有能量管理]

        Args:
            state (Tuple[int, int]): 当前状态。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        # 1. 感知：从环境状态提取特征
        features = self.perceive(state)
        
        # 2. 决策：根据特征决定行动
        action = self.decide(features)
        
        # 3. 行动：在环境中执行动作
        # [阶段 7 -> 8: 决策 -> 动作]
        next_state, reward, done = self.env.step(action)
        
        return next_state, reward, done

    def reset(self):
        """
        重置智能体状态。
        """
        # BaselineAgent没有需要重置的内部状态
        pass


class EnergyAgent(BaseAgent):
    """
    能量智能体，使用注意力机制并考虑能量管理。
    焦虑会影响注意力分配。
    """
    def __init__(self, env: GridWorld, init_energy: float, threshold: float, name: str = "EnergyAgent"):
        """
        初始化能量智能体。

        Args:
            env (GridWorld): 环境实例。
            init_energy (float): 初始能量值。
            threshold (float): 能量阈值，低于此值产生最大焦虑。
            name (str, optional): 智能体名称。默认为"EnergyAgent"。
        """
        super().__init__(env, name)
        self.feature_dim = 2  # (dx, dy) 特征
        self.attention_module = AttentionModule(feature_dim=self.feature_dim)
        self.energy_module = EnergyModule(init_energy=init_energy, threshold=threshold)
        self.energy_exhausted = False

    def perceive(self, state: Tuple[int, int]) -> np.ndarray:
        """
        从状态中提取特征：到目标的相对距离。
        [阶段 2 -> 3: 信号 -> 数据]

        Args:
            state (Tuple[int, int]): 当前状态 (行, 列)。

        Returns:
            np.ndarray: [dx, dy] 到目标的相对距离。
        """
        agent_row, agent_col = state
        target_row, target_col = self.env.target_pos
        
        # 计算相对距离
        dx = target_col - agent_col
        dy = target_row - agent_row
        
        return np.array([dx, dy])

    def decide(self, features: np.ndarray) -> int:
        """
        根据特征、能量状态和生存焦虑决定行动。
        [阶段 4 -> 7: 信息 -> 知识 -> 智慧 -> 决策]

        Args:
            features (np.ndarray): [dx, dy] 特征。

        Returns:
            int: 选择的动作 (0:上, 1:下, 2:左, 3:右)。
        """
        # 获取当前焦虑度
        # [阶段 1 -> (内部): 能量状态转化为焦虑信号]
        anxiety = self.energy_module.get_anxiety()
        
        # --- 生存焦虑驱动Attention ---
        # 使用注意力模块计算每个动作的权重，受焦虑影响
        # [阶段 3 -> 4: 数据 -> 信息] (焦虑影响信息处理)
        attention_weights = self.attention_module.compute_attention(features, anxiety=anxiety)
        
        # 选择权重最高的动作
        action = np.argmax(attention_weights)
        return int(action)

    def step(self, state: Tuple[int, int]) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一步完整的感知-决策-行动循环，同时管理能量。
        [完整的八阶段闭环，包括能量管理]

        Args:
            state (Tuple[int, int]): 当前状态。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        # 首先消耗能量
        # [阶段 8 -> 1: 动作 -> 能量]
        if self.energy_module.consume(amount=1.0):
            # 能量耗尽
            self.energy_exhausted = True
            return state, -1.0, True  # 失败，返回当前状态并结束
        
        # 1. 感知：从环境状态提取特征
        # [阶段 2 -> 3: 信号 -> 数据]
        features = self.perceive(state)
        
        # 2. 决策：根据特征和焦虑度决定行动
        # [阶段 4 -> 7: 信息 -> 知识 -> 智慧 -> 决策]
        action = self.decide(features)
        
        # 3. 行动：在环境中执行动作
        # [阶段 7 -> 8: 决策 -> 动作]
        next_state, reward, done = self.env.step(action)
        
        # 检查是否完成目标
        if done and next_state == self.env.target_pos:
            # 成功到达目标
            pass
        elif self.energy_exhausted:
            # 能量耗尽导致失败
            done = True
            reward = -1.0  # 失败的惩罚
        
        return next_state, reward, done

    def reset(self):
        """
        重置智能体状态，包括能量。
        """
        self.energy_module.reset()
        self.energy_exhausted = False
        
    def get_remaining_energy(self) -> float:
        """
        获取剩余能量。

        Returns:
            float: 剩余能量值。
        """
        return self.energy_module.get_energy()
    
    def is_energy_exhausted(self) -> bool:
        """
        检查能量是否耗尽。

        Returns:
            bool: 能量是否耗尽。
        """
        return self.energy_exhausted

# 测试代码
if __name__ == '__main__':
    # 创建环境
    env = GridWorld(width=5, height=5, target_pos=(4, 4))
    
    # 测试BaselineAgent
    print("\n--- 测试 BaselineAgent ---")
    baseline_agent = BaselineAgent(env)
    state = env.reset()
    done = False
    steps = 0
    
    while not done and steps < 20:
        next_state, reward, done = baseline_agent.step(state)
        print(f"步骤 {steps+1}: 状态 {state} -> 动作 -> 新状态 {next_state}, 奖励 {reward}")
        state = next_state
        steps += 1
        
        if done:
            print(f"到达目标，用了 {steps} 步")
            
    # 测试EnergyAgent
    print("\n--- 测试 EnergyAgent ---")
    energy_agent = EnergyAgent(env, init_energy=15, threshold=5)
    state = env.reset()
    energy_agent.reset()
    done = False
    steps = 0
    
    while not done and steps < 20:
        current_energy = energy_agent.get_remaining_energy()
        next_state, reward, done = energy_agent.step(state)
        print(f"步骤 {steps+1}: 状态 {state} -> 动作 -> 新状态 {next_state}, 奖励 {reward}, 剩余能量 {energy_agent.get_remaining_energy()}")
        state = next_state
        steps += 1
        
        if energy_agent.is_energy_exhausted():
            print(f"能量耗尽，失败于第 {steps} 步")
            break
        elif done:
            print(f"到达目标，用了 {steps} 步，剩余能量 {energy_agent.get_remaining_energy()}") 