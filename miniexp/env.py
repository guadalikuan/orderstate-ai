import numpy as np
from typing import Tuple

class GridWorld:
    """
    简单的网格世界环境。
    状态表示：(行, 列)
    动作空间：0: 上, 1: 下, 2: 左, 3: 右
    """
    def __init__(self, width: int, height: int, target_pos: Tuple[int, int], start_pos: Tuple[int, int] = (0, 0)):
        """
        初始化环境。

        Args:
            width (int): 网格宽度。
            height (int): 网格高度。
            target_pos (Tuple[int, int]): 目标位置 (行, 列)。
            start_pos (Tuple[int, int]): 起始位置 (行, 列)，默认为(0,0)。
        """
        if not (0 <= target_pos[0] < height and 0 <= target_pos[1] < width):
            raise ValueError("目标位置超出边界")
        if not (0 <= start_pos[0] < height and 0 <= start_pos[1] < width):
            raise ValueError("起始位置超出边界")

        self.width = width
        self.height = height
        self.target_pos = target_pos
        self.start_pos = start_pos
        self.agent_pos = self.start_pos
        self.action_map = {
            0: (-1, 0),  # 上
            1: (1, 0),   # 下
            2: (0, -1),  # 左
            3: (0, 1)    # 右
        }
        print(f"环境初始化：大小=({width}x{height}), 起点={start_pos}, 终点={target_pos}")

    def reset(self) -> Tuple[int, int]:
        """
        重置环境到初始状态。

        Returns:
            Tuple[int, int]: 初始状态 (智能体位置)。
        """
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一个动作。

        Args:
            action (int): 要执行的动作 (0-3)。

        Returns:
            Tuple[Tuple[int, int], float, bool]: (下一个状态, 奖励, 是否结束)。
        """
        if action not in self.action_map:
            raise ValueError(f"无效动作: {action}")

        # [阶段 8 -> 1: 动作 -> (潜在的)能量消耗(在Agent中处理)]
        # [阶段 1 -> 2: (环境变化) -> 信号]
        dr, dc = self.action_map[action]
        next_r, next_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # 检查边界
        if not (0 <= next_r < self.height and 0 <= next_c < self.width):
            # 撞墙，状态不变，小惩罚
            next_state = self.agent_pos
            reward = -0.1
            done = False
        else:
            # 移动
            self.agent_pos = (next_r, next_c)
            next_state = self.agent_pos
            if self.agent_pos == self.target_pos:
                # 到达目标
                reward = 1.0
                done = True
            else:
                # 普通移动
                reward = -0.01  # 轻微的移动成本
                done = False

        return next_state, reward, done

    def render(self) -> None:
        """
        打印小地图，显示当前环境状态。
        """
        print("-" * (self.width * 2 + 3))
        for r in range(self.height):
            print("|", end=" ")
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    print("A", end=" ")
                elif (r, c) == self.target_pos:
                    print("T", end=" ")
                else:
                    print(".", end=" ")
            print("|")
        print("-" * (self.width * 2 + 3))

# 测试代码
if __name__ == '__main__':
    env = GridWorld(width=5, height=4, target_pos=(3, 4))
    state = env.reset()
    env.render()
    
    # 测试移动
    state, reward, done = env.step(3)  # 右
    env.render()
    print(f"状态: {state}, 奖励: {reward}, 是否结束: {done}") 