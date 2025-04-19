import numpy as np
from typing import Tuple

class GridWorld:
    """
    简单的网格世界环境。
    这是实验的基础环境，提供了一个二维网格，智能体需要从起始位置移动到目标位置。
    
    状态表示：(行, 列) - 使用二元组表示智能体在网格中的位置
    动作空间：0: 上, 1: 下, 2: 左, 3: 右 - 四个基本移动方向
    
    环境特点：
    - 边界限制：智能体不能移出网格边界
    - 奖励设置：到达目标得到正奖励，普通移动有小成本，撞墙有惩罚
    - 终止条件：智能体到达目标位置时任务完成
    """
    def __init__(self, width: int, height: int, target_pos: Tuple[int, int], start_pos: Tuple[int, int] = (0, 0)):
        """
        初始化网格世界环境。

        Args:
            width (int): 网格宽度，定义网格的列数。
            height (int): 网格高度，定义网格的行数。
            target_pos (Tuple[int, int]): 目标位置坐标 (行, 列)，智能体需要到达的终点。
            start_pos (Tuple[int, int], optional): 起始位置坐标 (行, 列)，智能体的起点，默认为(0,0)，即左上角。
        
        Raises:
            ValueError: 如果目标位置或起始位置超出网格边界，则抛出错误。
        """
        # 验证目标位置是否在网格范围内
        if not (0 <= target_pos[0] < height and 0 <= target_pos[1] < width):
            raise ValueError("目标位置超出边界")
        # 验证起始位置是否在网格范围内
        if not (0 <= start_pos[0] < height and 0 <= start_pos[1] < width):
            raise ValueError("起始位置超出边界")

        # 初始化环境参数
        self.width = width  # 网格宽度
        self.height = height  # 网格高度
        self.target_pos = target_pos  # 目标位置
        self.start_pos = start_pos  # 起始位置
        self.agent_pos = self.start_pos  # 当前智能体位置，初始为起始位置
        
        # 动作映射字典，将动作索引映射到实际的方向变化 (行变化, 列变化)
        self.action_map = {
            0: (-1, 0),  # 上：行减1，列不变
            1: (1, 0),   # 下：行加1，列不变
            2: (0, -1),  # 左：行不变，列减1
            3: (0, 1)    # 右：行不变，列加1
        }
        print(f"环境初始化：大小=({width}x{height}), 起点={start_pos}, 终点={target_pos}")

    def reset(self) -> Tuple[int, int]:
        """
        重置环境到初始状态，将智能体放回起始位置。
        每次新的回合开始时调用此方法。

        Returns:
            Tuple[int, int]: 初始状态，即智能体的起始位置坐标 (行, 列)。
        """
        self.agent_pos = self.start_pos  # 重置智能体位置到起始位置
        return self.agent_pos  # 返回初始状态

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """
        执行一个动作，更新环境状态，并返回新状态、奖励和是否终止的信息。
        这是环境交互的核心方法，实现了八阶段循环中的动作到环境变化部分。

        Args:
            action (int): 要执行的动作索引 (0-3)，分别对应上、下、左、右。

        Returns:
            Tuple[Tuple[int, int], float, bool]: 一个三元组，包含：
                - 下一个状态：智能体新的位置坐标 (行, 列)
                - 奖励：执行动作后获得的奖励值
                - 是否结束：如果到达目标位置则为True，否则为False

        Raises:
            ValueError: 如果动作索引不在有效范围内，则抛出错误。
        """
        # 检查动作是否有效
        if action not in self.action_map:
            raise ValueError(f"无效动作: {action}")

        # [阶段 8 -> 1: 动作 -> (潜在的)能量消耗(在Agent中处理)]
        # [阶段 1 -> 2: (环境变化) -> 信号]
        
        # 获取动作对应的方向变化
        dr, dc = self.action_map[action]
        # 计算执行动作后的新位置
        next_r, next_c = self.agent_pos[0] + dr, self.agent_pos[1] + dc

        # 检查是否超出网格边界
        if not (0 <= next_r < self.height and 0 <= next_c < self.width):
            # 撞墙情况：智能体位置不变，给予小惩罚
            next_state = self.agent_pos  # 状态不变
            reward = -0.1  # 撞墙惩罚
            done = False  # 游戏未结束
        else:
            # 正常移动情况：更新智能体位置
            self.agent_pos = (next_r, next_c)  # 更新位置
            next_state = self.agent_pos  # 新状态
            
            if self.agent_pos == self.target_pos:
                # 到达目标情况：给予正奖励，回合结束
                reward = 1.0  # 到达目标的正奖励
                done = True  # 游戏结束
            else:
                # 普通移动情况：给予小成本，继续游戏
                reward = -0.01  # 移动成本，鼓励智能体尽快到达目标
                done = False  # 游戏未结束

        return next_state, reward, done

    def render(self) -> None:
        """
        打印小地图，直观显示当前环境状态。
        在控制台上显示网格世界的可视化表示，便于调试和观察。
        
        地图表示：
        - 'A': 表示智能体当前位置
        - 'T': 表示目标位置
        - '.': 表示空白区域
        - 边框用'-'和'|'表示
        """
        # 打印上边框
        print("-" * (self.width * 2 + 3))
        
        # 打印网格内容
        for r in range(self.height):
            print("|", end=" ")  # 左边框
            for c in range(self.width):
                if (r, c) == self.agent_pos:
                    print("A", end=" ")  # 智能体位置
                elif (r, c) == self.target_pos:
                    print("T", end=" ")  # 目标位置
                else:
                    print(".", end=" ")  # 空白区域
            print("|")  # 右边框
        
        # 打印下边框
        print("-" * (self.width * 2 + 3))

# 测试代码 - 如果直接运行此文件，将执行以下示例
if __name__ == '__main__':
    # 创建一个5x4的网格环境，目标位置在(3,4)
    env = GridWorld(width=5, height=4, target_pos=(3, 4))
    # 重置环境，获取初始状态
    state = env.reset()
    # 显示初始环境
    env.render()
    
    # 测试移动 - 向右移动一步
    state, reward, done = env.step(3)  # 3表示向右移动
    # 显示移动后的环境
    env.render()
    # 打印执行结果
    print(f"状态: {state}, 奖励: {reward}, 是否结束: {done}") 