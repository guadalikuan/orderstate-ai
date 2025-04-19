import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Any, Optional

class Environment:
    """
    基础环境类。
    提供网格世界，包含食物、障碍物和奖励机制。
    """
    # 动作定义 [上, 下, 左, 右]
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    # 奖励设置
    REWARDS = {
        "food": 5.0,       # 获取食物奖励
        "obstacle": -2.0,   # 障碍物惩罚
        "boundary": -1.0,   # 边界惩罚
        "move": -0.05,      # 移动成本
        "revisit": -0.1     # 重复访问惩罚
    }
    
    def __init__(self, 
                 grid_size: Tuple[int, int] = (10, 10), 
                 num_foods: int = 5,
                 num_obstacles: int = 10,
                 food_regen_prob: float = 0.1,
                 random_seed: Optional[int] = None):
        """
        初始化环境。

        Args:
            grid_size (Tuple[int, int], optional): 网格尺寸，默认为(10, 10)。
            num_foods (int, optional): 食物数量，默认为5。
            num_obstacles (int, optional): 障碍物数量，默认为10。
            food_regen_prob (float, optional): 食物再生概率，默认为0.1。
            random_seed (Optional[int], optional): 随机种子，默认为None。
        """
        # 基础参数
        self.height, self.width = grid_size
        self.num_foods = num_foods
        self.num_obstacles = num_obstacles
        self.food_regen_prob = food_regen_prob
        
        # 随机种子设置
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 环境状态
        self.grid = np.zeros((self.height, self.width), dtype=int)
        self.agent_pos = (0, 0)
        self.food_positions = []
        self.obstacle_positions = []
        self.visited_positions = set()
        
        # 环境数据记录
        self.timestep = 0
        self.history = {
            "agent_positions": [],
            "rewards": [],
            "foods_collected": 0,
            "obstacle_collisions": 0,
            "boundary_collisions": 0
        }
        
        # 初始化环境
        self.reset()
    
    def reset(self) -> np.ndarray:
        """
        重置环境到初始状态。

        Returns:
            np.ndarray: 初始观察向量。
        """
        # 清空网格
        self.grid = np.zeros((self.height, self.width), dtype=int)
        
        # 重置智能体位置（左上角）
        self.agent_pos = (0, 0)
        self.visited_positions = {self.agent_pos}
        
        # 放置障碍物（值为-1）
        self.obstacle_positions = []
        for _ in range(self.num_obstacles):
            pos = self._get_random_empty_position()
            if pos:
                self.grid[pos] = -1
                self.obstacle_positions.append(pos)
        
        # 放置食物（值为1）
        self.food_positions = []
        for _ in range(self.num_foods):
            pos = self._get_random_empty_position()
            if pos:
                self.grid[pos] = 1
                self.food_positions.append(pos)
        
        # 重置环境历史记录
        self.timestep = 0
        self.history = {
            "agent_positions": [self.agent_pos],
            "rewards": [],
            "foods_collected": 0,
            "obstacle_collisions": 0,
            "boundary_collisions": 0
        }
        
        # 返回初始观察
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        执行一步环境交互。

        Args:
            action (int): 动作索引 [0=上, 1=下, 2=左, 3=右]。

        Returns:
            Tuple[np.ndarray, float, bool, Dict]: 观察, 奖励, 是否结束, 信息。
        """
        self.timestep += 1
        
        # 获取动作位移
        if 0 <= action < len(self.ACTIONS):
            dx, dy = self.ACTIONS[action]
        else:
            dx, dy = 0, 0  # 无效动作，保持不动
            
        # 计算新位置
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        new_pos = (new_x, new_y)
        
        # 检查边界
        is_boundary_collision = not (0 <= new_x < self.height and 0 <= new_y < self.width)
        
        # 默认移动成本
        reward = self.REWARDS["move"]
        info = {"event": "move"}
        
        if is_boundary_collision:
            # 边界碰撞，保持原位置，受到惩罚
            new_pos = self.agent_pos
            reward = self.REWARDS["boundary"]
            info = {"event": "boundary_collision"}
            self.history["boundary_collisions"] += 1
        else:
            # 检查位置上的物体
            grid_value = self.grid[new_pos]
            
            if grid_value == -1:  # 障碍物
                # 碰到障碍物，保持原位置，受到惩罚
                new_pos = self.agent_pos
                reward = self.REWARDS["obstacle"]
                info = {"event": "obstacle_collision"}
                self.history["obstacle_collisions"] += 1
            elif grid_value == 1:  # 食物
                # 获取食物，得到奖励，食物消失
                reward = self.REWARDS["food"]
                info = {"event": "food_collected"}
                self.grid[new_pos] = 0
                self.food_positions.remove(new_pos)
                self.history["foods_collected"] += 1
                
                # 概率生成新食物
                if np.random.random() < self.food_regen_prob:
                    self._regenerate_food()
            
            # 检查是否访问过的位置
            if new_pos in self.visited_positions:
                # 重复访问惩罚
                reward += self.REWARDS["revisit"]
                info["revisit"] = True
            
            # 更新位置访问记录
            self.visited_positions.add(new_pos)
        
        # 更新智能体位置
        self.agent_pos = new_pos
        
        # 更新历史记录
        self.history["agent_positions"].append(self.agent_pos)
        self.history["rewards"].append(reward)
        
        # 检查是否结束（食物全部收集）
        done = len(self.food_positions) == 0
        
        # 获取观察
        observation = self._get_observation()
        
        return observation, reward, done, info
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前环境状态的观察向量。
        
        观察包含：
        1-2: 智能体相对于网格中心的位置
        3-4: 最近食物的相对位置
        5-6: 最近障碍物的相对位置
        7-8: 最近边界的相对位置

        Returns:
            np.ndarray: 观察向量。
        """
        # 智能体位置（相对于网格中心的标准化坐标）
        agent_x, agent_y = self.agent_pos
        grid_center_x, grid_center_y = self.height / 2, self.width / 2
        rel_x = (agent_x - grid_center_x) / (self.height / 2)
        rel_y = (agent_y - grid_center_y) / (self.width / 2)
        
        # 找到最近的食物
        nearest_food_rel_x, nearest_food_rel_y = 0, 0
        if self.food_positions:
            nearest_food = min(self.food_positions, 
                              key=lambda pos: abs(pos[0] - agent_x) + abs(pos[1] - agent_y))
            nearest_food_rel_x = (nearest_food[0] - agent_x) / self.height
            nearest_food_rel_y = (nearest_food[1] - agent_y) / self.width
        
        # 找到最近的障碍物
        nearest_obs_rel_x, nearest_obs_rel_y = 0, 0
        if self.obstacle_positions:
            nearest_obs = min(self.obstacle_positions, 
                             key=lambda pos: abs(pos[0] - agent_x) + abs(pos[1] - agent_y))
            nearest_obs_rel_x = (nearest_obs[0] - agent_x) / self.height
            nearest_obs_rel_y = (nearest_obs[1] - agent_y) / self.width
        
        # 最近边界距离（标准化）
        bound_up = agent_x / self.height
        bound_down = (self.height - 1 - agent_x) / self.height
        bound_left = agent_y / self.width
        bound_right = (self.width - 1 - agent_y) / self.width
        nearest_bound_x = min(bound_up, bound_down)
        nearest_bound_y = min(bound_left, bound_right)
        
        # 组合观察向量
        observation = np.array([
            rel_x, rel_y,                         # 智能体相对位置
            nearest_food_rel_x, nearest_food_rel_y,  # 最近食物相对位置
            nearest_obs_rel_x, nearest_obs_rel_y,    # 最近障碍物相对位置
            nearest_bound_x, nearest_bound_y         # 最近边界相对位置
        ], dtype=np.float32)
        
        return observation
    
    def _get_random_empty_position(self) -> Optional[Tuple[int, int]]:
        """
        获取一个随机的空位置。

        Returns:
            Optional[Tuple[int, int]]: 随机空位置坐标，如果没有空位置则返回None。
        """
        # 尝试最多100次找到一个空位置
        for _ in range(100):
            x = np.random.randint(0, self.height)
            y = np.random.randint(0, self.width)
            pos = (x, y)
            
            # 检查位置是否为空且不是智能体位置
            if self.grid[pos] == 0 and pos != self.agent_pos:
                return pos
        
        # 找不到空位置
        return None
    
    def _regenerate_food(self) -> None:
        """
        在网格中随机位置生成一个新的食物。
        """
        pos = self._get_random_empty_position()
        if pos:
            self.grid[pos] = 1
            self.food_positions.append(pos)
    
    def render(self, mode: str = 'human') -> Optional[plt.Figure]:
        """
        渲染环境。

        Args:
            mode (str, optional): 渲染模式，可选'human'或'rgb_array'，默认为'human'。

        Returns:
            Optional[plt.Figure]: 渲染的图像对象，如果mode=human则返回None。
        """
        # 创建图像
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 绘制网格
        ax.set_xlim([-0.5, self.width - 0.5])
        ax.set_ylim([self.height - 0.5, -0.5])
        ax.set_xticks(np.arange(-0.5, self.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.height, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)
        
        # 绘制障碍物
        for obs_pos in self.obstacle_positions:
            i, j = obs_pos
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='black'))
        
        # 绘制食物
        for food_pos in self.food_positions:
            i, j = food_pos
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color='green'))
            plt.text(j, i, "F", ha='center', va='center', color='white', fontsize=12)
        
        # 绘制智能体
        i, j = self.agent_pos
        ax.add_patch(plt.Rectangle((j - 0.4, i - 0.4), 0.8, 0.8, color='blue'))
        plt.text(j, i, "A", ha='center', va='center', color='white', fontsize=12)
        
        # 绘制智能体轨迹
        if len(self.history["agent_positions"]) > 1:
            traj_x = [pos[1] for pos in self.history["agent_positions"]]
            traj_y = [pos[0] for pos in self.history["agent_positions"]]
            ax.plot(traj_x, traj_y, 'r-', alpha=0.5, linewidth=2)
        
        # 绘制统计信息
        stats = (
            f"步数: {self.timestep}\n"
            f"食物: {self.history['foods_collected']}\n"
            f"障碍碰撞: {self.history['obstacle_collisions']}\n"
            f"边界碰撞: {self.history['boundary_collisions']}"
        )
        ax.text(self.width - 0.5, -0.5, stats, ha='right', va='bottom', fontsize=10)
        
        # 设置标题
        plt.title(f"网格环境 - 步数: {self.timestep}")
        
        if mode == 'human':
            plt.show()
            return None
        elif mode == 'rgb_array':
            return fig
        
    def get_state(self) -> Dict[str, Any]:
        """
        获取环境当前状态。

        Returns:
            Dict[str, Any]: 当前环境状态信息。
        """
        return {
            "grid": self.grid.copy(),
            "agent_pos": self.agent_pos,
            "food_positions": self.food_positions.copy(),
            "obstacle_positions": self.obstacle_positions.copy(),
            "visited_positions": self.visited_positions.copy(),
            "timestep": self.timestep,
            "history": {k: v.copy() if isinstance(v, list) else v 
                       for k, v in self.history.items()}
        }
    
    def load_state(self, state: Dict[str, Any]) -> None:
        """
        加载环境状态。

        Args:
            state (Dict[str, Any]): 要加载的环境状态。
        """
        self.grid = state["grid"].copy()
        self.agent_pos = state["agent_pos"]
        self.food_positions = state["food_positions"].copy()
        self.obstacle_positions = state["obstacle_positions"].copy()
        self.visited_positions = state["visited_positions"].copy()
        self.timestep = state["timestep"]
        self.history = {k: v.copy() if isinstance(v, list) else v 
                       for k, v in state["history"].items()} 