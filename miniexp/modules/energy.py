import numpy as np

class EnergyModule:
    """
    管理智能体的能量状态并计算生存焦虑。
    当能量接近阈值时，会产生生存焦虑。
    """
    def __init__(self, init_energy: float, threshold: float, max_energy: float = None):
        """
        初始化能量模块。

        Args:
            init_energy (float): 初始能量值。
            threshold (float): 能量阈值，低于此值表示危险。
            max_energy (float, optional): 能量上限。默认等于初始能量。
        """
        if threshold <= 0 or init_energy <= 0:
            raise ValueError("能量参数必须为正值")
        if threshold >= init_energy:
            raise ValueError("阈值必须小于初始能量")
        
        self.init_energy = init_energy
        self.threshold = threshold
        self.max_energy = max_energy if max_energy is not None else init_energy
        self.current_energy = init_energy
        
    def reset(self):
        """重置能量到初始值"""
        self.current_energy = self.init_energy
        
    def consume(self, amount: float = 1.0) -> bool:
        """
        消耗能量。

        Args:
            amount (float): 消耗的能量量。默认为 1.0。

        Returns:
            bool: 能量是否耗尽 (低于或等于0)。
        """
        # [阶段 8 -> 1: 动作消耗能量]
        self.current_energy = max(0, self.current_energy - amount)
        return self.current_energy <= 0
        
    def add(self, amount: float):
        """
        增加能量。

        Args:
            amount (float): 增加的能量量。
        """
        self.current_energy = min(self.max_energy, self.current_energy + amount)
        
    def get_energy(self) -> float:
        """
        获取当前能量。

        Returns:
            float: 当前能量值。
        """
        return self.current_energy
        
    def get_anxiety(self) -> float:
        """
        计算生存焦虑度。
        能量越接近阈值，焦虑度越高。能量低于阈值时，焦虑度达到最大值。

        Returns:
            float: 生存焦虑度 (范围通常在 0 到 2 之间)：
                  - 能量高于阈值时：焦虑度随能量接近阈值而增大，接近0到1之间
                  - 能量低于阈值时：最大焦虑，值为2.0
        """
        # [阶段 1: 能量状态转化为内部焦虑信号]
        if self.current_energy <= self.threshold:
            # 能量低于阈值，最大焦虑
            return 2.0
        else:
            # 能量在阈值和初始值之间，焦虑随能量减少而增加
            # 映射到0-1之间
            energy_ratio = (self.current_energy - self.threshold) / (self.init_energy - self.threshold)
            # 使用非线性函数使焦虑在能量接近阈值时快速增加
            anxiety = 1.0 - np.sqrt(energy_ratio)
            return max(0, anxiety)

# 测试代码
if __name__ == '__main__':
    energy_module = EnergyModule(init_energy=100, threshold=20)
    print(f"初始能量: {energy_module.get_energy()}, 焦虑度: {energy_module.get_anxiety():.2f}")
    
    # 测试能量消耗和焦虑度变化
    for i in range(5):
        energy_module.consume(20)
        print(f"剩余能量: {energy_module.get_energy()}, 焦虑度: {energy_module.get_anxiety():.2f}") 