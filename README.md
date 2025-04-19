# 八阶段闭环与生存焦虑驱动注意力实验

这个项目实现了一个智能体模拟系统，演示"能量→信号→数据→信息→知识→智慧→决策→动作→能量"八阶段闭环及"生存焦虑驱动Attention"机制。

## 系统概述

本系统模拟了不同智能体在多种环境中的行为表现，特别关注在能量受限情况下智能体的注意力分配策略。

### 核心概念

1. **八阶段闭环**：一个完整的认知-行动循环，包括：
   - 能量 → 信号：智能体的能量状态影响内部焦虑信号
   - 信号 → 数据：环境状态被感知并转化为特征表示
   - 数据 → 信息：通过注意力机制处理特征
   - 信息 → 知识：智能体内部模型解释信息
   - 知识 → 智慧：智能体根据知识权衡不同决策
   - 智慧 → 决策：选择最佳行动
   - 决策 → 动作：执行选定的动作
   - 动作 → 能量：行动消耗能量，循环回到第一阶段

2. **生存焦虑驱动注意力**：
   - 当能量接近阈值时，智能体产生焦虑感
   - 焦虑度越高，注意力分布越集中（更警觉）
   - 焦虑度越低，注意力分布越均匀（更探索）

## 安装指南

### 使用Anaconda（推荐）

1. 克隆仓库：
   ```bash
   git clone <repository-url>
   cd orderstate-ai
   ```

2. 使用环境配置文件创建Conda环境：
   ```bash
   conda env create -f environment.yml
   ```

3. 激活环境：
   ```bash
   conda activate orderstate-ai
   ```

### 使用pip

1. 克隆仓库：
   ```bash
   git clone <repository-url>
   cd orderstate-ai
   ```

2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```

## 运行实验

### 命令行实验

运行基本实验：
```bash
python -m miniexp.main
```

使用特定环境类型和回合数：
```bash
python -m miniexp.main --env medium --episodes 50
```

查看所有选项：
```bash
python -m miniexp.main --help
```

或使用快捷脚本：
```bash
python run_experiment.py --env advanced --episodes 20
```

### Web可视化实验

启动Web可视化界面：
```bash
python run_web_experiment.py
```

这将启动服务器，并自动打开浏览器访问http://localhost:5000

## 系统组件

### 环境系统
- **SimpleEnvironment**: 基础环境，无障碍物
- **MediumEnvironment**: 中等环境，有静态障碍物
- **AdvancedEnvironment**: 高级环境，有动态障碍物和捕食者

### 智能体系统
- **BaselineAgent**: 基线智能体，随机决策
- **EnergyAgent**: 能量智能体，包含能量管理和焦虑机制

### 实验系统
- **Experiment**: 实验配置和执行
- **MetricsRecorder**: 记录和统计实验指标
- **ExperimentReport**: 生成实验报告

### 可视化系统
- Web界面：实时展示实验过程
- 八阶段循环可视化：显示每个阶段的状态
- 性能指标图表：展示智能体表现

## 常见问题

1. **中文显示问题**：
   如果图表中的中文显示为方块，请确保系统安装了中文字体，并在Windows系统上安装pywin32。

2. **Web界面启动失败**：
   - 检查端口5000是否被占用：`netstat -an | findstr 5000`
   - 确认依赖版本兼容：特别是flask, werkzeug和socketio的版本

3. **环境创建失败**：
   请检查您的Python版本（推荐3.8-3.9），确保兼容性。

## 许可证

MIT

## 联系方式

如有问题或建议，请[提交issue](https://github.com/yourusername/orderstate-ai/issues)。 