"""
OrderState AI实验库
用于智能体进化实验的工具包
"""

__version__ = "0.1.0"

# 导入环境模块
from miniexp.env.simple import SimpleEnvironment
from miniexp.env.medium import MediumEnvironment 
from miniexp.env.advanced import AdvancedEnvironment
from miniexp.env.base import BaseEnvironment

# 导入智能体模块
from miniexp.agent.baseline import BaselineAgent
from miniexp.agent.energy import EnergyAgent
from miniexp.agent.base import BaseAgent

# 导入实验模块
from miniexp.experiment.experiment import Experiment
from miniexp.experiment.metrics_recorder import MetricsRecorder
from miniexp.experiment.report import ExperimentReport
from miniexp.experiment.config import ExperimentConfig, EnvironmentType, AgentType

# 导入工具函数
from miniexp.utils import setup_chinese_font, create_fonts_directory

# 自动设置中文字体支持
setup_chinese_font() 