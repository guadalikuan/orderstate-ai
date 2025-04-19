import os
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.font_manager import FontProperties
import platform

def setup_chinese_font():
    """
    设置matplotlib的中文字体支持
    
    此函数尝试找到系统上支持中文的字体，并应用到matplotlib配置中
    """
    # 检测操作系统
    system = platform.system()
    
    # 根据操作系统类型提供常见的中文字体列表
    if system == 'Windows':
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi']
    elif system == 'Linux':
        chinese_fonts = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'AR PL UMing CN', 'AR PL UKai CN']
    elif system == 'Darwin':  # macOS
        chinese_fonts = ['STHeiti', 'Heiti TC', 'Songti SC', 'Songti TC', 'PingFang SC', 'PingFang TC']
    else:
        chinese_fonts = ['SimHei', 'Microsoft YaHei', 'STHeiti']
    
    # 尝试在系统中查找这些字体
    font_found = False
    for font_name in chinese_fonts:
        try:
            font_path = fm.findfont(FontProperties(family=font_name), fallback_to_default=False)
            if font_path:
                plt.rcParams['font.family'] = [font_name]
                mpl.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
                font_found = True
                print(f"使用中文字体: {font_name}")
                break
        except:
            continue
    
    # 如果没有找到系统字体，尝试加载自带的中文字体
    if not font_found:
        try:
            # 检查是否存在fonts目录和字体文件
            font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
            if os.path.exists(font_dir):
                # 尝试加载目录中的字体
                for font_file in os.listdir(font_dir):
                    if font_file.endswith('.ttf') or font_file.endswith('.otf'):
                        font_path = os.path.join(font_dir, font_file)
                        # 添加字体文件
                        font_manager_path = fm.fontManager.addfont(font_path)
                        # 使用添加的字体
                        plt.rcParams['font.family'] = ['sans-serif']
                        plt.rcParams['font.sans-serif'] = [os.path.splitext(font_file)[0]] + plt.rcParams['font.sans-serif']
                        mpl.rcParams['axes.unicode_minus'] = False
                        font_found = True
                        print(f"使用内置中文字体: {font_file}")
                        break
        except Exception as e:
            print(f"加载内置字体出错: {e}")
    
    if not font_found:
        print("警告: 未找到中文字体，图表中的中文可能无法正确显示")
        print("请安装中文字体或将中文字体文件添加到 miniexp/fonts/ 目录中")
        
        # 最后尝试使用matplotlib的默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans'] + plt.rcParams['font.sans-serif']
        mpl.rcParams['axes.unicode_minus'] = False

def create_fonts_directory():
    """
    创建fonts目录（如果不存在）
    """
    font_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts')
    if not os.path.exists(font_dir):
        os.makedirs(font_dir)
        print(f"已创建字体目录: {font_dir}")
        print("请将中文字体文件(.ttf或.otf)放入此目录")
    
    return font_dir 