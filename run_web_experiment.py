#!/usr/bin/env python
"""
八阶段闭环与生存焦虑驱动注意力实验 - Web可视化启动脚本
"""
import os
import sys
import webbrowser
import threading
import time

# 确保可以找到miniexp包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 修复eventlet和WebSocket支持
try:
    import eventlet
    eventlet.monkey_patch()  # 应用monkey patch，使其支持异步IO
    print("已加载eventlet支持")
except ImportError:
    print("警告: eventlet未安装，WebSocket可能无法正常工作")
    print("请运行: pip install eventlet")

# 从Web模块导入Flask应用
from miniexp.web.app import app, socketio

def open_browser():
    """在新线程中打开浏览器"""
    # 延迟2秒，等待服务器启动
    time.sleep(2)
    # 自动打开浏览器
    webbrowser.open('http://localhost:5000')

if __name__ == '__main__':
    print("=" * 60)
    print("八阶段闭环与生存焦虑驱动注意力实验 - Web可视化版")
    print("=" * 60)
    print("正在启动Web服务器...")
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动Flask应用，使用eventlet作为Web服务器
    socketio.run(app, debug=True, host='0.0.0.0', port=5000, allow_unsafe_werkzeug=True) 