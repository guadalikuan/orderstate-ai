#!/usr/bin/env python
"""
八阶段闭环与生存焦虑驱动注意力实验 - 简化Web可视化启动脚本
"""
import os
import sys
import webbrowser
import threading
import time
from flask import Flask, render_template, jsonify

# 确保可以找到miniexp包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__, 
            template_folder='miniexp/web/templates',
            static_folder='miniexp/web/static')

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """返回系统状态"""
    return jsonify({
        'status': 'running',
        'message': '系统运行正常',
        'time': time.time()
    })

def open_browser():
    """在新线程中打开浏览器"""
    # 延迟2秒，等待服务器启动
    time.sleep(2)
    # 自动打开浏览器
    webbrowser.open('http://localhost:5002')

if __name__ == '__main__':
    print("=" * 60)
    print("八阶段闭环与生存焦虑驱动注意力实验 - 简化Web可视化版")
    print("=" * 60)
    print("正在启动Web服务器...")
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    # 启动Flask应用
    app.run(debug=True, host='0.0.0.0', port=5002) 