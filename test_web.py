from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "Hello! Web服务正在运行！"

if __name__ == '__main__':
    print("启动测试Web服务器...")
    app.run(debug=True, host='0.0.0.0', port=5001) 