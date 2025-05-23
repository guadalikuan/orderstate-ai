// 当文档加载完成时执行
document.addEventListener('DOMContentLoaded', function() {
    // 全局变量
    let socket;
    let resultsChart = null;
    let gridWidth = parseInt(document.getElementById('gridWidth').value);
    let gridHeight = parseInt(document.getElementById('gridHeight').value);
    let cycleVisualizer = null;
    let isRunning = false;
    
    // DOM元素
    const startButton = document.getElementById('startButton');
    const stopButton = document.getElementById('stopButton');
    const gridWorldContainer = document.getElementById('gridWorld');
    const messageContainer = document.getElementById('messageContainer');
    const resultsContainer = document.getElementById('resultsContainer');
    const currentAgentEl = document.getElementById('currentAgent');
    const currentEpisodeEl = document.getElementById('currentEpisode');
    const totalEpisodesEl = document.getElementById('totalEpisodes');
    const currentStepEl = document.getElementById('currentStep');
    const energyInfoEl = document.getElementById('energyInfo');
    const currentEnergyEl = document.getElementById('currentEnergy');
    const currentAnxietyEl = document.getElementById('currentAnxiety');
    const energyBarEl = document.getElementById('energyBar');
    const anxietyBarEl = document.getElementById('anxietyBar');
    
    // 初始化Socket.IO连接
    function initializeSocket() {
        try {
            console.log("尝试初始化Socket.IO连接...");
            // 连接到服务器（强制使用WebSocket传输）
            socket = io({
                transports: ['websocket', 'polling'],
                reconnectionAttempts: 5,
                reconnectionDelay: 1000
            });
            
            socket.on('connect', function() {
                console.log("Socket.IO连接成功！Socket ID:", socket.id);
                addMessage("已连接到服务器");
            });
            
            socket.on('connect_error', function(error) {
                console.error("Socket.IO连接错误:", error);
                addMessage("连接服务器时出错: " + error);
            });
            
            // 服务器消息处理
            socket.on('server_message', function(data) {
                console.log("收到服务器消息:", data);
                addMessage(data.message);
            });
            
            // 状态更新处理
            socket.on('state_update', function(data) {
                console.log("收到状态更新:", data);
                updateGridWorld(data);
                updateStatusInfo(data);
            });
            
            // 实验完成处理
            socket.on('experiment_complete', function(data) {
                console.log("实验完成:", data);
                addMessage(data.message);
                fetchResults();
                updateUIState(false);
            });
            
            // 连接断开处理
            socket.on('disconnect', function(reason) {
                console.log("Socket.IO连接断开, 原因:", reason);
                addMessage(`与服务器的连接已断开: ${reason}`);
                updateUIState(false);
            });
            
            // 调试用的事件，显示所有收到的事件
            socket.onAny((event, ...args) => {
                console.log(`收到事件 ${event}:`, args);
            });
        } catch (error) {
            console.error("初始化Socket.IO连接时出错:", error);
            addMessage("无法初始化Socket.IO连接: " + error);
        }
    }
    
    // 初始化网格世界
    function initializeGridWorld() {
        gridWidth = parseInt(document.getElementById('gridWidth').value);
        gridHeight = parseInt(document.getElementById('gridHeight').value);
        
        // 设置网格模板
        gridWorldContainer.style.gridTemplateColumns = `repeat(${gridWidth}, 1fr)`;
        gridWorldContainer.style.gridTemplateRows = `repeat(${gridHeight}, 1fr)`;
        
        // 清空现有网格
        gridWorldContainer.innerHTML = '';
        
        // 创建网格单元格
        for (let row = 0; row < gridHeight; row++) {
            for (let col = 0; col < gridWidth; col++) {
                const cell = document.createElement('div');
                cell.className = 'grid-cell';
                cell.id = `cell-${row}-${col}`;
                cell.dataset.row = row;
                cell.dataset.col = col;
                
                // 右下角是目标位置
                if (row === gridHeight - 1 && col === gridWidth - 1) {
                    cell.classList.add('target');
                    cell.innerHTML = 'T';
                }
                
                // 左上角是起始位置
                if (row === 0 && col === 0) {
                    cell.classList.add('agent');
                    cell.innerHTML = 'A';
                }
                
                gridWorldContainer.appendChild(cell);
            }
        }
    }
    
    // 更新网格世界
    function updateGridWorld(data) {
        console.log("收到状态更新:", data);
        
        // 如果网格尺寸变化，重新初始化
        if (data.grid_width !== gridWidth || data.grid_height !== gridHeight) {
            gridWidth = data.grid_width;
            gridHeight = data.grid_height;
            initializeGridWorld();
        }
        
        // 清除所有agent标记
        document.querySelectorAll('.grid-cell.agent').forEach(cell => {
            cell.classList.remove('agent', 'energy-agent', 'baseline-agent');
            if (!cell.classList.contains('target')) {
                cell.innerHTML = '';
            }
        });
        
        try {
            // 设置新的agent位置
            const [row, col] = data.state;
            console.log(`尝试更新Agent位置: row=${row}, col=${col}`);
            const cellId = `cell-${row}-${col}`;
            console.log(`寻找元素: #${cellId}`);
            
            const cell = document.getElementById(cellId);
            if (cell) {
                console.log(`找到单元格 ${cellId}，更新显示`);
                cell.classList.add('agent');
                
                // 根据智能体类型添加不同样式
                if (data.agent === 'EnergyAgent') {
                    cell.classList.add('energy-agent');
                    if (cell.classList.contains('target')) {
                        cell.innerHTML = 'A+T';
                    } else {
                        // 添加能量显示
                        if (data.energy !== undefined) {
                            cell.innerHTML = `<div class="agent-indicator">A</div>
                                             <div class="energy-display">${data.energy.toFixed(1)}</div>`;
                        } else {
                            cell.innerHTML = 'A';
                        }
                    }
                } else {
                    cell.classList.add('baseline-agent');
                    if (cell.classList.contains('target')) {
                        cell.innerHTML = 'A+T';
                    } else {
                        cell.innerHTML = 'A';
                    }
                }
                
                // 添加动画效果
                cell.classList.add('pulse');
                setTimeout(() => {
                    cell.classList.remove('pulse');
                }, 500);
            } else {
                console.error(`未找到单元格: #${cellId}`);
                // 调试信息：输出所有网格单元格的ID
                const allCells = document.querySelectorAll('.grid-cell');
                console.log(`当前网格有 ${allCells.length} 个单元格`);
                if (allCells.length < 20) {
                    const cellIds = Array.from(allCells).map(c => c.id);
                    console.log("可用的单元格ID:", cellIds);
                }
            }
        } catch (error) {
            console.error("更新网格世界时出错:", error);
        }
    }
    
    // 更新状态信息
    function updateStatusInfo(data) {
        currentAgentEl.textContent = data.agent_name || data.agent;
        currentStepEl.textContent = data.step;
        
        // 如果是EnergyAgent，显示能量信息
        if ((data.agent_name === 'EnergyAgent' || data.agent === 'EnergyAgent') && data.energy !== undefined) {
            energyInfoEl.style.display = 'block';
            currentEnergyEl.textContent = data.energy.toFixed(1);
            energyBarEl.style.width = `${(data.energy / 100) * 100}%`;
            
            if (data.anxiety !== undefined) {
                currentAnxietyEl.textContent = data.anxiety.toFixed(2);
                anxietyBarEl.style.width = `${data.anxiety * 100}%`;
            }
        } else {
            energyInfoEl.style.display = 'none';
        }
        
        // 获取八阶段循环状态
        if (isRunning) {
            fetchCycleState();
        }
    }
    
    // 获取循环状态
    function fetchCycleState() {
        fetch('/api/cycle_state')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    // 更新循环可视化
                    if (cycleVisualizer) {
                        cycleVisualizer.updateState(data.current);
                    }
                } else if (cycleVisualizer && data) {
                    // 兼容旧格式：将数据转换为可视化器需要的格式
                    const cycleData = {};
                    for (const [stage, stateObj] of Object.entries(data.states || {})) {
                        cycleData[stage] = {
                            value: stateObj.value
                        };
                    }
                    // 更新可视化器
                    cycleVisualizer.updateState(cycleData);
                }
            })
            .catch(error => console.error('获取循环状态失败:', error));
    }
    
    // 添加消息
    function addMessage(message) {
        const messageEl = document.createElement('div');
        messageEl.className = 'message';
        messageEl.textContent = message;
        messageContainer.appendChild(messageEl);
        
        // 自动滚动到底部
        messageContainer.scrollTop = messageContainer.scrollHeight;
    }
    
    // 获取实验结果
    function fetchResults() {
        fetch('/api/experiment_results')
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    displayResults(data.results);
                } else {
                    addMessage('获取结果失败: ' + data.message);
                }
            })
            .catch(error => {
                addMessage('获取结果时出错: ' + error);
            });
    }
    
    // 显示实验结果
    function displayResults(results) {
        resultsContainer.innerHTML = '';
        
        // 创建结果表格
        const table = document.createElement('table');
        table.className = 'results-table table table-sm table-striped';
        
        // 表头
        const thead = document.createElement('thead');
        const headerRow = document.createElement('tr');
        headerRow.innerHTML = `
            <th>智能体</th>
            <th>成功率</th>
            <th>平均步数</th>
            <th>剩余能量</th>
        `;
        thead.appendChild(headerRow);
        table.appendChild(thead);
        
        // 表体
        const tbody = document.createElement('tbody');
        for (const [agent, metrics] of Object.entries(results)) {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${agent}</td>
                <td>${metrics['成功率 (%)'].toFixed(1)}%</td>
                <td>${metrics['平均步数'].toFixed(1)}</td>
                <td>${agent === 'EnergyAgent' ? metrics['平均剩余能量'].toFixed(1) : 'N/A'}</td>
            `;
            tbody.appendChild(row);
        }
        table.appendChild(tbody);
        resultsContainer.appendChild(table);
        
        // 创建图表
        createResultsChart(results);
    }
    
    // 创建结果图表
    function createResultsChart(results) {
        const agentNames = Object.keys(results);
        
        // 准备图表数据
        const successRates = agentNames.map(agent => results[agent]['成功率 (%)']);
        const avgSteps = agentNames.map(agent => results[agent]['平均步数']);
        const energyExhaustedRates = agentNames.map(agent => {
            return results[agent]['能量耗尽失败率 (%)'] || 0;
        });
        
        // 销毁现有图表（如果存在）
        if (resultsChart) {
            resultsChart.destroy();
        }
        
        // 创建新图表
        const ctx = document.getElementById('resultsChart').getContext('2d');
        resultsChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: agentNames,
                datasets: [
                    {
                        label: '成功率 (%)',
                        data: successRates,
                        backgroundColor: 'rgba(75, 192, 192, 0.5)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                        borderWidth: 1
                    },
                    {
                        label: '平均步数',
                        data: avgSteps,
                        backgroundColor: 'rgba(255, 159, 64, 0.5)',
                        borderColor: 'rgba(255, 159, 64, 1)',
                        borderWidth: 1
                    },
                    {
                        label: '能量耗尽率 (%)',
                        data: energyExhaustedRates,
                        backgroundColor: 'rgba(255, 99, 132, 0.5)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
    
    // 更新UI状态
    function updateUIState(running) {
        isRunning = running;
        startButton.disabled = running;
        stopButton.disabled = !running;
        
        // 如果实验在运行，开始定期获取循环状态
        if (running) {
            // 立即获取一次
            fetchCycleState();
            
            // 然后每秒获取一次
            if (!window.cycleStateInterval) {
                window.cycleStateInterval = setInterval(fetchCycleState, 1000);
            }
        } else {
            // 停止获取
            if (window.cycleStateInterval) {
                clearInterval(window.cycleStateInterval);
                window.cycleStateInterval = null;
            }
        }
    }
    
    // 开始实验处理
    function handleStartExperiment() {
        // 读取配置
        const config = {
            grid_width: parseInt(document.getElementById('gridWidth').value),
            grid_height: parseInt(document.getElementById('gridHeight').value),
            episodes: parseInt(document.getElementById('episodes').value),
            max_steps: parseInt(document.getElementById('maxSteps').value),
            init_energy: parseFloat(document.getElementById('initEnergy').value),
            energy_threshold: parseFloat(document.getElementById('energyThreshold').value),
            display_interval: parseFloat(document.getElementById('displayInterval').value)
        };
        
        // 发送开始请求
        fetch('/api/start_experiment', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(config)
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addMessage(data.message);
                
                // 更新UI
                updateUIState(true);
                
                // 更新总回合数显示
                totalEpisodesEl.textContent = config.episodes * 2; // 两个智能体
                
                // 初始化网格世界
                gridWidth = config.grid_width;
                gridHeight = config.grid_height;
                initializeGridWorld();
                
                // 清空结果
                resultsContainer.innerHTML = '<p class="text-center text-muted">实验进行中...</p>';
                if (resultsChart) {
                    resultsChart.destroy();
                    resultsChart = null;
                }
            } else {
                addMessage('启动实验失败: ' + data.message);
            }
        })
        .catch(error => {
            addMessage('启动实验时出错: ' + error);
        });
    }
    
    // 停止实验处理
    function handleStopExperiment() {
        fetch('/api/stop_experiment', {
            method: 'POST'
        })
        .then(response => response.json())
        .then(data => {
            if (data.status === 'success') {
                addMessage(data.message);
                updateUIState(false);
            } else {
                addMessage('停止实验失败: ' + data.message);
            }
        })
        .catch(error => {
            addMessage('停止实验时出错: ' + error);
        });
    }
    
    // 初始化
    function initialize() {
        console.log("初始化应用...");
        
        // 初始化Socket.IO连接
        initializeSocket();
        
        // 初始化网格世界
        initializeGridWorld();
        
        // 初始化循环可视化器
        cycleVisualizer = new CycleVisualizer('orderStateCycle');
        
        // 实验控制按钮事件监听
        startButton.addEventListener('click', handleStartExperiment);
        stopButton.addEventListener('click', handleStopExperiment);
        
        // 添加表单字段验证
        document.querySelectorAll('#experimentForm input').forEach(input => {
            input.addEventListener('change', function() {
                if (this.id === 'gridWidth' || this.id === 'gridHeight') {
                    initializeGridWorld();
                }
            });
        });
        
        // 初始禁用停止按钮
        stopButton.disabled = true;
        
        // 清空消息容器
        messageContainer.innerHTML = '';
        addMessage('系统就绪。请设置参数并点击"开始实验"按钮。');
    }
    
    // 执行初始化
    initialize();
}); 