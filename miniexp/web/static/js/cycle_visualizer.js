// 八阶段循环可视化器
class CycleVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.stageColors = {
            energy: "#FF5733",      // 红橙色
            signal: "#FFC300",      // 黄色
            data: "#DAF7A6",        // 浅绿
            information: "#C70039", // 深红
            knowledge: "#900C3F",   // 紫红
            wisdom: "#581845",      // 紫色
            decision: "#2471A3",    // 蓝色
            action: "#148F77"       // 绿色
        };
        this.initLayout();
    }
    
    initLayout() {
        // 创建八个阶段的容器
        const stages = ['energy', 'signal', 'data', 'information', 
                        'knowledge', 'wisdom', 'decision', 'action'];
        
        // 清空容器
        this.container.innerHTML = '';
        
        // 创建循环图容器
        const cycleDiv = document.createElement('div');
        cycleDiv.className = 'cycle-container';
        this.container.appendChild(cycleDiv);
        
        // 创建八个阶段节点
        stages.forEach((stage, index) => {
            const stageDiv = document.createElement('div');
            stageDiv.className = 'cycle-stage';
            stageDiv.id = `stage-${stage}`;
            stageDiv.style.backgroundColor = this.stageColors[stage];
            
            // 计算位置（围成一个圆）
            const angle = (index / stages.length) * 2 * Math.PI;
            const radius = 150; // 圆的半径
            const x = Math.cos(angle) * radius + radius + 20;
            const y = Math.sin(angle) * radius + radius + 20;
            
            stageDiv.style.left = `${x}px`;
            stageDiv.style.top = `${y}px`;
            
            // 添加标签
            const label = document.createElement('div');
            label.className = 'stage-label';
            label.textContent = stage.charAt(0).toUpperCase() + stage.slice(1);
            stageDiv.appendChild(label);
            
            // 添加值容器
            const valueDiv = document.createElement('div');
            valueDiv.className = 'stage-value';
            valueDiv.id = `${stage}-value`;
            stageDiv.appendChild(valueDiv);
            
            cycleDiv.appendChild(stageDiv);
            
            // 添加到下一阶段的连接线
            if (index < stages.length - 1) {
                this.createArrow(cycleDiv, index, index + 1, stages.length);
            } else {
                // 从最后一个到第一个的连接
                this.createArrow(cycleDiv, index, 0, stages.length);
            }
        });
        
        // 添加详细信息面板
        const detailPanel = document.createElement('div');
        detailPanel.className = 'cycle-details';
        detailPanel.id = 'cycle-details';
        detailPanel.innerHTML = '<h4>阶段详情</h4><div id="stage-details">选择一个阶段查看详情</div>';
        this.container.appendChild(detailPanel);
    }
    
    createArrow(container, fromIndex, toIndex, totalStages) {
        const arrowDiv = document.createElement('div');
        arrowDiv.className = 'cycle-arrow';
        
        // 计算起点和终点
        const fromAngle = (fromIndex / totalStages) * 2 * Math.PI;
        const toAngle = (toIndex / totalStages) * 2 * Math.PI;
        const radius = 150;
        const centerX = radius + 20;
        const centerY = radius + 20;
        
        // 计算箭头起点和终点
        const fromX = Math.cos(fromAngle) * radius + centerX;
        const fromY = Math.sin(fromAngle) * radius + centerY;
        const toX = Math.cos(toAngle) * radius + centerX;
        const toY = Math.sin(toAngle) * radius + centerY;
        
        // 计算线的长度和角度
        const length = Math.sqrt(Math.pow(toX - fromX, 2) + Math.pow(toY - fromY, 2));
        const angle = Math.atan2(toY - fromY, toX - fromX) * 180 / Math.PI;
        
        // 设置箭头样式
        arrowDiv.style.width = `${length}px`;
        arrowDiv.style.left = `${fromX}px`;
        arrowDiv.style.top = `${fromY}px`;
        arrowDiv.style.transform = `rotate(${angle}deg)`;
        
        container.appendChild(arrowDiv);
    }
    
    updateState(cycleData) {
        // 更新各阶段状态显示
        for (const stage in cycleData) {
            const valueDiv = document.getElementById(`${stage}-value`);
            if (!valueDiv) continue;
            
            const data = cycleData[stage].value;
            
            // 根据数据类型显示简要信息
            if (data === null || data === undefined) {
                valueDiv.textContent = "等待中";
                valueDiv.style.backgroundColor = "#ccc";
            } else {
                // 根据数据类型显示不同的内容
                if (typeof data === 'number') {
                    valueDiv.textContent = data.toFixed(1);
                } else if (typeof data === 'object') {
                    // 对象类型，显示摘要
                    valueDiv.textContent = "数据就绪";
                } else {
                    valueDiv.textContent = String(data).substr(0, 10);
                }
                valueDiv.style.backgroundColor = "rgba(255,255,255,0.7)";
                
                // 高亮当前激活的阶段
                const stageDiv = document.getElementById(`stage-${stage}`);
                stageDiv.classList.add('active-stage');
                setTimeout(() => {
                    stageDiv.classList.remove('active-stage');
                }, 1000);
            }
            
            // 添加点击事件显示详情
            valueDiv.onclick = () => {
                this.showStageDetails(stage, data);
            };
        }
    }
    
    showStageDetails(stage, data) {
        const detailsDiv = document.getElementById('stage-details');
        detailsDiv.innerHTML = `<h5>${stage.charAt(0).toUpperCase() + stage.slice(1)}阶段</h5>`;
        
        if (data === null || data === undefined) {
            detailsDiv.innerHTML += `<p>暂无数据</p>`;
            return;
        }
        
        // 根据不同阶段显示不同的详细信息
        switch(stage) {
            case 'energy':
                detailsDiv.innerHTML += `
                    <p>当前能量: ${data}</p>
                    <div class="progress">
                        <div class="progress-bar bg-success" style="width: ${data}%"></div>
                    </div>
                `;
                break;
            case 'signal':
                if (typeof data === 'object') {
                    detailsDiv.innerHTML += `
                        <p>智能体位置: (${data.position[0]}, ${data.position[1]})</p>
                        <p>目标位置: (${data.target[0]}, ${data.target[1]})</p>
                    `;
                }
                break;
            case 'data':
                if (Array.isArray(data)) {
                    detailsDiv.innerHTML += `
                        <p>特征向量: [${data.join(', ')}]</p>
                    `;
                }
                break;
            case 'information':
                if (typeof data === 'object') {
                    detailsDiv.innerHTML += `
                        <p>焦虑水平: ${data.anxiety.toFixed(2)}</p>
                        <p>注意力权重: [${data.attention_weights.map(w => w.toFixed(2)).join(', ')}]</p>
                    `;
                }
                break;
            case 'knowledge':
                if (typeof data === 'object') {
                    detailsDiv.innerHTML += `
                        <p>可能动作: ${data.possible_actions.join(', ')}</p>
                        <p>动作价值: [${data.values.map(v => v.toFixed(2)).join(', ')}]</p>
                    `;
                }
                break;
            case 'wisdom':
                if (typeof data === 'object') {
                    detailsDiv.innerHTML += `
                        <p>最佳动作索引: ${data.best_action_index}</p>
                        <p>决策置信度: ${(data.confidence * 100).toFixed(1)}%</p>
                    `;
                }
                break;
            case 'decision':
                detailsDiv.innerHTML += `
                    <p>选择的动作: ${data}</p>
                    <p>动作含义: ${this.getActionMeaning(data)}</p>
                `;
                break;
            case 'action':
                if (typeof data === 'object') {
                    detailsDiv.innerHTML += `
                        <p>执行动作: ${data.action}</p>
                        <p>新位置: (${data.next_state[0]}, ${data.next_state[1]})</p>
                        <p>奖励: ${data.reward.toFixed(2)}</p>
                        <p>是否结束: ${data.done ? '是' : '否'}</p>
                    `;
                } else {
                    detailsDiv.innerHTML += `<p>${data}</p>`;
                }
                break;
            default:
                detailsDiv.innerHTML += `<pre>${JSON.stringify(data, null, 2)}</pre>`;
        }
    }
    
    getActionMeaning(action) {
        switch(parseInt(action)) {
            case 0: return "上";
            case 1: return "下";
            case 2: return "左";
            case 3: return "右";
            default: return "未知";
        }
    }
} 