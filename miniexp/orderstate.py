import numpy as np
from typing import Dict, Any, Tuple, List, Optional, Union
import time

class OrderStateManager:
    """
    八阶段序态循环管理器
    负责协调 能量→信号→数据→信息→知识→智慧→决策→动作→能量 的完整循环流程
    """
    
    # 定义八个阶段
    STAGES = ['energy', 'signal', 'data', 'information', 'knowledge', 'wisdom', 'decision', 'action']
    
    def __init__(self):
        """
        初始化序态循环管理器
        """
        # 初始化各阶段状态存储
        self.states = {stage: None for stage in self.STAGES}
        
        # 记录当前活跃阶段
        self.current_stage = None
        
        # 历史记录
        self.history = []
        
        # 计时器，用于记录各阶段处理时间
        self.timers = {stage: 0.0 for stage in self.STAGES}
        
        # 循环计数
        self.cycle_count = 0
        
        # 状态变更回调函数
        self.state_change_callbacks = []
    
    def register_callback(self, callback_func):
        """
        注册状态变更回调函数
        
        Args:
            callback_func: 回调函数，接收参数(stage, value)
        """
        self.state_change_callbacks.append(callback_func)
    
    def update_state(self, stage: str, value: Any) -> None:
        """
        更新指定阶段的状态
        
        Args:
            stage: 阶段名称，必须是STAGES中的一个
            value: 该阶段的新状态值
        """
        if stage not in self.STAGES:
            raise ValueError(f"无效的阶段名称: {stage}。必须是以下之一: {self.STAGES}")
        
        # 记录开始时间
        start_time = time.time()
        
        # 更新状态
        self.states[stage] = value
        self.current_stage = stage
        
        # 记录处理时间
        self.timers[stage] = time.time() - start_time
        
        # 添加到历史记录
        self.history.append({
            'stage': stage,
            'value': value,
            'timestamp': time.time()
        })
        
        # 调用所有回调函数
        for callback in self.state_change_callbacks:
            callback(stage, value)
        
        # 检查是否完成一个完整循环
        if stage == 'action' and self._is_cycle_complete():
            self.cycle_count += 1
            print(f"完成第 {self.cycle_count} 个序态循环")
    
    def _is_cycle_complete(self) -> bool:
        """
        检查是否完成了一个完整循环
        
        Returns:
            bool: 是否完成了一个完整循环
        """
        # 检查所有阶段是否都有值
        return all(self.states[stage] is not None for stage in self.STAGES)
    
    def get_current_state(self) -> Dict[str, Any]:
        """
        获取当前的完整状态
        
        Returns:
            Dict[str, Any]: 包含所有阶段当前状态的字典
        """
        result = {
            'states': self.states.copy(),
            'current_stage': self.current_stage,
            'cycle_count': self.cycle_count,
            'timers': self.timers.copy()
        }
        return result
    
    def get_stage_state(self, stage: str) -> Any:
        """
        获取指定阶段的当前状态
        
        Args:
            stage: 阶段名称
            
        Returns:
            Any: 该阶段的当前状态值
        """
        if stage not in self.STAGES:
            raise ValueError(f"无效的阶段名称: {stage}")
        return self.states[stage]
    
    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        获取历史记录
        
        Args:
            limit: 返回的最大记录数，默认为10
            
        Returns:
            List[Dict[str, Any]]: 历史记录列表
        """
        return self.history[-limit:] if limit > 0 else self.history.copy()
    
    def reset(self) -> None:
        """
        重置序态循环管理器
        """
        self.states = {stage: None for stage in self.STAGES}
        self.current_stage = None
        self.history = []
        self.timers = {stage: 0.0 for stage in self.STAGES}
        self.cycle_count = 0
    
    def get_analytics(self) -> Dict[str, Any]:
        """
        获取序态循环的分析数据
        
        Returns:
            Dict[str, Any]: 序态循环分析数据
        """
        if not self.history:
            return {'message': '暂无数据'}
        
        # 计算各阶段平均处理时间
        stage_times = {stage: [] for stage in self.STAGES}
        for record in self.history:
            stage = record['stage']
            if 'processing_time' in record:
                stage_times[stage].append(record['processing_time'])
        
        avg_times = {stage: np.mean(times) if times else 0.0 
                    for stage, times in stage_times.items()}
        
        # 计算状态转换频率
        transitions = {}
        for i in range(1, len(self.history)):
            prev_stage = self.history[i-1]['stage']
            curr_stage = self.history[i]['stage']
            key = f"{prev_stage}->{curr_stage}"
            transitions[key] = transitions.get(key, 0) + 1
        
        return {
            'avg_processing_times': avg_times,
            'transitions': transitions,
            'cycle_count': self.cycle_count,
            'total_updates': len(self.history)
        }
        
# 全局单例实例，方便其他模块访问
order_state_manager = OrderStateManager() 