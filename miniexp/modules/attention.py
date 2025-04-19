import numpy as np

class AttentionModule:
    """
    计算注意力权重模块。
    根据输入特征和焦虑度计算注意力分布。
    焦虑度越高，注意力分布越集中（尖锐）。
    """
    def __init__(self, feature_dim: int, hidden_dim: int = 32):
        """
        初始化注意力模块。

        Args:
            feature_dim (int): 输入特征维度。
            hidden_dim (int, optional): 隐藏层维度，默认为32。
        """
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        
        # 为了简化实现，我们使用一个简单的线性层来将特征投影到动作空间
        # 在实际应用中，这可能是一个更复杂的神经网络
        # 随机初始化权重
        self.W = np.random.randn(feature_dim, hidden_dim) * 0.1
        self.V = np.random.randn(hidden_dim, 4) * 0.1  # 4个动作的权重
        
    def compute_attention(self, features: np.ndarray, anxiety: float = 0.0) -> np.ndarray:
        """
        计算注意力权重。

        Args:
            features (np.ndarray): 输入特征，形状为 (feature_dim,)。
            anxiety (float, optional): 焦虑度，范围通常在0到2之间。默认为0.0。

        Returns:
            np.ndarray: 注意力权重，形状为 (4,)，对应四个动作的概率分布。
        """
        # [阶段 3 -> 4: 数据(特征) -> 信息(注意力权重)]
        
        # 简单前向传播计算
        hidden = np.tanh(features.dot(self.W))
        logits = hidden.dot(self.V)
        
        # --- 生存焦虑驱动Attention ---
        # 焦虑影响注意力分布的尖锐程度（温度参数）
        # 焦虑越高，温度越低，分布越尖锐（集中）
        temperature = 1.0 / (1.0 + anxiety)
        
        # 防止数值问题
        max_logit = np.max(logits)
        logits = logits - max_logit
        
        # 应用温度缩放后的softmax
        scaled_logits = logits / max(temperature, 0.1)  # 避免除以0
        exp_logits = np.exp(scaled_logits)
        weights = exp_logits / np.sum(exp_logits)
        
        return weights
    
    def _softmax(self, x, temperature=1.0):
        """
        计算softmax，带温度参数。
        温度越低，分布越尖锐。

        Args:
            x (np.ndarray): 输入数组。
            temperature (float): 温度参数，默认为1.0。

        Returns:
            np.ndarray: softmax结果。
        """
        x = x / max(temperature, 1e-8)  # 避免除以0
        exp_x = np.exp(x - np.max(x))  # 减去最大值以避免溢出
        return exp_x / np.sum(exp_x)

# 测试代码
if __name__ == '__main__':
    # 创建注意力模块实例
    attn_module = AttentionModule(feature_dim=2)
    
    # 测试不同焦虑度下的注意力分布
    feature = np.array([1.0, 2.0])  # 示例特征
    
    print("低焦虑状态下的注意力分布:")
    attention_low = attn_module.compute_attention(feature, anxiety=0.1)
    print(attention_low)
    
    print("\n中等焦虑状态下的注意力分布:")
    attention_med = attn_module.compute_attention(feature, anxiety=1.0)
    print(attention_med)
    
    print("\n高焦虑状态下的注意力分布:")
    attention_high = attn_module.compute_attention(feature, anxiety=2.0)
    print(attention_high) 