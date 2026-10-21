import torch
import torch.nn as nn


class TransformerScorer(nn.Module):
    """Lightweight Transformer scorer for visual tokens."""

    def __init__(
        self,
        in_features: int,
        hidden_dim: int = 1792,
        init_scale: float = 0.0001,
    ):
        super().__init__()
        self.in_features = in_features#输入维度
        self.hidden_dim = hidden_dim#输出维度

        # Lightweight projection layers for Key and Query.
        self.k_proj = nn.Linear(in_features, hidden_dim)
        self.q_proj = nn.Linear(in_features, hidden_dim)


        self._init_near_zero(init_scale)

    def _init_near_zero(self, scale: float = 0.0001):
        """Initialize parameters with small values to ensure near-zero output."""
        nn.init.normal_(self.k_proj.weight, std=scale)
        nn.init.zeros_(self.k_proj.bias)

        nn.init.normal_(self.q_proj.weight, std=scale)
        nn.init.zeros_(self.q_proj.bias)

    def forward(
        self,
        x: torch.Tensor,                            # 输入的视觉 Tokens [Batch_Size, Seq_Len, Vis_Dim]
    ) -> torch.Tensor:
        """
        前向传播逻辑：
        1. 计算视觉自身的 Query (q) 和 Key (k)。
        2. 计算最终的 Attention 分数，取平均值作为 Token 的重要性分数。
        """

        # 1. 基础投影：计算视觉自身的 K 和 Q
        k = self.k_proj(x) # [B, S, H]
        q = self.q_proj(x) # [B, S, H]

        # 2. 计算最终分数 (Score Calculation)
        # 使用 Self-Attention 机制：Score = Q * K^T
        # 这里计算的是每个 Token 与所有其他 Token 的关联程度
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.hidden_dim ** 0.5)
        
        # 对最后一个维度求平均，得到每个 Token 的全局重要性分数
        # Shape: [Batch, Seq_Len]
        scores = attn_weights.mean(dim=-1)

        return scores


if __name__ == '__main__':
    # Test Basic Functionality
    model = TransformerScorer(in_features=3584)
    x = torch.randn(1,220,3584)
    y = model(x)
    print("Basic Output:", y.shape)
