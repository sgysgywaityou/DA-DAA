import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class LabelAttention(nn.Module):
    """标签注意力机制"""

    def __init__(self, hidden_dim: int, num_labels: int, dropout: float = 0.1):
        super(LabelAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels

        # 标签嵌入
        self.label_embeddings = nn.Embedding(num_labels, hidden_dim)

        # 注意力层
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # 缩放因子
        self.scale = math.sqrt(hidden_dim)

    def forward(self,
                text_features: torch.Tensor,  # [batch, seq_len, hidden]
                label_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, hidden_dim = text_features.shape

        # 获取标签嵌入
        if label_ids is None:
            # 使用所有标签
            label_ids = torch.arange(self.num_labels, device=text_features.device)

        label_embeds = self.label_embeddings(label_ids)  # [num_labels, hidden]

        # 计算注意力
        Q = self.query_proj(text_features)  # [batch, seq_len, hidden]
        K = self.key_proj(label_embeds).unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_labels, hidden]
        V = self.value_proj(label_embeds).unsqueeze(0).expand(batch_size, -1, -1)  # [batch, num_labels, hidden]

        # 注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # [batch, seq_len, num_labels]
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 上下文向量
        context = torch.matmul(attention_weights.transpose(-2, -1), text_features)  # [batch, num_labels, hidden]

        return context, attention_weights


class DivergenceAlignedAttention(nn.Module):
    """分布对齐注意力(DAA)"""

    def __init__(self,
                 hidden_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 kl_gamma: float = 0.1,
                 use_uniform_target: bool = False):

        super(DivergenceAlignedAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.kl_gamma = kl_gamma
        self.use_uniform_target = use_uniform_target

        # 多头注意力投影
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                is_low_freq: bool = False) -> torch.Tensor:

        batch_size, seq_len_q, _ = query.shape
        _, seq_len_k, _ = key.shape

        # 线性投影
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        # 多头划分
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)  # [batch, heads, seq_len_q, seq_len_k]

        # KL散度约束的目标分布
        if self.use_uniform_target and is_low_freq:
            # 低频：均匀分布
            target_dist = torch.ones_like(scores) / seq_len_k
        else:
            # 高频：均值池化作为目标分布
            target_dist = torch.mean(scores, dim=-1, keepdim=True)  # [batch, heads, seq_len_q, 1]
            target_dist = target_dist.expand_as(scores)

        # KL散度约束项
        kl_term = self.kl_gamma * torch.log(target_dist + 1e-10)
        scores = scores + kl_term

        # Softmax归一化
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 上下文向量
        context = torch.matmul(attention_weights, V)  # [batch, heads, seq_len_q, head_dim]

        # 合并多头
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.hidden_dim)

        # 输出投影
        output = self.out_proj(context)

        return output


class CollaborativeAttention(nn.Module):
    """协作注意力：双向标题-内容融合"""

    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super(CollaborativeAttention, self).__init__()

        # 标题到内容的注意力
        self.title2content = DivergenceAlignedAttention(hidden_dim, num_heads)

        # 内容到标题的注意力
        self.content2title = DivergenceAlignedAttention(hidden_dim, num_heads)

        # 门控融合
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,
                title_features: torch.Tensor,
                content_features: torch.Tensor,
                is_low_freq: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # 标题→内容
        title2content = self.title2content(
            query=content_features,
            key=title_features,
            value=title_features,
            is_low_freq=is_low_freq
        )

        # 内容→标题
        content2title = self.content2title(
            query=title_features,
            key=content_features,
            value=content_features,
            is_low_freq=is_low_freq
        )

        # 门控融合
        gate_t = self.gate(torch.cat([title_features, title2content], dim=-1))
        gate_c = self.gate(torch.cat([content_features, content2title], dim=-1))

        fused_title = gate_t * title_features + (1 - gate_t) * title2content
        fused_content = gate_c * content_features + (1 - gate_c) * content2title

        # 层归一化
        fused_title = self.layer_norm(fused_title)
        fused_content = self.layer_norm(fused_content)

        return fused_title, fused_content