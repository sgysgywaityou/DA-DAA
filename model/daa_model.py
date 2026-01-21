import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from typing import Tuple, Dict, Optional

from .attention import LabelAttention, CollaborativeAttention
from .domain_align import DomainAlignmentModule


class FeatureExtractor(nn.Module):
    """特征提取器：BERT + BiGRU"""

    def __init__(self,
                 bert_model: str = 'bert-base-uncased',
                 hidden_dim: int = 256,
                 gru_layers: int = 2,
                 dropout: float = 0.1):
        super(FeatureExtractor, self).__init__()

        # BERT编码器
        self.bert = BertModel.from_pretrained(bert_model)
        bert_dim = self.bert.config.hidden_size

        # BiGRU
        self.gru = nn.GRU(
            input_size=bert_dim,
            hidden_size=hidden_dim // 2,  # 双向所以一半
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0
        )

        # 适配层
        self.adapter = nn.Sequential(
            nn.Linear(bert_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.hidden_dim = hidden_dim

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # BERT编码
        bert_output = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )

        # 获取最后隐藏状态
        last_hidden = bert_output.last_hidden_state  # [batch, seq_len, bert_dim]

        # 适配层
        adapted = self.adapter(last_hidden)  # [batch, seq_len, hidden_dim]

        # BiGRU
        gru_output, _ = self.gru(adapted)  # [batch, seq_len, hidden_dim]

        return gru_output


class DA_DAA_Model(nn.Module):
    """完整的DA-DAA模型"""

    def __init__(self,
                 num_labels: int,
                 bert_model: str = 'bert-base-uncased',
                 hidden_dim: int = 256,
                 gru_layers: int = 2,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 kl_gamma_high: float = 0.1,
                 kl_gamma_low: float = 0.5,
                 use_uniform_target_low: bool = True,
                 use_gmmd: bool = True,
                 use_glmmd: bool = True):

        super(DA_DAA_Model, self).__init__()

        self.num_labels = num_labels
        self.hidden_dim = hidden_dim

        # 特征提取器（共享权重）
        self.feature_extractor = FeatureExtractor(
            bert_model=bert_model,
            hidden_dim=hidden_dim,
            gru_layers=gru_layers,
            dropout=dropout
        )

        # 标签注意力
        self.label_attention = LabelAttention(
            hidden_dim=hidden_dim,
            num_labels=num_labels,
            dropout=dropout
        )

        # 协作注意力（DAA）
        self.collaborative_attention = CollaborativeAttention(
            hidden_dim=hidden_dim,
            num_heads=num_heads
        )

        # 域对齐模块
        self.domain_alignment = DomainAlignmentModule(
            feature_dim=hidden_dim,
            use_gmmd=use_gmmd,
            use_glmmd=use_glmmd
        )

        # 分类器
        self.classifier_high = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )

        self.classifier_low = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_labels)
        )

        self.classifier_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels)
        )

        # 池化层
        self.pooling = nn.AdaptiveAvgPool1d(1)

        # 初始化参数
        self._init_weights()

    def _init_weights(self):
        """初始化权重"""
        for module in [self.classifier_high, self.classifier_low, self.classifier_fuse]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self,
                input_ids_text: torch.Tensor,
                attention_mask_text: torch.Tensor,
                input_ids_title: torch.Tensor,
                attention_mask_title: torch.Tensor,
                is_low_freq: torch.Tensor,
                hf_mask: Optional[torch.Tensor] = None,
                lf_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:

        batch_size = input_ids_text.size(0)

        # 1. 特征提取
        text_features = self.feature_extractor(input_ids_text, attention_mask_text)
        title_features = self.feature_extractor(input_ids_title, attention_mask_title)

        # 2. 标签注意力
        text_label_att, _ = self.label_attention(text_features)
        title_label_att, _ = self.label_attention(title_features)

        # 3. 协作注意力（DAA）
        fused_title, fused_content = self.collaborative_attention(
            title_features=title_label_att,
            content_features=text_label_att,
            is_low_freq=is_low_freq.any()  # 批次中是否有低频样本
        )

        # 4. 池化得到句子表示
        title_rep = fused_title.mean(dim=1)  # [batch, hidden]
        content_rep = fused_content.mean(dim=1)  # [batch, hidden]

        # 5. 拼接特征
        combined_rep = torch.cat([title_rep, content_rep], dim=-1)  # [batch, hidden*2]

        # 6. 分类预测
        pred_high = torch.sigmoid(self.classifier_high(combined_rep))
        pred_low = torch.sigmoid(self.classifier_low(combined_rep))
        pred_fuse = torch.sigmoid(self.classifier_fuse(combined_rep))