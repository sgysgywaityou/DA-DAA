import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class GatingMechanism(nn.Module):
    """门控机制"""

    def __init__(self, feature_dim: int):
        super(GatingMechanism, self).__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )

    def forward(self, feat_high: torch.Tensor, feat_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # 拼接特征
        combined = torch.cat([feat_high, feat_low], dim=-1)

        # 计算门控权重
        gate_weight = self.gate_net(combined)

        # 应用门控
        feat_high_gated = feat_high * gate_weight
        feat_low_gated = feat_low * gate_weight

        return feat_high_gated, feat_low_gated


class GMMD(nn.Module):
    """门控最大均值差异(GMMD)"""

    def __init__(self, kernel_type: str = 'rbf', sigma: float = 1.0):
        super(GMMD, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def _rbf_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """RBF核函数"""
        x_norm = (x ** 2).sum(1).view(-1, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.exp(-dist / (2 * self.sigma ** 2))

    def _linear_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """线性核函数"""
        return torch.mm(x, y.t())

    def forward(self,
                feat_high: torch.Tensor,
                feat_low: torch.Tensor,
                gate_weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 应用门控权重
        if gate_weight is not None:
            feat_high = feat_high * gate_weight
            feat_low = feat_low * gate_weight

        # 选择核函数
        if self.kernel_type == 'rbf':
            kernel = self._rbf_kernel
        else:
            kernel = self._linear_kernel

        # 计算MMD
        K_high_high = kernel(feat_high, feat_high)
        K_low_low = kernel(feat_low, feat_low)
        K_high_low = kernel(feat_high, feat_low)

        n_high = feat_high.size(0)
        n_low = feat_low.size(0)

        mmd = (K_high_high.sum() / (n_high * n_high) +
               K_low_low.sum() / (n_low * n_low) -
               2 * K_high_low.sum() / (n_high * n_low))

        return mmd


class GLMMD(nn.Module):
    """门控局部最大均值差异(GLMMD)"""

    def __init__(self, kernel_type: str = 'rbf', sigma: float = 1.0):
        super(GLMMD, self).__init__()
        self.kernel_type = kernel_type
        self.sigma = sigma

    def _compute_sample_weights(self,
                                labels: torch.Tensor,
                                num_classes: int) -> torch.Tensor:
        """计算样本权重（基于标签分布）"""
        batch_size = labels.size(0)
        weights = torch.zeros(batch_size, device=labels.device)

        for c in range(num_classes):
            class_mask = labels[:, c] > 0
            if class_mask.sum() > 0:
                weights[class_mask] = 1.0 / class_mask.sum().float()

        return weights / weights.sum()  # 归一化

    def forward(self,
                feat_high: torch.Tensor,
                feat_low: torch.Tensor,
                labels_high: torch.Tensor,
                labels_low: torch.Tensor,
                gate_weight: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 应用门控权重
        if gate_weight is not None:
            feat_high = feat_high * gate_weight
            feat_low = feat_low * gate_weight

        num_classes = labels_high.size(1)
        total_lmmd = 0.0

        # 对每个类别计算LMMD
        for c in range(num_classes):
            # 获取当前类别的样本
            high_mask = labels_high[:, c] > 0
            low_mask = labels_low[:, c] > 0

            if high_mask.sum() > 0 and low_mask.sum() > 0:
                feat_high_c = feat_high[high_mask]
                feat_low_c = feat_low[low_mask]

                # 计算类别权重
                n_high_c = feat_high_c.size(0)
                n_low_c = feat_low_c.size(0)

                # 使用RBF核（可根据需要扩展）
                def rbf_kernel(x, y):
                    x_norm = (x ** 2).sum(1).view(-1, 1)
                    y_norm = (y ** 2).sum(1).view(1, -1)
                    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
                    return torch.exp(-dist / (2 * self.sigma ** 2))

                K_high_high = rbf_kernel(feat_high_c, feat_high_c)
                K_low_low = rbf_kernel(feat_low_c, feat_low_c)
                K_high_low = rbf_kernel(feat_high_c, feat_low_c)

                # 计算当前类别的LMMD
                lmmd_c = (K_high_high.sum() / (n_high_c * n_high_c) +
                          K_low_low.sum() / (n_low_c * n_low_c) -
                          2 * K_high_low.sum() / (n_high_c * n_low_c))

                total_lmmd += lmmd_c

        # 平均LMMD
        return total_lmmd / num_classes


class DomainAlignmentModule(nn.Module):
    """域对齐模块：整合GMMD和GLMMD"""

    def __init__(self,
                 feature_dim: int,
                 use_gmmd: bool = True,
                 use_glmmd: bool = True,
                 kernel_type: str = 'rbf',
                 sigma: float = 1.0):

        super(DomainAlignmentModule, self).__init__()
        self.use_gmmd = use_gmmd
        self.use_glmmd = use_glmmd

        # 门控机制
        self.gating = GatingMechanism(feature_dim)

        # GMMD
        if use_gmmd:
            self.gmmd = GMMD(kernel_type=kernel_type, sigma=sigma)

        # GLMMD
        if use_glmmd:
            self.glmmd = GLMMD(kernel_type=kernel_type, sigma=sigma)

    def forward(self,
                feat_high: torch.Tensor,
                feat_low: torch.Tensor,
                labels_high: torch.Tensor,
                labels_low: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        # 应用门控机制
        feat_high_gated, feat_low_gated = self.gating(feat_high, feat_low)

        # 计算门控权重（用于后续损失计算）
        combined = torch.cat([feat_high, feat_low], dim=-1)
        gate_weight = self.gating.gate_net(combined)

        # 计算域对齐损失
        gmmd_loss = torch.tensor(0.0, device=feat_high.device)
        glmmd_loss = torch.tensor(0.0, device=feat_high.device)

        if self.use_gmmd:
            gmmd_loss = self.gmmd(feat_high_gated, feat_low_gated, gate_weight)

        if self.use_glmmd:
            glmmd_loss = self.glmmd(feat_high_gated, feat_low_gated,
                                    labels_high, labels_low, gate_weight)

        return gmmd_loss, glmmd_loss