import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MultiLabelLoss(nn.Module):
    """多标签分类损失"""

    def __init__(self,
                 mu_high: float = 0.4,
                 sigma_low: float = 0.4,
                 psi_fuse: float = 0.2):

        super(MultiLabelLoss, self).__init__()
        self.mu_high = mu_high
        self.sigma_low = sigma_low
        self.psi_fuse = psi_fuse

        # BCE损失函数
        self.bce_loss = nn.BCELoss(reduction='none')

    def forward(self,
                pred_high: torch.Tensor,
                pred_low: torch.Tensor,
                pred_fuse: torch.Tensor,
                labels: torch.Tensor,
                hf_mask: torch.Tensor,
                lf_mask: torch.Tensor,
                is_low_freq: torch.Tensor) -> Tuple[torch.Tensor, dict]:

        batch_size = labels.size(0)
        device = labels.device

        # 分离高频和低频标签
        labels_high = labels[:, hf_mask]
        labels_low = labels[:, lf_mask]

        # 分离高频和低频样本
        high_sample_mask = ~is_low_freq
        low_sample_mask = is_low_freq

        # 高频样本损失
        loss_high = torch.tensor(0.0, device=device)
        if high_sample_mask.sum() > 0:
            pred_high_samples = pred_high[high_sample_mask]
            labels_high_samples = labels_high[high_sample_mask]

            loss_high = self.bce_loss(pred_high_samples, labels_high_samples)
            loss_high = loss_high.mean() if loss_high.dim() > 0 else loss_high

        # 低频样本损失
        loss_low = torch.tensor(0.0, device=device)
        if low_sample_mask.sum() > 0:
            pred_low_samples = pred_low[low_sample_mask]
            labels_low_samples = labels_low[low_sample_mask]

            loss_low = self.bce_loss(pred_low_samples, labels_low_samples)
            loss_low = loss_low.mean() if loss_low.dim() > 0 else loss_low

        # 融合损失
        loss_fuse = self.bce_loss(pred_fuse, labels)
        loss_fuse = loss_fuse.mean() if loss_fuse.dim() > 0 else loss_fuse

        # 加权总损失
        total_loss = (self.mu_high * loss_high +
                      self.sigma_low * loss_low +
                      self.psi_fuse * loss_fuse)

        loss_dict = {
            'loss_high': loss_high.item() if isinstance(loss_high, torch.Tensor) else loss_high,
            'loss_low': loss_low.item() if isinstance(loss_low, torch.Tensor) else loss_low,
            'loss_fuse': loss_fuse.item() if isinstance(loss_fuse, torch.Tensor) else loss_fuse,
            'loss_total': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        }

        return total_loss, loss_dict


class TotalLoss(nn.Module):
    """总损失：分类损失 + 域对齐损失"""

    def __init__(self,
                 mu_high: float = 0.4,
                 sigma_low: float = 0.4,
                 psi_fuse: float = 0.2,
                 zeta_mmd: float = 0.5,
                 phi_lmmd: float = 0.5):
        super(TotalLoss, self).__init__()
        self.mu_high = mu_high
        self.sigma_low = sigma_low
        self.psi_fuse = psi_fuse
        self.zeta_mmd = zeta_mmd
        self.phi_lmmd = phi_lmmd

        self.class_loss = MultiLabelLoss(mu_high, sigma_low, psi_fuse)

    def forward(self,
                pred_high: torch.Tensor,
                pred_low: torch.Tensor,
                pred_fuse: torch.Tensor,
                labels: torch.Tensor,
                hf_mask: torch.Tensor,
                lf_mask: torch.Tensor,
                is_low_freq: torch.Tensor,
                gmmd_loss: torch.Tensor,
                glmmd_loss: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        # 分类损失
        class_loss, loss_dict = self.class_loss(
            pred_high, pred_low, pred_fuse,
            labels, hf_mask, lf_mask, is_low_freq
        )

        # 域对齐损失
        domain_loss = (self.zeta_mmd * gmmd_loss +
                       self.phi_lmmd * glmmd_loss)

        # 总损失
        total_loss = class_loss + domain_loss

        # 更新损失字典
        loss_dict.update({
            'gmmd_loss': gmmd_loss.item() if isinstance(gmmd_loss, torch.Tensor) else gmmd_loss,
            'glmmd_loss': glmmd_loss.item() if isinstance(glmmd_loss, torch.Tensor) else glmmd_loss,
            'domain_loss': domain_loss.item() if isinstance(domain_loss, torch.Tensor) else domain_loss,
            'total_loss': total_loss.item() if isinstance(total_loss, torch.Tensor) else total_loss
        })

        return total_loss, loss_dict