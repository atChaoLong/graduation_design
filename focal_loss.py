import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for multi-label classification.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    Designed to handle class imbalance by down-weighting easy examples.
    """

    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        # logits: (batch, num_labels) - raw unnormalized scores
        # targets: (batch, num_labels) - binary labels
        probs = torch.sigmoid(logits)

        # Compute focal weights
        # p_t = targets * probs + (1 - targets) * (1 - probs)
        p_t = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - p_t) ** self.gamma

        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')

        # Apply focal weighting
        focal_loss = self.alpha * focal_weight * bce

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
