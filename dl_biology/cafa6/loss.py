import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, ia_weight, alpha=None, gamma=3.0, pos_weight=None, reduction="mean"):
        """
        Args:
            alpha: Weighting factor for positive class (0-1)
            gamma: Focusing parameter (typically 2.0)
            pos_weight: Per-class weight for positive examples, shape [num_classes]
                       Typically: neg_count / pos_count for each class
            reduction: 'mean', 'sum', or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.reduction = reduction
        self.ia_weight = ia_weight

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits from model (before sigmoid), shape [batch, num_classes]
            targets: Binary labels, shape [batch, num_classes]
        """
        # Apply sigmoid to get probabilities
        p = torch.sigmoid(inputs)

        # Calculate binary cross-entropy with pos_weight
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, reduction="none", pos_weight=self.pos_weight  # Added here
        )

        # Calculate focal weight: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha weighting (optional)
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight

        # Final focal loss
        ia_weight_expanded = self.ia_weight.unsqueeze(0)
        focal_loss = focal_weight * bce_loss * ia_weight_expanded

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss
