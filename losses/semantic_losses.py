# Copyright (c) Facebook, Inc. and its affiliates.
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabCE(nn.Module):
    """
    Hard pixel mining with cross entropy loss, for semantic segmentation.
    This is used in TensorFlow DeepLab frameworks.
    Paper: DeeperLab: Single-Shot Image Parser
    Reference: https://github.com/tensorflow/models/blob/bd488858d610e44df69da6f89277e9de8a03722c/research/deeplab/utils/train_utils.py#L33  # noqa
    Arguments:
        ignore_label: Integer, label to ignore (typically 0 for background)
        top_k_percent_pixels: Float, the value lies in [0.0, 1.0]. When its
            value < 1.0, only compute the loss for the top k percent pixels
        weight: Tensor, a manual rescaling weight given to each class.
    """

    def __init__(self, ignore_label=0, top_k_percent_pixels=0.2, weight=None):
        super(DeepLabCE, self).__init__()
        self.top_k_percent_pixels = top_k_percent_pixels
        self.ignore_label = ignore_label
        self.criterion = nn.CrossEntropyLoss(
            weight=weight, ignore_index=ignore_label, reduction="none"
        )

    def forward(self, logits, labels, weights=None):
        """
        Args:
            logits: (B, C, H, W) tensor of raw logits
            labels: (B, H, W) tensor of class indices (0 is background/ignore)
        """
        if weights is None:
            pixel_losses = self.criterion(logits, labels).contiguous().view(-1)
        else:
            pixel_losses = self.criterion(logits, labels) * weights
            pixel_losses = pixel_losses.contiguous().view(-1)
        
        # Only compute loss for non-ignored pixels
        valid_pixels = labels.ne(self.ignore_label).view(-1)
        if valid_pixels.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
            
        pixel_losses = pixel_losses[valid_pixels]
        if self.top_k_percent_pixels == 1.0:
            return pixel_losses.mean()

        top_k_pixels = int(self.top_k_percent_pixels * pixel_losses.numel())
        pixel_losses, _ = torch.topk(pixel_losses, top_k_pixels)
        return pixel_losses.mean()


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0, ignore_index=0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) tensor of raw logits
            targets: (B, H, W) tensor of class indices (0 is background/ignore)
        """
        num_classes = inputs.shape[1]
        
        # Convert logits to probabilities
        inputs = F.softmax(inputs, dim=1)

        # Create one-hot encoding of targets, but exclude ignore_index
        targets_one_hot = F.one_hot(targets, num_classes=num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()   # (B, C, H, W)

        # Create mask for non-ignored pixels
        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # (B, 1, H, W)

        # Apply mask
        inputs = inputs * valid_mask
        targets_one_hot = targets_one_hot * valid_mask

        # Compute Dice score per class
        dims = (0, 2, 3)  # sum over batch, height, width
        intersection = torch.sum(inputs * targets_one_hot, dims)  # (C,)
        cardinality = torch.sum(inputs + targets_one_hot, dims) + 1e-8  
        
        dice_score = (2. * intersection + self.smooth) / (cardinality + self.smooth)  # (C,)
        # Compute mean Dice score over all classes except background
        dice_loss = 1. - dice_score[1:]  # exclude background class
        
        # Return mean loss over all non-background classes
        return dice_loss.mean()


class FocalLoss(nn.Module):
    """
    Multi-class Focal Loss, as in Lin et al. (2017).
    Args:
      alpha: balancing factor (float or list of per-class floats)
      gamma: focusing parameter >= 0
      ignore_index: label to ignore
    """
    def __init__(self, alpha=None, gamma=2.0, ignore_index=0):
        super().__init__()
        if alpha is None:
            # Initialize alpha with small non-zero weights for background and room
            self.alpha = torch.tensor([0.01, 0.01, 0.32, 0.32, 0.34])  # Small non-zero weights for stability
        elif isinstance(alpha, (list, tuple)):
            self.alpha = torch.tensor(alpha)
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.eps = 1e-6  # Small constant for numerical stability

    def forward(self, logits, targets):
        """
        Args:
            logits: (B, C, H, W) tensor of raw logits from the model
            targets: (B, H, W) tensor of class indices (0 is background/ignore)
        """
        # Move alpha to the same device as logits
        if isinstance(self.alpha, torch.Tensor):
            self.alpha = self.alpha.to(logits.device)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)  # Convert logits to probabilities first
        log_probs = F.log_softmax(logits, dim=1)  # More numerically stable than log(softmax)

        # Flatten all dimensions except channel
        probs_flat = probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])    # (B*H*W, C)
        log_probs_flat = log_probs.permute(0, 2, 3, 1).reshape(-1, logits.shape[1])  # (B*H*W, C)
        targets_flat = targets.view(-1)                                         # (B*H*W)

        # Mask out ignored indices
        valid = (targets_flat != self.ignore_index)
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
            
        targets_flat = targets_flat[valid]
        probs_flat = probs_flat[valid]
        log_probs_flat = log_probs_flat[valid]

        # Get target probabilities
        pt = probs_flat[range(len(probs_flat)), targets_flat]     # (N,)

        # Compute focal loss weight
        focal_weight = (1 - pt) ** self.gamma  # (N,)

        # Apply class balance weights
        if isinstance(self.alpha, torch.Tensor):
            alpha_weight = self.alpha[targets_flat]       # (N,)
            focal_weight = alpha_weight * focal_weight

        # Compute the final loss
        loss = -focal_weight * log_probs_flat[range(len(log_probs_flat)), targets_flat]
        return loss.mean()  


class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=0.5, gamma_focal=0.25, ignore_index=0):
        super(CombinedLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma_focal = gamma_focal
        # Initialize alpha for focal loss with small non-zero weights for stability
        focal_alpha = [0.01, 0.01, 0.32, 0.32, 0.34]  # Small non-zero weights for stability
        
        self.hard_mining_loss = DeepLabCE(ignore_label=ignore_index, top_k_percent_pixels=0.2)
        self.dice_loss = DiceLoss(ignore_index=ignore_index)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=2.0, ignore_index=ignore_index)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (B, C, H, W) tensor of raw logits
            targets: (B, H, W) tensor of class indices (0 is background/ignore)
        """
        
        ce = self.hard_mining_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        focal = self.focal_loss(inputs, targets)

        # Combine losses with clamping to prevent extreme values
        total_loss = self.alpha * ce + \
                    self.beta *dice + \
                    self.gamma_focal * focal
        
        return total_loss  # Return the total loss