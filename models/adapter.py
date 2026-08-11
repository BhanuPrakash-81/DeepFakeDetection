import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class SupervisedContrastiveLoss(nn.Module):
    """
    Supervised Contrastive Loss (SupCon).
    Forces the network to separate real physical features from synthetic ones by pulling
    embeddings of the same class together while pushing different classes apart in 128D space.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        features: (N, proj_dim) L2-normalized embeddings
        labels: (N,) or (N, 1) binary labels (0 for REAL, 1 for FAKE)
        """
        device = features.device
        batch_size = features.shape[0]
        if batch_size <= 1 or torch.unique(labels).numel() <= 1:
            return torch.tensor(0.0, device=device, requires_grad=True)

        labels = labels.view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # L2-normalize features
        features_norm = F.normalize(features, dim=1)
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature

        # For numerical stability
        logits_max, _ = torch.max(similarity_matrix, dim=1, keepdim=True)
        logits = similarity_matrix - logits_max.detach()

        # Mask out self-contrast (diagonal entries)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 1,
            torch.arange(batch_size).view(-1, 1).to(device), 0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-8)
        loss = - (self.temperature / 0.07) * mean_log_prob_pos
        loss = loss.mean()
        return loss

class CombinedSupConBCELoss(nn.Module):
    """
    Combined Loss Function:
    Calculates Supervised Contrastive Loss + Binary Cross Entropy (BCE) Loss
    Total Loss = BCE_Loss + alpha * SupCon_Loss
    Supports dynamic pos_weight for handling extreme class imbalances.
    """
    def __init__(self, alpha: float = 0.5, temperature: float = 0.07, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.supcon_loss = SupervisedContrastiveLoss(temperature=temperature)

    def forward(self, proj_features: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_bce = self.bce_loss(logits, labels.float())
        loss_supcon = self.supcon_loss(proj_features, labels)
        total_loss = loss_bce + self.alpha * loss_supcon
        return total_loss, loss_bce, loss_supcon

class LightweightAnatomicalAdapter(nn.Module):
    """
    PEFT Lightweight Anatomical Adapter PyTorch Module.
    Concatenates Stage 1 1D multi-modal features and processes them through an MLP fusion block.
    Outputs:
        1. Low-dimensional L2-normalized embedding (128D) for Contrastive Loss.
        2. Final binary logit (1D) for BCE classification.
    """
    def __init__(self, input_dim: int = 9281, hidden_dim: int = 256, proj_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        # Layer 1: Feature Compression & Normalization
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        # Layer 2: Intermediate Fusion Representation
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        # Output 1: Low-dimensional Projection Head for Contrastive Loss
        self.proj_head = nn.Linear(hidden_dim, proj_dim)

        # Output 2: Final Binary Classification Head
        self.cls_head = nn.Linear(proj_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        h = self.drop1(self.act1(self.bn1(self.fc1(x))))
        h = self.drop2(self.act2(self.bn2(self.fc2(h))))

        # Output 1: L2-normalized contrastive projection vector
        proj_feat = F.normalize(self.proj_head(h), dim=1)

        # Output 2: Binary classification logit
        logits = self.cls_head(proj_feat).squeeze(-1)

        return proj_feat, logits
