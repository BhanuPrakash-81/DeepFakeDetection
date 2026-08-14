import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional, List
from sklearn.metrics import roc_auc_score

from utils.paths import get_features_output_path, get_checkpoint_path, resolve_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelAndTrain")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ==========================================
# 1. Early Stopping Utility (Validation Loss & AUC-ROC)
# ==========================================
class EarlyStoppingAUC:
    """
    Upgraded Early Stopping mechanism monitoring Validation AUC-ROC.
    Stops training if Validation AUC-ROC does not improve after `patience` consecutive epochs.
    Mode is set to 'max' (higher AUC is better).
    """
    def __init__(self, patience: int = 15, min_delta: float = 1e-4, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_auc = -float("inf") if mode == "max" else float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_auc: float, epoch: int) -> bool:
        if self.mode == "max":
            improved = val_auc > self.best_auc + self.min_delta
        else:
            improved = val_auc < self.best_auc - self.min_delta

        if improved:
            self.best_auc = val_auc
            self.best_epoch = epoch
            self.counter = 0
            return True # Improved best metric
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best Val AUC: {self.best_auc:.4f} at Epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False # No improvement


class CB_FocalLoss(nn.Module):
    """
    Class-Balanced Focal Loss (Cui et al., CVPR 2019).
    Calculates class weights based on the effective number of samples:
        E_n = (1 - beta^N) / (1 - beta)
        weight_i = (1 - beta) / (1 - beta^N_i)
        alpha_i = weight_i / sum(weight) * num_classes
    """
    def __init__(self, samples_per_class: List[int], beta: float = 0.999, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.reduction = reduction

        # Calculate effective number of samples per class
        effective_num = 1.0 - np.power(beta, samples_per_class)
        weights = (1.0 - beta) / np.array(effective_num, dtype=np.float32)
        weights = weights / np.sum(weights) * len(samples_per_class)

        # Register class weights buffer [alpha_0 (REAL), alpha_1 (FAKE)]
        self.register_buffer("class_weights", torch.tensor(weights, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.class_weights[1] * targets + self.class_weights[0] * (1.0 - targets)

        cb_focal_loss = alpha_t * ((1.0 - p_t) ** self.gamma) * bce_loss

        if self.reduction == "mean":
            return cb_focal_loss.mean()
        elif self.reduction == "sum":
            return cb_focal_loss.sum()
        return cb_focal_loss


class FocalLoss(nn.Module):
    """
    Custom Focal Loss Class for handling severe class imbalance in deepfake detection.
    FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
    Numerical stability is ensured via F.binary_cross_entropy_with_logits.
    """
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
        focal_loss = alpha_t * (1.0 - p_t) ** self.gamma * bce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss

# Aliases for backwards compatibility
BinaryFocalLoss = FocalLoss
EarlyStopping = EarlyStoppingAUC


# ==========================================
# 2. Modality Feature Dropout & Multimodal Dynamic Adapter
# ==========================================
class FeatureChannelDropout(nn.Module):
    """
    Randomly zeroes out entire modality feature channels during training (p=0.3).
    Prevents the network from relying solely on any single modality (e.g. POS-rPPG),
    forcing the model to learn robust temporal and spatial features.
    """
    def __init__(self, p: float = 0.3):
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.p <= 0.0:
            return x
        # 1D channel mask per sample
        mask = (torch.rand(x.shape[0], 1, device=x.device) > self.p).float()
        return (x * mask) / (1.0 - self.p)


class MultimodalGatedAttentionAdapter(nn.Module):
    """
    Quality-Aware Dynamic Modality Neglect Architecture for Multimodal Deepfake Detection.
    Evaluates incoming modality feature vectors (Spatial, Temporal, Biological),
    estimates video quality dynamically, and applies Softmax gating to assign
    proportional attention weights across modalities.
    """
    def __init__(
        self,
        spatial_dim: int = 1280,
        temporal_dim: int = 512,
        biological_dim: int = 32,
        latent_dim: int = 256,
        dropout: float = 0.5,
        modality_dropout: float = 0.3
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.biological_dim = biological_dim
        self.latent_dim = latent_dim

        # Feature Channel Dropout per Modality
        self.modality_drop = FeatureChannelDropout(p=modality_dropout)

        # Projection Heads with Heavy Dropout Regularization (p=0.5)
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        self.proj_biological = nn.Sequential(
            nn.Linear(biological_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(p=dropout)
        )

        # Dynamic Gating Network with Dropout
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 3)
        )

        # Final Classification Head with Dropout
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x_S = x[:, :self.spatial_dim]
        x_T = x[:, self.spatial_dim:self.spatial_dim + self.temporal_dim]
        x_B = x[:, self.spatial_dim + self.temporal_dim:]

        h_S = self.proj_spatial(x_S)
        h_T = self.proj_temporal(x_T)
        h_B = self.proj_biological(x_B)

        # Apply Feature Channel Dropout to prevent over-reliance on biological branch
        h_S = self.modality_drop(h_S)
        h_T = self.modality_drop(h_T)
        h_B = self.modality_drop(h_B)

        h_cat = torch.cat([h_S, h_T, h_B], dim=-1)

        # Compute Quality Bias from spatial feature variance (proxy for image degradation)
        s_var = torch.var(x_S, dim=-1, keepdim=True)
        q_score = torch.sigmoid(s_var - 1.0)
        quality_bias = torch.cat([0.5 * q_score, 0.5 * (1.0 - q_score), torch.zeros_like(q_score)], dim=-1)

        gate_logits = self.gating_network(h_cat) + quality_bias
        attn_weights = F.softmax(gate_logits, dim=-1)

        w_S = attn_weights[:, 0:1]
        w_T = attn_weights[:, 1:2]
        w_B = attn_weights[:, 2:3]

        h_S_weighted = h_S * w_S
        h_T_weighted = h_T * w_T
        h_B_weighted = h_B * w_B

        h_fused = torch.cat([h_S_weighted, h_T_weighted, h_B_weighted], dim=-1)
        logits = self.cls_head(h_fused).squeeze(-1)

        return logits, attn_weights



from sklearn.preprocessing import StandardScaler

# ==========================================
# 3. PyTorch Dataset & DataLoader Integration
# ==========================================
class MultimodalDataset(Dataset):
    """
    PyTorch Dataset loading real fused .npz feature matrix.
    Supports single or multiple comma-separated .npz feature files.
    Applies StandardScaler normalization to feature vectors X.
    """
    def __init__(self, npz_path: str, scaler: Optional[StandardScaler] = None, fit_scaler: bool = True):
        if isinstance(npz_path, str) and "," in npz_path:
            raw_paths = [p.strip() for p in npz_path.split(",")]
        elif isinstance(npz_path, (list, tuple)):
            raw_paths = npz_path
        else:
            raw_paths = [npz_path]

        X_list = []
        y_list = []
        spatial_dim, temporal_dim, biological_dim = 1280, 512, 32

        for path_item in raw_paths:
            resolved_npz = str(resolve_path(path_item))
            if not os.path.exists(resolved_npz) and os.path.exists("/content/drive/MyDrive/extracted_features.npz"):
                resolved_npz = "/content/drive/MyDrive/extracted_features.npz"

            if not os.path.exists(resolved_npz):
                raise FileNotFoundError(f"ERROR: Extracted dataset file not found at '{resolved_npz}'.")

            data = np.load(resolved_npz)
            if "X" in data:
                X_item = torch.nan_to_num(torch.tensor(data["X"], dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
                X_list.append(X_item)
            else:
                raise KeyError(f"Key 'X' missing in {resolved_npz}. Available keys: {list(data.keys())}")

            if "y" in data:
                y_item = torch.tensor(data["y"], dtype=torch.float32)
                y_list.append(y_item)
            else:
                raise KeyError(f"Key 'y' missing in {resolved_npz}. Available keys: {list(data.keys())}")

            spatial_dim = int(data.get("spatial_dim", spatial_dim))
            temporal_dim = int(data.get("temporal_dim", temporal_dim))
            biological_dim = int(data.get("biological_dim", biological_dim))

        X_concat = torch.cat(X_list, dim=0).numpy()
        self.y = torch.cat(y_list, dim=0)
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.biological_dim = biological_dim

        if scaler is not None:
            self.scaler = scaler
            X_scaled = self.scaler.transform(X_concat)
        elif fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X_concat)
        else:
            self.scaler = None
            X_scaled = X_concat

        self.X = torch.tensor(X_scaled, dtype=torch.float32)
        self.scaler_mean = self.scaler.mean_ if self.scaler is not None else None
        self.scaler_scale = self.scaler.scale_ if self.scaler is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]



from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import GroupShuffleSplit

def train_adapter(
    npz_path: Optional[str] = None,
    save_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 15,
    dropout: float = 0.5,
    weight_decay: float = 1e-2,
    balance_classes: bool = True,
    use_focal_loss: bool = True,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0
):
    """
    Trains MultimodalGatedAttentionAdapter with Subject-Isolated GroupShuffleSplit, 
    L2 Weight Decay (1e-2), CosineAnnealingWarmRestarts, and FeatureChannelDropout (p=0.3).
    """
    if npz_path is None:
        target_npz = get_features_output_path("extracted_features.npz")
        str_npz_path = str(target_npz)
    else:
        str_npz_path = npz_path

    target_ckpt = get_checkpoint_path("attention_adapter.pth") if save_path is None else resolve_path(save_path)
    focal_ckpt = get_checkpoint_path("attention_adapter_focal.pth")
    str_ckpt_path = str(target_ckpt)
    str_focal_ckpt_path = str(focal_ckpt)

    # Ensure parent checkpoint directory exists
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    dataset = MultimodalDataset(str_npz_path)
    logger.info(f"Loaded Real Dataset '{str_npz_path}' | Samples: {len(dataset)} | Feature Matrix X: {dataset.X.shape}")

    # Subject-Isolated Group-Based Data Splitting (preventing identity leakage)
    if hasattr(dataset, "groups") and dataset.groups is not None:
        groups = dataset.groups
    else:
        # Group contiguous video frames (16 frames per video block) to prevent subject/frame leakage
        groups = np.arange(len(dataset)) // 16

    gss = GroupShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
    train_indices, val_indices = next(gss.split(dataset.X, dataset.y, groups=groups))

    train_ds = torch.utils.data.Subset(dataset, train_indices)
    val_ds = torch.utils.data.Subset(dataset, val_indices)

    logger.info(f"Subject-Isolated Split -> Train Samples: {len(train_ds)} | Val Samples: {len(val_ds)}")

    # Class distribution in train split
    train_targets = torch.tensor([dataset[i][1] for i in train_indices])
    num_real = (train_targets == 0.0).sum().item()
    num_fake = (train_targets == 1.0).sum().item()

    logger.info(f"Training set raw class distribution -> Real (0.0): {num_real} | Fake (1.0): {num_fake}")

    if balance_classes and num_real > 0 and num_fake > 0:
        class_weights = [1.0 / num_real, 1.0 / num_fake]
        sample_weights = [class_weights[int(t.item())] for t in train_targets]
        sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
        pos_weight = torch.tensor([num_real / num_fake]).to(device)
        logger.info(f"Enabled Equal Ratio Class Balancing via WeightedRandomSampler & pos_weight ({pos_weight.item():.4f})")
    else:
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        pos_weight = None

    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultimodalGatedAttentionAdapter(
        spatial_dim=dataset.spatial_dim,
        temporal_dim=dataset.temporal_dim,
        biological_dim=dataset.biological_dim,
        dropout=dropout,
        modality_dropout=0.3
    ).to(device)

    # Auto-detect class distribution directly from dataset y
    all_y = dataset.y.numpy()
    num_reals = int((all_y == 0.0).sum())
    num_fakes = int((all_y == 1.0).sum())
    samples_per_class = [num_reals, num_fakes]
    logger.info(f"Auto-Detected Class Counts -> REAL (0.0): {num_reals} | FAKE (1.0): {num_fakes}")

    if use_focal_loss:
        criterion = CB_FocalLoss(samples_per_class=samples_per_class, beta=0.999, gamma=focal_gamma)
        logger.info(f"Using Class-Balanced FocalLoss (samples={samples_per_class}, beta=0.999, gamma={focal_gamma})")
    else:
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info("Using standard BCEWithLogitsLoss.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    early_stopping = EarlyStoppingAUC(patience=patience, mode="max")

    logger.info(f"Starting training (Max Epochs: {epochs}, Patience: {patience}, Dropout: {dropout}, WeightDecay: {weight_decay})...")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)

            optimizer.zero_grad()
            logits, attn = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y_b)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            train_correct += (preds == y_b).sum().item()
            train_total += len(y_b)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        avg_weights = np.zeros(3)
        val_probs_list, val_targets_list = [], []

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits, attn = model(X_b)
                loss = criterion(logits, y_b)

                val_loss += loss.item() * len(y_b)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
                avg_weights += attn.mean(dim=0).cpu().numpy() * len(y_b)
                val_probs_list.append(probs.cpu())
                val_targets_list.append(y_b.cpu())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_weights /= max(val_total, 1)

        val_probs_all = torch.cat(val_probs_list).numpy()
        val_targets_all = torch.cat(val_targets_list).numpy()
        if len(np.unique(val_targets_all)) > 1:
            val_auc = float(roc_auc_score(val_targets_all, val_probs_all))
        else:
            val_auc = 0.5000

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% AUC: {val_auc:.4f} | "
            f"Weights -> S: {avg_weights[0]:.2f}, T: {avg_weights[1]:.2f}, B: {avg_weights[2]:.2f}"
        )

        is_best = early_stopping(val_auc, epoch)
        if is_best:
            ckpt_dict = {
                "model_state_dict": model.state_dict(),
                "scaler_mean": dataset.scaler_mean.tolist() if dataset.scaler_mean is not None else None,
                "scaler_scale": dataset.scaler_scale.tolist() if dataset.scaler_scale is not None else None,
                "spatial_dim": dataset.spatial_dim,
                "temporal_dim": dataset.temporal_dim,
                "biological_dim": dataset.biological_dim,
                "best_val_auc": val_auc
            }
            torch.save(ckpt_dict, str_ckpt_path)
            torch.save(ckpt_dict, str_focal_ckpt_path)
            logger.info(f"--> Saved best model checkpoint to '{str_ckpt_path}' and '{str_focal_ckpt_path}' (Val AUC-ROC: {val_auc:.4f})")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at Epoch {epoch}! Best Val AUC-ROC: {early_stopping.best_auc:.4f} at Epoch {early_stopping.best_epoch}.")
            break

    # Restore optimal checkpoint
    if os.path.exists(str_ckpt_path):
        ckpt_loaded = torch.load(str_ckpt_path, map_location=device, weights_only=False)
        state = ckpt_loaded.get("model_state_dict", ckpt_loaded) if isinstance(ckpt_loaded, dict) else ckpt_loaded
        model.load_state_dict(state)
        logger.info(f"Loaded optimal checkpoint weights from epoch {early_stopping.best_epoch}.")




if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Attention Adapter Training Pipeline")
    parser.add_argument("--features", type=str, default=None, help="Path to .npz dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--dropout", type=float, default=0.4, help="Dropout probability")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="L2 weight decay")
    args = parser.parse_args()

    train_adapter(
        npz_path=args.features,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dropout=args.dropout,
        weight_decay=args.weight_decay
    )
