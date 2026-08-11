import os
import sys
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Any, Optional

from utils.paths import get_features_output_path, get_checkpoint_path, resolve_path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ModelAndTrain")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


# ==========================================
# 1. Early Stopping Utility
# ==========================================
class EarlyStopping:
    """
    Early Stopping mechanism to monitor validation loss during training.
    Stops training early if validation loss does not improve after `patience` consecutive epochs,
    preventing over-fitting and saving model generalization performance.
    """
    def __init__(self, patience: int = 7, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss: float, epoch: int) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0
            return True # Improved
        else:
            self.counter += 1
            logger.info(f"EarlyStopping counter: {self.counter} out of {self.patience} (Best Val Loss: {self.best_loss:.4f} at Epoch {self.best_epoch})")
            if self.counter >= self.patience:
                self.early_stop = True
            return False # No improvement


# ==========================================
# 2. Multimodal Dynamic Gated Attention Adapter
# ==========================================
class MultimodalGatedAttentionAdapter(nn.Module):
    """
    Dynamic Modality Neglect Architecture for Multimodal Deepfake Detection.
    Evaluates incoming modality feature vectors (Spatial, Temporal, Biological),
    projects them into a shared latent space, and applies Softmax gating to assign
    dynamic attention weights. Dropout regularization (p=0.4) prevents co-adaptation.
    """
    def __init__(
        self,
        spatial_dim: int = 1280,
        temporal_dim: int = 512,
        biological_dim: int = 32,
        latent_dim: int = 256,
        dropout: float = 0.4
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.biological_dim = biological_dim
        self.latent_dim = latent_dim

        # Projection Heads with Dropout Regularization
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

        h_cat = torch.cat([h_S, h_T, h_B], dim=-1)

        gate_logits = self.gating_network(h_cat)
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


# ==========================================
# 3. PyTorch Dataset & DataLoader Integration
# ==========================================
class MultimodalDataset(Dataset):
    """
    PyTorch Dataset loading real fused .npz feature matrix.
    Extracts features using key 'X' (shape: (N, 1824)) and labels using key 'y' (shape: (N,)).
    Raises explicit FileNotFoundError if dataset path does not exist.
    """
    def __init__(self, npz_path: str):
        resolved_npz = str(resolve_path(npz_path))
        
        # Drive root fallback check for Google Colab
        if not os.path.exists(resolved_npz) and os.path.exists("/content/drive/MyDrive/extracted_features.npz"):
            resolved_npz = "/content/drive/MyDrive/extracted_features.npz"

        if not os.path.exists(resolved_npz):
            raise FileNotFoundError(
                f"ERROR: Extracted dataset file not found at '{resolved_npz}'.\n"
                f"Checked Locations:\n"
                f" - /content/drive/MyDrive/extracted_features.npz\n"
                f" - /content/drive/MyDrive/DeepFake_Outputs/features/extracted_features.npz\n"
                f" - ./outputs/features/extracted_features.npz\n"
                f"Please ensure Stage 1 (extract_features.py) has completed successfully!"
            )

        data = np.load(resolved_npz)

        # Exact Archive Key Mapping: 'X' for features, 'y' for labels
        if "X" in data:
            self.X = torch.tensor(data["X"], dtype=torch.float32)
        else:
            raise KeyError(f"Key 'X' missing in {resolved_npz}. Available keys: {list(data.keys())}")

        if "y" in data:
            self.y = torch.tensor(data["y"], dtype=torch.float32)
        else:
            raise KeyError(f"Key 'y' missing in {resolved_npz}. Available keys: {list(data.keys())}")

        self.spatial_dim = int(data.get("spatial_dim", 1280))
        self.temporal_dim = int(data.get("temporal_dim", 512))
        self.biological_dim = int(data.get("biological_dim", 32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_adapter(
    npz_path: Optional[str] = None,
    save_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2,
    patience: int = 7,
    dropout: float = 0.4,
    weight_decay: float = 1e-4
):
    """
    Trains MultimodalGatedAttentionAdapter on extracted .npz features with train/val split.
    Uses exact key mapping ('X', 'y'), Dropout (p=0.4), Weight Decay (1e-4), and EarlyStopping (patience=7).
    """
    target_npz = get_features_output_path("extracted_features.npz") if npz_path is None else resolve_path(npz_path)
    target_ckpt = get_checkpoint_path("attention_adapter.pth") if save_path is None else resolve_path(save_path)

    str_npz_path = str(target_npz)
    str_ckpt_path = str(target_ckpt)

    # Colab Drive root fallback check
    if not os.path.exists(str_npz_path) and os.path.exists("/content/drive/MyDrive/extracted_features.npz"):
        str_npz_path = "/content/drive/MyDrive/extracted_features.npz"

    if not os.path.exists(str_npz_path):
        raise FileNotFoundError(
            f"ERROR: Extracted features file not found at: '{str_npz_path}'\n"
            f"Expected Locations:\n"
            f" - Colab Drive Root: '/content/drive/MyDrive/extracted_features.npz'\n"
            f" - Colab Drive Subfolder: '/content/drive/MyDrive/DeepFake_Outputs/features/extracted_features.npz'\n"
            f" - Local Machine: './outputs/features/extracted_features.npz'\n"
            f"Please run Stage 1 (extract_features.py) first to extract feature vectors!"
        )

    # Ensure parent checkpoint directory exists
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    dataset = MultimodalDataset(str_npz_path)
    logger.info(f"Loaded Real Dataset '{str_npz_path}' | Samples: {len(dataset)} | Feature Matrix X: {dataset.X.shape}")

    val_size = max(1, int(len(dataset) * val_split))
    train_size = len(dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = MultimodalGatedAttentionAdapter(
        spatial_dim=dataset.spatial_dim,
        temporal_dim=dataset.temporal_dim,
        biological_dim=dataset.biological_dim,
        dropout=dropout
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    early_stopping = EarlyStopping(patience=patience)

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

        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                logits, attn = model(X_b)
                loss = criterion(logits, y_b)

                val_loss += loss.item() * len(y_b)
                preds = (torch.sigmoid(logits) >= 0.5).float()
                val_correct += (preds == y_b).sum().item()
                val_total += len(y_b)
                avg_weights += attn.mean(dim=0).cpu().numpy() * len(y_b)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        avg_weights /= max(val_total, 1)

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
            f"Weights -> S: {avg_weights[0]:.2f}, T: {avg_weights[1]:.2f}, B: {avg_weights[2]:.2f}"
        )

        is_best = early_stopping(val_loss, epoch)
        if is_best:
            torch.save(model.state_dict(), str_ckpt_path)
            logger.info(f"--> Saved best model checkpoint to '{str_ckpt_path}' (Val Loss: {val_loss:.4f})")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at Epoch {epoch}! Best Val Loss: {early_stopping.best_loss:.4f} at Epoch {early_stopping.best_epoch}.")
            break

    # Restore optimal checkpoint
    if os.path.exists(str_ckpt_path):
        model.load_state_dict(torch.load(str_ckpt_path, map_location=device))
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
