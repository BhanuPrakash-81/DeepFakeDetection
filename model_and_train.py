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
# 1. Multimodal Dynamic Gated Attention Adapter
# ==========================================
class MultimodalGatedAttentionAdapter(nn.Module):
    """
    Dynamic Modality Neglect Architecture for Multimodal Deepfake Detection.
    Evaluates incoming modality feature vectors (Spatial, Temporal, Biological),
    projects them into a shared latent space, and applies Softmax gating to assign
    dynamic attention weights. Corrupted/noisy modalities receive weight ~ 0.0.
    """
    def __init__(
        self,
        spatial_dim: int = 1280,
        temporal_dim: int = 512,
        biological_dim: int = 32,
        latent_dim: int = 256,
        dropout: float = 0.3
    ):
        super().__init__()
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.biological_dim = biological_dim
        self.latent_dim = latent_dim

        # Projection Heads to Shared Latent Space (256D)
        self.proj_spatial = nn.Sequential(
            nn.Linear(spatial_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_temporal = nn.Sequential(
            nn.Linear(temporal_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.proj_biological = nn.Sequential(
            nn.Linear(biological_dim, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Dynamic Gating Network
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3)
        )

        # Classification Head
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
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
# 2. PyTorch Dataset & Training Loop
# ==========================================
class MultimodalDataset(Dataset):
    """
    PyTorch Dataset loading real fused .npz feature vectors.
    Flexibly unpacks 'X' or 'features', and 'y' or 'labels'.
    Raises explicit FileNotFoundError if dataset path does not exist.
    """
    def __init__(self, npz_path: str):
        resolved_npz = str(resolve_path(npz_path))
        if not os.path.exists(resolved_npz):
            raise FileNotFoundError(
                f"ERROR: Dataset file not found at '{resolved_npz}'.\n"
                f"Colab Drive Location : /content/drive/MyDrive/DeepFake_Outputs/features/extracted_features.npz\n"
                f"Local Machine Location: ./outputs/features/extracted_features.npz\n"
                f"Please ensure Stage 1 (extract_features.py) has completed successfully!"
            )

        data = np.load(resolved_npz)

        # Unpack feature matrix (support 'X' or 'features')
        if "X" in data:
            self.X = torch.tensor(data["X"], dtype=torch.float32)
        elif "features" in data:
            self.X = torch.tensor(data["features"], dtype=torch.float32)
        else:
            raise KeyError(f"Neither 'X' nor 'features' key found in {resolved_npz}. Keys present: {list(data.keys())}")

        # Unpack labels (support 'y' or 'labels')
        if "y" in data:
            self.y = torch.tensor(data["y"], dtype=torch.float32)
        elif "labels" in data:
            self.y = torch.tensor(data["labels"], dtype=torch.float32)
        else:
            raise KeyError(f"Neither 'y' nor 'labels' key found in {resolved_npz}. Keys present: {list(data.keys())}")

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
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2
):
    """
    Trains MultimodalGatedAttentionAdapter on extracted .npz features with train/val split.
    Strictly requires the real dataset; saves model weights directly to Google Drive (Colab) or Local outputs.
    """
    target_npz = get_features_output_path("extracted_features.npz") if npz_path is None else resolve_path(npz_path)
    target_ckpt = get_checkpoint_path("attention_adapter.pth") if save_path is None else resolve_path(save_path)

    str_npz_path = str(target_npz)
    str_ckpt_path = str(target_ckpt)

    # Strictly require real .npz dataset file; NO fallback creation during production training!
    if not os.path.exists(str_npz_path):
        raise FileNotFoundError(
            f"ERROR: Extracted features file not found at: '{str_npz_path}'\n"
            f"Expected Google Drive Location (Colab): '/content/drive/MyDrive/DeepFake_Outputs/features/extracted_features.npz'\n"
            f"Expected Local Location: './outputs/features/extracted_features.npz'\n"
            f"Please run Stage 1 (extract_features.py) first to extract the feature vectors!"
        )

    # Ensure parent checkpoint directory exists
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    dataset = MultimodalDataset(str_npz_path)
    logger.info(f"Loaded Real Dataset '{str_npz_path}' | Samples: {len(dataset)} | Feature Dim: {dataset.X.shape[1]}")

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
        biological_dim=dataset.biological_dim
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_loss = float("inf")
    logger.info(f"Starting training for {epochs} epochs on {train_size} train, {val_size} val samples...")

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

        if epoch % 5 == 0 or epoch == epochs or val_loss < best_val_loss:
            logger.info(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
                f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% | "
                f"Weights -> S: {avg_weights[0]:.2f}, T: {avg_weights[1]:.2f}, B: {avg_weights[2]:.2f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str_ckpt_path)
            logger.info(f"Saved best model checkpoint to '{str_ckpt_path}' (Val Loss: {val_loss:.4f})")

    logger.info("Training pipeline completed successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Attention Adapter Training Pipeline")
    parser.add_argument("--features", type=str, default=None, help="Path to .npz dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Checkpoint output path")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    train_adapter(
        npz_path=args.features,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )
