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

        # Dynamic Gating Network: Evaluates concatenated representations and outputs 3 attention logits
        # Concatenated latent dimension = 3 * 256 = 768
        self.gating_network = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3) # Logits for [Spatial, Temporal, Biological]
        )

        # Classification Head on Modulated Latent Features
        self.cls_head = nn.Sequential(
            nn.Linear(latent_dim * 3, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1) # Binary classification logit
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (Batch, spatial_dim + temporal_dim + biological_dim) concatenated vector
        Returns:
            logits: (Batch,) raw classification logit
            attention_weights: (Batch, 3) Softmax weights [w_S, w_T, w_B]
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Slice input vector into individual branch representations
        x_S = x[:, :self.spatial_dim]
        x_T = x[:, self.spatial_dim:self.spatial_dim + self.temporal_dim]
        x_B = x[:, self.spatial_dim + self.temporal_dim:]

        # Project into shared latent space
        h_S = self.proj_spatial(x_S)  # (B, 256)
        h_T = self.proj_temporal(x_T)  # (B, 256)
        h_B = self.proj_biological(x_B) # (B, 256)

        # Concatenate latent representations for gating evaluation
        h_cat = torch.cat([h_S, h_T, h_B], dim=-1) # (B, 768)

        # Compute dynamic gating logits & Softmax attention weights
        gate_logits = self.gating_network(h_cat) # (B, 3)
        attn_weights = F.softmax(gate_logits, dim=-1) # (B, 3) -> [w_S, w_T, w_B]

        # Modulate latent features with dynamic attention weights
        w_S = attn_weights[:, 0:1] # (B, 1)
        w_T = attn_weights[:, 1:2] # (B, 1)
        w_B = attn_weights[:, 2:3] # (B, 1)

        h_S_weighted = h_S * w_S
        h_T_weighted = h_T * w_T
        h_B_weighted = h_B * w_B

        # Fuse weighted features
        h_fused = torch.cat([h_S_weighted, h_T_weighted, h_B_weighted], dim=-1) # (B, 768)

        # Final binary classification logit
        logits = self.cls_head(h_fused).squeeze(-1) # (B,)

        return logits, attn_weights


# ==========================================
# 2. PyTorch Dataset & Training Loop
# ==========================================
class MultimodalDataset(Dataset):
    """PyTorch Dataset loading fused .npz feature vectors."""
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.X = torch.tensor(data["X"], dtype=torch.float32)
        self.y = torch.tensor(data["y"], dtype=torch.float32)
        self.spatial_dim = int(data.get("spatial_dim", 1280))
        self.temporal_dim = int(data.get("temporal_dim", 512))
        self.biological_dim = int(data.get("biological_dim", 32))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def train_adapter(
    npz_path: str = "features_out/extracted_features.npz",
    save_path: str = "attention_adapter.pth",
    epochs: int = 50,
    batch_size: int = 16,
    lr: float = 1e-3,
    val_split: float = 0.2
):
    """
    Trains the MultimodalGatedAttentionAdapter on extracted .npz features with train/val split.
    """
    if not os.path.exists(npz_path):
        logger.warning(f"Feature file '{npz_path}' not found. Creating synthetic dataset for demonstration training.")
        os.makedirs(os.path.dirname(npz_path) or ".", exist_ok=True)
        create_dummy_npz(npz_path, num_samples=100)

    dataset = MultimodalDataset(npz_path)
    val_size = int(len(dataset) * val_split)
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
        # Training Phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

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
        val_loss = 0.0
        val_correct = 0
        val_total = 0
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
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved best model checkpoint to '{save_path}' (Val Loss: {val_loss:.4f})")

    logger.info("Training pipeline completed successfully.")


def create_dummy_npz(filepath: str, num_samples: int = 100):
    """Creates synthetic .npz feature file for training pipeline testing."""
    spatial_dim, temporal_dim, biological_dim = 1280, 512, 32
    X_spatial = np.random.randn(num_samples, spatial_dim).astype(np.float32)
    X_temporal = np.random.randn(num_samples, temporal_dim).astype(np.float32)
    X_biological = np.random.randn(num_samples, biological_dim).astype(np.float32)

    # Corrupt some biological signals with heavy noise to test gating neglect
    X_biological[::3] = np.random.randn(biological_dim).astype(np.float32) * 50.0

    X_fused = np.concatenate([X_spatial, X_temporal, X_biological], axis=1)
    y = np.random.randint(0, 2, size=num_samples).astype(np.int32)

    np.savez_compressed(
        filepath,
        X=X_fused,
        X_spatial=X_spatial,
        X_temporal=X_temporal,
        X_biological=X_biological,
        y=y,
        spatial_dim=spatial_dim,
        temporal_dim=temporal_dim,
        biological_dim=biological_dim
    )
    logger.info(f"Created synthetic dataset at '{filepath}' with shape {X_fused.shape}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 2: Attention Adapter Training Pipeline")
    parser.add_argument("--features", type=str, default="features_out/extracted_features.npz", help="Path to .npz dataset")
    parser.add_argument("--save_path", type=str, default="attention_adapter.pth", help="Checkpoint output path")
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
