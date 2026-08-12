import os
import sys
import yaml
import logging
import argparse
from typing import Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import roc_auc_score

# Ensure project root directory is accessible in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model_and_train import MultimodalGatedAttentionAdapter, CB_FocalLoss, EarlyStoppingAUC
from utils.paths import get_features_output_path, get_checkpoint_path, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("TrainAdapterConcatDataset")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Custom Feature Dataset Class
# ==========================================
class FeatureDataset(Dataset):
    """
    Custom PyTorch Dataset that loads multimodal features 
    (Spatial: 1280, Temporal: 512, Biological: 32) and ground-truth binary labels
    (0.0 = REAL, 1.0 = FAKE) from a .npz feature file.
    """
    def __init__(self, npz_path: str):
        resolved_path = str(resolve_path(npz_path))
        if not os.path.exists(resolved_path):
            raise FileNotFoundError(f"Feature dataset not found at '{resolved_path}'.")

        data = np.load(resolved_path)
        if "X" in data and "y" in data:
            self.X = torch.nan_to_num(torch.tensor(data["X"], dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
            self.y = torch.tensor(data["y"], dtype=torch.float32)
        else:
            raise KeyError(f"Expected keys 'X' and 'y' in {resolved_path}. Found keys: {list(data.keys())}")

        self.spatial_dim = int(data.get("spatial_dim", 1280))
        self.temporal_dim = int(data.get("temporal_dim", 512))
        self.biological_dim = int(data.get("biological_dim", 32))
        self.file_path = resolved_path

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# Helper to load batch size from config.yaml or default to 64
def get_config_batch_size(default_bs: int = 64) -> int:
    config_path = resolve_path("configs/config.yaml")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f)
                if cfg and "classifier" in cfg and "pytorch_adapter_params" in cfg["classifier"]:
                    return int(cfg["classifier"]["pytorch_adapter_params"].get("batch_size", default_bs))
        except Exception:
            pass
    return default_bs


# ==========================================
# 2. ConcatDataset Master Pipeline
# ==========================================
def train_adapter_concat_pipeline(
    dfd_path: Optional[str] = None,
    ff_path: Optional[str] = None,
    save_path: Optional[str] = None,
    epochs: int = 50,
    batch_size: Optional[int] = None,
    lr: float = 1e-3,
    patience: int = 15,
    dropout: float = 0.5,
    weight_decay: float = 1e-2,
    focal_beta: float = 0.999,
    focal_gamma: float = 2.0
):
    """
    Unified Master Dataset Training Pipeline using ConcatDataset:
    - Loads DFD Features (data/extracted_features.npz or features_out/extracted_features.npz)
    - Loads FF++ Features (data/extracted_features_ff++.npz or features_out/extracted_features_ff++.npz)
    - Wraps both datasets using torch.utils.data.ConcatDataset
    - Applies WeightedRandomSampler for 50:50 equal ratio class balance
    - Stratified 80/20 train/val split
    - Trains MultimodalGatedAttentionAdapter with AdamW and CosineAnnealingWarmRestarts
    """
    # 1. Resolve Dataset File Paths
    default_dfd_paths = [
        "features_out/extracted_features.npz",
        "data/extracted_features.npz",
        "outputs/features/extracted_features.npz"
    ]
    default_ff_paths = [
        "features_out/extracted_features_ff++.npz",
        "data/extracted_features_ff++.npz",
        "outputs/features/extracted_features_ff++.npz"
    ]

    resolved_dfd = None
    if dfd_path is not None and os.path.exists(str(resolve_path(dfd_path))):
        resolved_dfd = str(resolve_path(dfd_path))
    else:
        for p in default_dfd_paths:
            if os.path.exists(str(resolve_path(p))):
                resolved_dfd = str(resolve_path(p))
                break

    resolved_ff = None
    if ff_path is not None and os.path.exists(str(resolve_path(ff_path))):
        resolved_ff = str(resolve_path(ff_path))
    else:
        for p in default_ff_paths:
            if os.path.exists(str(resolve_path(p))):
                resolved_ff = str(resolve_path(p))
                break

    if resolved_dfd is None or not os.path.exists(resolved_dfd):
        raise FileNotFoundError(f"DFD feature dataset not found. Checked locations: {default_dfd_paths}")
    if resolved_ff is None or not os.path.exists(resolved_ff):
        raise FileNotFoundError(f"FF++ feature dataset not found. Checked locations: {default_ff_paths}")

    # Set batch size from config or fallback to 64
    if batch_size is None:
        batch_size = get_config_batch_size(default_bs=64)

    # 2. Instantiate Datasets & ConcatDataset
    dataset_dfd = FeatureDataset(resolved_dfd)
    dataset_ff = FeatureDataset(resolved_ff)

    master_dataset = ConcatDataset([dataset_dfd, dataset_ff])

    # Log required dataset shapes prior to training
    logger.info("======================================================================")
    logger.info("       MULTIMODAL DEEPFAKE DETECTION - CONCAT DATASET PIPELINE        ")
    logger.info("======================================================================")
    logger.info(f" Total DFD samples          : {len(dataset_dfd)}")
    logger.info(f" Total FF++ samples         : {len(dataset_ff)}")
    logger.info(f" Combined Master Dataset size: {len(master_dataset)}")
    logger.info(f" Batch Size                 : {batch_size}")
    logger.info("----------------------------------------------------------------------")

    # 3. Stratified 80/20 Train/Val Split
    val_size = int(len(master_dataset) * 0.20)
    train_size = len(master_dataset) - val_size

    train_ds, val_ds = torch.utils.data.random_split(
        master_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )

    # 4. Class Balancing & WeightedRandomSampler Setup (50:50 equal ratio)
    train_targets = torch.tensor([train_ds[i][1].item() for i in range(len(train_ds))])
    num_real = (train_targets == 0.0).sum().item()
    num_fake = (train_targets == 1.0).sum().item()

    logger.info(f" Combined Train Split Counts -> REAL (0.0): {int(num_real)} | FAKE (1.0): {int(num_fake)}")

    class_weights = [1.0 / max(num_real, 1), 1.0 / max(num_fake, 1)]
    sample_weights = [class_weights[int(t.item())] for t in train_targets]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # 5. Initialize Adapter Model & Loss Setup
    model = MultimodalGatedAttentionAdapter(
        spatial_dim=dataset_dfd.spatial_dim,
        temporal_dim=dataset_dfd.temporal_dim,
        biological_dim=dataset_dfd.biological_dim,
        dropout=dropout,
        modality_dropout=0.3
    ).to(device)

    # Class-Balanced Focal Loss with auto-detected combined class counts
    all_combined_y = torch.cat([dataset_dfd.y, dataset_ff.y]).numpy()
    master_num_reals = int((all_combined_y == 0.0).sum())
    master_num_fakes = int((all_combined_y == 1.0).sum())
    samples_per_class = [master_num_reals, master_num_fakes]

    criterion = CB_FocalLoss(samples_per_class=samples_per_class, beta=focal_beta, gamma=focal_gamma)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    early_stopping = EarlyStoppingAUC(patience=patience, mode="max")

    # 6. Checkpoint Setup
    target_ckpt = get_checkpoint_path("attention_adapter_focal.pth") if save_path is None else resolve_path(save_path)
    primary_ckpt = get_checkpoint_path("attention_adapter.pth")
    target_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # 7. Training Loop Across Epochs
    logger.info(f"Initiating Training Loop across {epochs} Epochs on {len(train_ds)} train samples...")

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
        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        val_probs_list, val_targets_list = [], []
        avg_weights = np.zeros(3)

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
        val_auc = float(roc_auc_score(val_targets_all, val_probs_all)) if len(np.unique(val_targets_all)) > 1 else 0.5

        logger.info(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc*100:.1f}% AUC: {val_auc:.4f} | "
            f"Weights -> S: {avg_weights[0]:.2f}, T: {avg_weights[1]:.2f}, B: {avg_weights[2]:.2f}"
        )

        is_best = early_stopping(val_auc, epoch)
        if is_best:
            torch.save(model.state_dict(), str(target_ckpt))
            torch.save(model.state_dict(), str(primary_ckpt))
            logger.info(f"--> Saved best ConcatDataset Model Checkpoint to '{target_ckpt}' (Val AUC-ROC: {val_auc:.4f})")

        if early_stopping.early_stop:
            logger.info(f"Early stopping triggered at Epoch {epoch}! Best Val AUC-ROC: {early_stopping.best_auc:.4f} at Epoch {early_stopping.best_epoch}.")
            break

    # Restore optimal checkpoint
    if os.path.exists(str(target_ckpt)):
        model.load_state_dict(torch.load(str(target_ckpt), map_location=device))
        logger.info(f"Loaded optimal checkpoint weights from epoch {early_stopping.best_epoch}.")


def train_adapter(*args, **kwargs):
    """Alias for train_adapter_concat_pipeline."""
    return train_adapter_concat_pipeline(*args, **kwargs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal Deepfake Detection - ConcatDataset Training Script")
    parser.add_argument("--dfd_features", type=str, default="features_out/extracted_features.npz", help="Path to DFD feature dataset")
    parser.add_argument("--ff_features", type=str, default="features_out/extracted_features_ff++.npz", help="Path to FF++ feature dataset")
    parser.add_argument("--save_path", type=str, default=None, help="Output checkpoint path")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping AUC patience")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="L2 Weight decay")
    args = parser.parse_args()

    train_adapter_concat_pipeline(
        dfd_path=args.dfd_features,
        ff_path=args.ff_features,
        save_path=args.save_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        patience=args.patience,
        dropout=args.dropout,
        weight_decay=args.weight_decay
    )
