import os
from pathlib import Path
from typing import Union, Dict, Any
import yaml

# Determine project root directory dynamically (parent of utils/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Detect Colab environment & Google Drive mount
COLAB_DRIVE_DIR = Path("/content/drive/MyDrive")
IS_COLAB = COLAB_DRIVE_DIR.exists()

def get_base_output_dir() -> Path:
    """
    Returns base output directory dynamically:
    - If Google Colab (Drive mounted): /content/drive/MyDrive/DeepFake_Outputs/
    - Otherwise (Local Machine): PROJECT_ROOT / "outputs"
    Automatically creates parent directories.
    """
    if IS_COLAB:
        output_dir = COLAB_DRIVE_DIR / "DeepFake_Outputs"
    else:
        output_dir = PROJECT_ROOT / "outputs"
        
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def resolve_path(relative_path: Union[str, Path]) -> Path:
    """
    Resolves any path dynamically relative to PROJECT_ROOT if not already absolute.
    """
    path_obj = Path(relative_path) if isinstance(relative_path, str) else relative_path
    if path_obj.is_absolute():
        return path_obj
    return (PROJECT_ROOT / path_obj).resolve()

def get_features_output_path(filename: str = "extracted_features.npz") -> Path:
    """
    Returns path for extracted .npz features.
    Checks Drive root (/content/drive/MyDrive/extracted_features.npz) first on Colab.
    """
    if IS_COLAB:
        drive_root_file = COLAB_DRIVE_DIR / filename
        if drive_root_file.exists():
            return drive_root_file
        out_dir = COLAB_DRIVE_DIR / "DeepFake_Outputs" / "features"
    else:
        out_dir = get_base_output_dir() / "features"
        
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename

def get_checkpoint_path(filename: str = "attention_adapter.pth") -> Path:
    """Returns path for model checkpoint .pth inside environment output dir."""
    out_dir = get_base_output_dir() / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename

def get_eval_output_path(filename: str = "roc_curve.png") -> Path:
    """Returns path for evaluation plots/results inside environment output dir."""
    out_dir = get_base_output_dir() / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / filename

def load_config(config_path: Union[str, Path] = "configs/config.yaml") -> Dict[str, Any]:
    """Loads YAML configuration cleanly using dynamic relative path resolution."""
    resolved_cfg_path = resolve_path(config_path)
    if not resolved_cfg_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_cfg_path}")
    with open(resolved_cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config
