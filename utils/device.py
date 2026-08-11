import torch
import psutil
from typing import Tuple, Dict, Any
from utils.logging import setup_logger

logger = setup_logger("DeviceUtil")

def get_device(preference: str = "auto") -> torch.device:
    """
    Selects PyTorch device based on preference ('auto', 'cuda', 'cpu') and hardware availability.
    """
    if preference == "cuda":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            logger.warning("CUDA requested but not available. Falling back to CPU.")
            device = torch.device("cpu")
    elif preference == "cpu":
        device = torch.device("cpu")
    else:  # auto
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
            
    logger.info(f"Using compute device: {device}")
    return device

def get_memory_usage() -> Dict[str, Any]:
    """
    Returns memory statistics for RAM and GPU (if CUDA available).
    """
    process = psutil.Process()
    ram_bytes = process.memory_info().rss
    ram_mb = ram_bytes / (1024 * 1024)
    
    stats = {
        "ram_used_mb": round(ram_mb, 2),
        "ram_percent": psutil.virtual_memory().percent
    }
    
    if torch.cuda.is_available():
        stats["gpu_allocated_mb"] = round(torch.cuda.memory_allocated() / (1024 * 1024), 2)
        stats["gpu_reserved_mb"] = round(torch.cuda.memory_reserved() / (1024 * 1024), 2)
    else:
        stats["gpu_allocated_mb"] = 0.0
        stats["gpu_reserved_mb"] = 0.0
        
    return stats
