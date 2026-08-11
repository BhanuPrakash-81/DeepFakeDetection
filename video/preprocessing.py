import cv2
import numpy as np
import torch
import torchvision.transforms as T
from typing import Tuple, List, Optional

# ImageNet normalization standard
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

class ImagePreprocessor:
    """
    Handles RGB conversion, ROI cropping with spatial safety margins,
    resizing, and PyTorch ImageNet tensor normalization.
    """
    def __init__(self, target_size: Tuple[int, int] = (112, 112)):
        self.target_size = target_size
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        ])

    def crop_roi(self, frame_bgr: np.ndarray, bbox: Tuple[int, int, int, int], margin: float = 0.15) -> np.ndarray:
        """
        Crops ROI bbox (xmin, ymin, xmax, ymax) from frame with margin expansion.
        Clamps to image boundaries.
        """
        h, w = frame_bgr.shape[:2]
        xmin, ymin, xmax, ymax = bbox
        
        bw = xmax - xmin
        bh = ymax - ymin
        
        margin_x = int(bw * margin)
        margin_y = int(bh * margin)
        
        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(w, xmax + margin_x)
        crop_ymax = min(h, ymax + margin_y)
        
        # Guard against zero height/width
        if crop_xmax <= crop_xmin or crop_ymax <= crop_ymin:
            crop = frame_bgr
        else:
            crop = frame_bgr[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            
        return crop

    def preprocess_patch(self, patch_bgr: np.ndarray) -> torch.Tensor:
        """
        Converts BGR numpy image patch to PyTorch normalized tensor of shape (3, H, W).
        """
        patch_rgb = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2RGB)
        patch_resized = cv2.resize(patch_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        tensor = self.transform(patch_resized)
        return tensor

    def batch_preprocess_patches(self, patch_bgr_list: List[np.ndarray]) -> torch.Tensor:
        """
        Converts list of BGR patches into batched tensor of shape (N, 3, H, W).
        """
        tensors = [self.preprocess_patch(p) for p in patch_bgr_list]
        if not tensors:
            return torch.empty((0, 3, self.target_size[0], self.target_size[1]))
        return torch.stack(tensors, dim=0)
