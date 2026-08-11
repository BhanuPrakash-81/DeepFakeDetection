import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from typing import Dict, Tuple
from utils.logging import setup_logger

logger = setup_logger("VisualEncoder")

class SharedVisualEncoder(nn.Module):
    """
    Single shared pretrained visual backbone for processing all cropped facial ROIs
    (left eye, right eye, nose, mouth, chin/jaw) in a single batched forward pass.
    Backbone weights are completely frozen.
    """
    def __init__(self, backbone_name: str = "resnet18", pretrained: bool = True, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        self.device = device
        
        self.backbone, self.feature_dim = self._build_backbone(self.backbone_name, pretrained)
        
        # Freeze weights
        for param in self.backbone.parameters():
            param.requires_grad = False
            
        self.backbone.eval()
        self.backbone.to(self.device)
        logger.info(f"Initialized shared visual encoder ({self.backbone_name}, feature_dim={self.feature_dim}, device={self.device})")

    def _build_backbone(self, name: str, pretrained: bool) -> Tuple[nn.Module, int]:
        """Loads torchvision model and replaces classifier with Identity layer."""
        if name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            model = models.resnet18(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "mobilenet_v3_small":
            weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            model = models.mobilenet_v3_small(weights=weights)
            feat_dim = model.classifier[0].in_features if isinstance(model.classifier, nn.Sequential) else 576
            model.classifier = nn.Identity()
        elif name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            model = models.resnet50(weights=weights)
            feat_dim = model.fc.in_features
            model.fc = nn.Identity()
        elif name == "efficientnet_b0":
            weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = models.efficientnet_b0(weights=weights)
            feat_dim = model.classifier[1].in_features if isinstance(model.classifier, nn.Sequential) else 1280
            model.classifier = nn.Identity()
        elif name == "vit_b_16":
            weights = models.ViT_B_16_Weights.DEFAULT if pretrained else None
            model = models.vit_b_16(weights=weights)
            feat_dim = model.heads.head.in_features if hasattr(model.heads, 'head') else 768
            model.heads = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {name}")
            
        return model, feat_dim

    def get_feature_dim(self) -> int:
        return self.feature_dim

    @torch.no_grad()
    def extract_features(self, roi_tensors: torch.Tensor) -> np.ndarray:
        """
        Processes batched ROI image tensors of shape (N, 3, H, W).
        Returns feature matrix of shape (N, feature_dim).
        """
        if roi_tensors.numel() == 0:
            return np.empty((0, self.feature_dim), dtype=np.float32)
            
        roi_tensors = roi_tensors.to(self.device)
        features = self.backbone(roi_tensors)
        
        if isinstance(features, torch.Tensor):
            features_np = features.cpu().numpy()
        else:
            features_np = features[0].cpu().numpy()
            
        return features_np
