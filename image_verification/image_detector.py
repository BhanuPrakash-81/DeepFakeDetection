import os
import sys
import time
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Optional, Tuple, Dict, Any, List

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ImageDeepfakeDetector")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==========================================
# 1. Feature Extractors for Single Images
# ==========================================
class ImageSpatialExtractor(nn.Module):
    """EfficientNet-B0 Spatial Feature Extractor for single image inputs (1280-dim)."""
    def __init__(self):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT
        backbone = models.efficientnet_b0(weights=weights)
        self.feature_extractor = backbone.features
        self.pool = backbone.avgpool
        self.feature_dim = 1280
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            feat = self.feature_extractor(x)
            feat = self.pool(feat)
            feat = torch.flatten(feat, 1)
        return feat


class ImageBiologicalExtractor:
    """Extracts skin tone color distribution and frequency artifacts from single face images (32-dim)."""
    def extract_features(self, face_crop: np.ndarray, forehead_crop: np.ndarray) -> np.ndarray:
        bio_vec = np.zeros(32, dtype=np.float32)
        if face_crop is None or face_crop.size == 0:
            return bio_vec

        # BGR channels
        b_channel, g_channel, r_channel = cv2.split(face_crop)
        mean_r, std_r = float(np.mean(r_channel)), float(np.std(r_channel))
        mean_g, std_g = float(np.mean(g_channel)), float(np.std(g_channel))
        mean_b, std_b = float(np.mean(b_channel)), float(np.std(b_channel))

        # YCrCb Color space skin response
        ycrcb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2YCrCb)
        mean_cr, std_cr = float(np.mean(ycrcb[:, :, 1])), float(np.std(ycrcb[:, :, 1]))
        mean_cb, std_cb = float(np.mean(ycrcb[:, :, 2])), float(np.std(ycrcb[:, :, 2]))

        # High frequency FFT power spectrum on Y channel
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift) + 1e-8
        log_spectrum = np.log(magnitude_spectrum)
        
        bio_vec[0] = mean_r / 255.0
        bio_vec[1] = std_r / 128.0
        bio_vec[2] = mean_g / 255.0
        bio_vec[3] = std_g / 128.0
        bio_vec[4] = mean_b / 255.0
        bio_vec[5] = std_b / 128.0
        bio_vec[6] = mean_cr / 255.0
        bio_vec[7] = std_cr / 128.0
        bio_vec[8] = mean_cb / 255.0
        bio_vec[9] = std_cb / 128.0
        bio_vec[10] = float(np.mean(log_spectrum))
        bio_vec[11] = float(np.std(log_spectrum))

        # Fill remaining slots with spectrum histogram stats
        hist, _ = np.histogram(log_spectrum, bins=20, range=(0, 15))
        hist_norm = hist.astype(np.float32) / (np.sum(hist) + 1e-8)
        bio_vec[12:32] = hist_norm[:20]

        return np.nan_to_num(bio_vec, nan=0.0, posinf=0.0, neginf=0.0)


# ==========================================
# 2. Multimodal Image Deepfake Verification Engine
# ==========================================
class ImageDeepfakeDetector:
    def __init__(self, model_path: Optional[str] = None):
        from model_and_train import MultimodalGatedAttentionAdapter
        from extract_features import FaceProcessor

        self.face_processor = FaceProcessor()
        self.spatial_extractor = ImageSpatialExtractor().to(device)
        self.bio_extractor = ImageBiologicalExtractor()

        default_ckpt = os.path.join("outputs", "checkpoints", "attention_adapter.pth")
        target_ckpt = default_ckpt if model_path is None else model_path

        self.model = MultimodalGatedAttentionAdapter().to(device)
        if os.path.exists(target_ckpt):
            self.model.load_state_dict(torch.load(target_ckpt, map_location=device))
            logger.info(f"Loaded trained checkpoint from '{target_ckpt}'")
        else:
            logger.warning(f"Checkpoint '{target_ckpt}' not found. Operating with base weights.")
        self.model.eval()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def analyze_image(self, image_input: Any) -> Dict[str, Any]:
        """
        Analyzes a single image input (file path, URL, or BGR numpy array).
        Returns a dictionary containing prediction label, confidence, attention weights, and bounding box.
        """
        if isinstance(image_input, str):
            if image_input.startswith("http://") or image_input.startswith("https://"):
                import urllib.request
                resp = urllib.request.urlopen(image_input)
                img_array = np.asarray(bytearray(resp.read()), dtype=np.uint8)
                img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            else:
                if not os.path.exists(image_input):
                    raise FileNotFoundError(f"Image file not found at: '{image_input}'")
                img_bgr = cv2.imread(image_input)
        elif isinstance(image_input, np.ndarray):
            img_bgr = image_input
        else:
            raise ValueError("Unsupported image input type. Expected file path, URL, or numpy BGR array.")

        if img_bgr is None or img_bgr.size == 0:
            raise ValueError("Invalid or corrupted image data.")

        face_crop, forehead_crop, bbox = self.face_processor.extract_face_and_forehead(img_bgr)
        if face_crop is None:
            face_crop = img_bgr
            forehead_crop = img_bgr
            h, w, _ = img_bgr.shape
            bbox = (0, 0, w, h)

        # Spatial Feature Extraction (1280-dim)
        fc_resized = cv2.resize(face_crop, (224, 224))
        fc_rgb = cv2.cvtColor(fc_resized, cv2.COLOR_BGR2RGB)
        t_crop = torch.from_numpy(fc_rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255.0
        t_crop = (t_crop - self.mean) / self.std

        spatial_feat = self.spatial_extractor(t_crop) # (1, 1280)

        # Motion data is unavailable for static images: zero out temporal feature vector (512-dim)
        temporal_feat = torch.zeros((1, 512), dtype=torch.float32, device=device)

        # Biological Feature Extraction (32-dim)
        bio_arr = self.bio_extractor.extract_features(face_crop, forehead_crop)
        bio_feat = torch.from_numpy(bio_arr).unsqueeze(0).float().to(device)

        # Multimodal Gated Attention Inference with explicit static image temporal weight masking
        with torch.no_grad():
            h_S = self.model.proj_spatial(spatial_feat)
            h_T = self.model.proj_temporal(temporal_feat)
            h_B = self.model.proj_biological(bio_feat)
            h_cat = torch.cat([h_S, h_T, h_B], dim=-1)

            gate_logits = self.model.gating_network(h_cat)
            attn_weights = F.softmax(gate_logits, dim=-1)

            # Explicitly disable/zero out temporal attention weight (w_T = 0.0) for static images (no motion data)
            attn_weights[:, 1] = 0.0
            scale = attn_weights[:, 0:1] + attn_weights[:, 2:3] + 1e-8
            attn_weights[:, 0:1] = attn_weights[:, 0:1] / scale
            attn_weights[:, 2:3] = attn_weights[:, 2:3] / scale

            w_S = attn_weights[:, 0:1]
            w_T = attn_weights[:, 1:2]
            w_B = attn_weights[:, 2:3]

            h_S_weighted = h_S * w_S
            h_T_weighted = h_T * w_T
            h_B_weighted = h_B * w_B

            h_fused = torch.cat([h_S_weighted, h_T_weighted, h_B_weighted], dim=-1)
            logits = self.model.cls_head(h_fused).squeeze(-1)

            prob = torch.sigmoid(logits).item()
            weights = attn_weights.squeeze(0).cpu().numpy()

        label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob * 100.0 if label == "FAKE" else (1.0 - prob) * 100.0

        # Render Annotated Visualization Image
        annotated_img = self.render_result_overlay(img_bgr, bbox, label, confidence, weights)

        return {
            "label": label,
            "confidence": confidence,
            "probability": prob,
            "weights": {
                "spatial": float(weights[0]),
                "temporal": float(weights[1]),
                "biological": float(weights[2])
            },
            "bbox": bbox,
            "annotated_image": annotated_img
        }

    def render_result_overlay(
        self,
        img_bgr: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str,
        conf: float,
        weights: np.ndarray
    ) -> np.ndarray:
        result_img = img_bgr.copy()
        xmin, ymin, xmax, ymax = bbox
        color = (0, 0, 255) if label == "FAKE" else (0, 255, 0)

        cv2.rectangle(result_img, (xmin, ymin), (xmax, ymax), color, 3)
        cv2.putText(result_img, f"{label} ({conf:.1f}%)", (xmin, max(30, ymin - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)

        # Draw Metrics HUD Panel
        panel_w, panel_h = 340, 140
        overlay = result_img.copy()
        cv2.rectangle(overlay, (15, 15), (15 + panel_w, 15 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, result_img, 0.25, 0, result_img)

        cv2.putText(result_img, "Image Deepfake Verification HUD", (25, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        modalities = [
            ("Spatial (EfficientNet)", weights[0], (255, 140, 0)),
            ("Frequency / Artifacts", weights[1], (0, 220, 255)),
            ("Biological / Skin Tone", weights[2], (100, 255, 100))
        ]

        y_offset = 60
        for name, weight, bar_color in modalities:
            cv2.putText(result_img, f"{name}: {weight:.2f}", (25, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)

            bar_x = 185
            max_bar_w = 150
            bar_w = int(weight * max_bar_w)
            cv2.rectangle(result_img, (bar_x, y_offset), (bar_x + max_bar_w, y_offset + 14), (60, 60, 60), -1)
            cv2.rectangle(result_img, (bar_x, y_offset), (bar_x + bar_w, y_offset + 14), bar_color, -1)
            y_offset += 26

        return result_img
