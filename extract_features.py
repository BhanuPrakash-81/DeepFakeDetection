import os
import sys
import glob
import time
import random
import logging
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import List, Tuple, Dict, Any, Optional

from models.landmark_detector import LandmarkDetector
from utils.paths import get_features_output_path, resolve_path, load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ExtractFeatures")

# Setup device dynamically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")


def determine_label_from_path(video_path: str) -> float:
    """
    Smart Path-Parsing Logic for Complex Deepfake Datasets (e.g. DFD / FaceForensics++).
    1. Normalizes absolute path and splits into directory components.
    2. Scans path parts for explicit manipulation markers (reversed order).
    """
    norm_path = os.path.normpath(video_path)
    path_parts = [p.lower() for p in norm_path.split(os.sep)]
    
    fake_keywords = {'manipulated', 'fake', 'altered', 'deepfakes', 'deepfake', 'manipulated_sequences', 'df', 'synth'}
    real_keywords = {'original', 'real', 'youtube', 'actors', 'original_sequences', 'pristine'}
    
    for part in reversed(path_parts):
        is_fake = any(k in part for k in fake_keywords)
        is_real = any(k in part for k in real_keywords)
        
        if is_fake and not is_real:
            return 1.0
        if is_real and not is_fake:
            return 0.0
            
    has_fake = any(any(k in part for k in fake_keywords) for part in path_parts)
    has_real = any(any(k in part for k in real_keywords) for part in path_parts)
    
    if has_fake:
        return 1.0
    elif has_real:
        return 0.0
    else:
        return 1.0 if "fake" in os.path.basename(video_path).lower() else 0.0


# ==========================================
# 1. MediaPipe Face Processing & Augmentation
# ==========================================
class FaceProcessor:
    """
    Isolates facial crops using MediaPipe Face Landmarker and applies real-world noise augmentations.
    """
    def __init__(self, model_path: str = "face_landmarker.task"):
        resolved_task_path = str(resolve_path(model_path))
        try:
            self.detector = LandmarkDetector(model_path=resolved_task_path)
            self.has_detector = True
        except Exception as e:
            logger.warning(f"Could not load LandmarkDetector from '{resolved_task_path}': {e}. Using fallback bounding box.")
            self.has_detector = False

    def extract_face_and_forehead(self, frame_bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        h, w, _ = frame_bgr.shape
        xmin, ymin, xmax, ymax = 0, 0, w, h

        if self.has_detector:
            lms = self.detector.detect_landmarks(frame_bgr)
            if lms is not None:
                xmin, ymin, xmax, ymax = LandmarkDetector.get_face_bbox(lms, (h, w), margin=0.15)
            else:
                xmin, ymin, xmax, ymax = 0, 0, w, h
        
        box_w = xmax - xmin
        box_h = ymax - ymin

        if box_w <= 0 or box_h <= 0:
            return None, None, None

        face_crop = frame_bgr[ymin:ymax, xmin:xmax]

        fh_ymin = ymin
        fh_ymax = ymin + int(box_h * 0.25)
        fh_xmin = xmin + int(box_w * 0.2)
        fh_xmax = xmax - int(box_w * 0.2)

        if fh_ymax > fh_ymin and fh_xmax > fh_xmin:
            forehead_crop = frame_bgr[fh_ymin:fh_ymax, fh_xmin:fh_xmax]
        else:
            forehead_crop = face_crop

        return face_crop, forehead_crop, (xmin, ymin, xmax, ymax)

    @staticmethod
    def apply_noise_augmentations(face_crop: np.ndarray, prob: float = 0.5) -> np.ndarray:
        augmented = face_crop.copy()
        if random.random() < prob:
            quality = random.randint(10, 50)
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
            _, enc_img = cv2.imencode('.jpg', augmented, encode_param)
            if enc_img is not None:
                decoded = cv2.imdecode(enc_img, 1)
                if decoded is not None:
                    augmented = decoded

        if random.random() < prob:
            ksize = random.choice([5, 7, 9, 11])
            augmented = cv2.GaussianBlur(augmented, (ksize, ksize), 0)

        return augmented


# ==========================================
# 2. Extractors: Spatial, Temporal, Biological
# ==========================================
class SpatialExtractor(nn.Module):
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


class TemporalShiftModule(nn.Module):
    def __init__(self, in_channels: int = 1280, out_channels: int = 512, n_div: int = 8):
        super().__init__()
        self.in_channels = in_channels
        self.n_div = n_div
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    def forward(self, seq_features: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            B, T, C = seq_features.shape
            if T <= 1:
                return seq_features.mean(dim=1)[:, :512]
            
            fold = C // self.n_div
            shifted = torch.zeros_like(seq_features)
            shifted[:, :-1, :fold] = seq_features[:, 1:, :fold]
            shifted[:, 1:, fold:2*fold] = seq_features[:, :-1, fold:2*fold]
            shifted[:, :, 2*fold:] = seq_features[:, :, 2*fold:]

            x = shifted.transpose(1, 2)
            feat = self.conv(x).squeeze(-1)
        return feat


class POSrPPGExtractor:
    def __init__(self, low_cutoff: float = 0.7, high_cutoff: float = 3.5):
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff

    def extract_pos_rppg(self, forehead_crops: List[np.ndarray], fps: float = 8.0) -> np.ndarray:
        if len(forehead_crops) < 4:
            return np.zeros(32, dtype=np.float32)

        rgb_series = []
        for crop in forehead_crops:
            if crop is None or crop.size == 0:
                continue
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mean_rgb = np.mean(rgb, axis=(0, 1))
            rgb_series.append(mean_rgb)

        if len(rgb_series) < 4:
            return np.zeros(32, dtype=np.float32)

        C = np.array(rgb_series)
        mean_C = np.mean(C, axis=0, keepdims=True) + 1e-8
        C_norm = C / mean_C

        S1 = C_norm[:, 1] - C_norm[:, 2]
        S2 = C_norm[:, 1] + C_norm[:, 2] - 2.0 * C_norm[:, 0]

        std_S1 = np.std(S1)
        std_S2 = np.std(S2) + 1e-8
        alpha = std_S1 / std_S2

        H = S1 + alpha * S2

        mean_h = float(np.mean(H))
        std_h = float(np.std(H))
        var_h = float(np.var(H))
        skew_h = float(np.mean((H - mean_h) ** 3) / (std_h ** 3 + 1e-8))

        N = len(H)
        freqs = np.fft.rfftfreq(N, d=1.0/fps)
        fft_mag = np.abs(np.fft.rfft(H)) ** 2

        valid_mask = (freqs >= self.low_cutoff) & (freqs <= self.high_cutoff)
        if np.any(valid_mask):
            valid_freqs = freqs[valid_mask]
            valid_mag = fft_mag[valid_mask]
            peak_idx = np.argmax(valid_mag)
            bpm = float(valid_freqs[peak_idx] * 60.0)
            denom = (np.sum(valid_mag) - valid_mag[peak_idx]) + 1e-8
            ratio = max(float(valid_mag[peak_idx] / denom), 1e-8)
            snr = float(10.0 * np.log10(ratio))
        else:
            bpm = 70.0
            snr = 0.0

        bio_vec = np.zeros(32, dtype=np.float32)
        bio_vec[0] = mean_h
        bio_vec[1] = std_h
        bio_vec[2] = var_h
        bio_vec[3] = skew_h
        bio_vec[4] = bpm
        bio_vec[5] = snr
        spec_len = min(26, len(fft_mag))
        bio_vec[6:6+spec_len] = fft_mag[:spec_len]

        bio_vec = np.nan_to_num(bio_vec, nan=0.0, posinf=0.0, neginf=0.0)
        return bio_vec


# ==========================================
# 3. Main Extraction Pipeline Script
# ==========================================
def extract_dataset(data_dir: str, output_path: Optional[str] = None, augment: bool = True, max_frames_per_video: int = 16, max_videos: Optional[int] = None):
    """
    Processes dataset of MP4 videos, extracts fused features, and saves extracted_features.npz.
    Dynamic path resolution handles Local vs Google Colab Drive storage automatically.
    """
    resolved_data_dir = str(resolve_path(data_dir))
    
    if output_path is None:
        target_npz_path = get_features_output_path("extracted_features.npz")
    else:
        target_npz_path = resolve_path(output_path)

    # Ensure parent output directory exists
    target_npz_path.parent.mkdir(parents=True, exist_ok=True)
    out_file = str(target_npz_path)

    # Safety First: Clean existing output file
    if os.path.exists(out_file):
        logger.info(f"Safety Cleanup: Deleting existing output file: {out_file}")
        os.remove(out_file)

    raw_video_paths = []
    for ext in ["*.mp4", "*.MP4", "*.avi", "*.mov", "*.mkv"]:
        raw_video_paths.extend(glob.glob(os.path.join(resolved_data_dir, "**", ext), recursive=True))

    abs_paths = sorted(list({os.path.abspath(p) for p in raw_video_paths}))
    logger.info(f"Discovered and deduplicated dataset: Total {len(abs_paths)} unique videos found.")

    if max_videos is not None and len(abs_paths) > max_videos:
        logger.info(f"Subsampling dataset to max {max_videos} videos for fast extraction.")
        abs_paths = abs_paths[:max_videos]

    if len(abs_paths) == 0:
        logger.warning(f"No video files found in '{resolved_data_dir}'. Generating synthetic dataset.")
        synthetic_dir = str(target_npz_path.parent / "synthetic_videos")
        os.makedirs(synthetic_dir, exist_ok=True)
        abs_paths = create_synthetic_videos(synthetic_dir, num_videos=10)

    face_processor = FaceProcessor(model_path="face_landmarker.task")
    spatial_extractor = SpatialExtractor().to(device)
    temporal_extractor = TemporalShiftModule().to(device)
    rppg_extractor = POSrPPGExtractor()

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    fused_features_list = []
    spatial_features_list = []
    temporal_features_list = []
    biological_features_list = []
    labels_list = []
    processed_paths = []

    for idx, v_path in enumerate(abs_paths):
        if (idx + 1) % 25 == 0 or idx == 0:
            logger.info(f"[{idx+1}/{len(abs_paths)}] Processing: {os.path.basename(v_path)}")
            
        cap = cv2.VideoCapture(v_path)
        face_crops = []
        forehead_crops = []
        frames_count = 0

        while cap.isOpened() and frames_count < max_frames_per_video:
            ret, frame = cap.read()
            if not ret:
                break
            
            face_crop, forehead_crop, _ = face_processor.extract_face_and_forehead(frame)
            if face_crop is not None:
                if augment:
                    face_crop = face_processor.apply_noise_augmentations(face_crop)
                face_crops.append(face_crop)
                forehead_crops.append(forehead_crop)
                frames_count += 1

        cap.release()

        if len(face_crops) == 0:
            continue

        tensor_crops = []
        for fc in face_crops:
            fc_resized = cv2.resize(fc, (224, 224))
            fc_rgb = cv2.cvtColor(fc_resized, cv2.COLOR_BGR2RGB)
            t_crop = torch.from_numpy(fc_rgb).permute(2, 0, 1).float() / 255.0
            tensor_crops.append(t_crop)

        batch_tensors = torch.stack(tensor_crops).to(device)
        batch_tensors = (batch_tensors - mean) / std

        spatial_feats = spatial_extractor(batch_tensors)
        spatial_vec = spatial_feats.mean(dim=0).cpu().numpy()

        seq_tensor = spatial_feats.unsqueeze(0)
        temporal_vec = temporal_extractor(seq_tensor).squeeze(0).cpu().numpy()

        bio_vec = rppg_extractor.extract_pos_rppg(forehead_crops)

        fused_vec = np.concatenate([spatial_vec, temporal_vec, bio_vec], axis=0)
        label = determine_label_from_path(v_path)

        spatial_features_list.append(spatial_vec)
        temporal_features_list.append(temporal_vec)
        biological_features_list.append(bio_vec)
        fused_features_list.append(fused_vec)
        labels_list.append(label)
        processed_paths.append(v_path)

    if len(fused_features_list) == 0:
        logger.error("No valid features extracted from dataset.")
        return

    labels_arr = np.array(labels_list, dtype=np.float32)

    np.savez_compressed(
        out_file,
        X=np.array(fused_features_list, dtype=np.float32),
        X_spatial=np.array(spatial_features_list, dtype=np.float32),
        X_temporal=np.array(temporal_features_list, dtype=np.float32),
        X_biological=np.array(biological_features_list, dtype=np.float32),
        y=labels_arr,
        video_paths=np.array(processed_paths),
        spatial_dim=1280,
        temporal_dim=512,
        biological_dim=32
    )
    logger.info(f"Successfully saved fused features to '{out_file}'. Shape: {np.array(fused_features_list).shape}")


def create_synthetic_videos(output_dir: str, num_videos: int = 10) -> List[str]:
    paths = []
    for i in range(num_videos):
        filename = f"synthetic_{'fake' if i % 2 == 1 else 'real'}_{i:02d}.mp4"
        filepath = os.path.join(output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filepath, fourcc, 10.0, (320, 240))

        for frame_idx in range(16):
            img = np.zeros((240, 320, 3), dtype=np.uint8)
            center = (160 + int(5 * np.sin(frame_idx)), 120)
            cv2.circle(img, center, 50, (200, 180, 150), -1)
            cv2.circle(img, (center[0]-15, center[1]-10), 5, (50, 50, 50), -1)
            cv2.circle(img, (center[0]+15, center[1]-10), 5, (50, 50, 50), -1)
            cv2.ellipse(img, (center[0], center[1]+15), (15, 8), 0, 0, 180, (50, 50, 200), 2)
            out.write(img)

        out.release()
        paths.append(filepath)
    return paths


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 1: Multimodal Feature Extraction Script")
    parser.add_argument("--data_dir", type=str, default="data", help="Path to raw input videos directory")
    parser.add_argument("--out_path", type=str, default=None, help="Custom output .npz path")
    parser.add_argument("--no_augment", action="store_true", help="Disable noise augmentations")
    parser.add_argument("--max_videos", type=int, default=None, help="Maximum videos to extract")
    args = parser.parse_args()

    extract_dataset(args.data_dir, args.out_path, augment=not args.no_augment, max_videos=args.max_videos)
