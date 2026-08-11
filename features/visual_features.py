import numpy as np
import torch
import cv2
from typing import List, Dict, Tuple, Optional, Any
from models.visual_encoder import SharedVisualEncoder
from models.landmark_detector import LandmarkDetector
from features.facial_regions import extract_roi_bboxes
from video.preprocessing import ImagePreprocessor
from utils.logging import setup_logger

logger = setup_logger("VisualFeatureExtractor")

VISUAL_REGIONS = ["left_eye", "right_eye", "nose", "mouth", "chin"]

class VisualFeatureExtractor:
    """
    Extracts deep visual features from facial ROIs across video frames using
    a single shared pretrained CNN backbone in batched forward passes.
    """
    def __init__(self, encoder: SharedVisualEncoder, preprocessor: Optional[ImagePreprocessor] = None):
        self.encoder = encoder
        self.preprocessor = preprocessor or ImagePreprocessor(target_size=(112, 112))
        self.feature_dim = encoder.get_feature_dim()

    def extract_video_visual_features(
        self,
        frames_bgr: List[np.ndarray],
        landmarks_seq: List[Optional[np.ndarray]]
    ) -> Dict[str, Any]:
        """
        Processes a sequence of frames and their detected landmarks.
        Returns:
            Dictionary containing:
            - 'region_embeddings': Dict mapping region_name -> np.ndarray of shape (num_valid_frames, feature_dim)
            - 'feature_vector': Aggregated flattened numpy array of temporal summary features
            - 'region_anomaly_scores': Dict mapping region_name -> float scalar anomaly indicator
        """
        # Collect all ROI patches across frames for batched processing
        frame_patch_map = [] # List of tuples: (frame_idx, region_name, patch_bgr)
        
        for f_idx, (frame, lms) in enumerate(zip(frames_bgr, landmarks_seq)):
            if lms is None:
                continue
            h, w = frame.shape[:2]
            bboxes = extract_roi_bboxes(lms, (h, w), margin=0.15)
            
            for reg in VISUAL_REGIONS:
                bbox = bboxes[reg]
                patch = self.preprocessor.crop_roi(frame, bbox, margin=0.0)
                frame_patch_map.append((f_idx, reg, patch))
                
        if not frame_patch_map:
            # Fallback if no landmarks detected in entire sequence
            dummy_dim = self.feature_dim * len(VISUAL_REGIONS) * 5 # (mean, std, max, min, var)
            return {
                "region_embeddings": {r: np.zeros((0, self.feature_dim), dtype=np.float32) for r in VISUAL_REGIONS},
                "feature_vector": np.zeros((dummy_dim,), dtype=np.float32),
                "region_anomaly_scores": {r: 0.0 for r in VISUAL_REGIONS}
            }
            
        # Prepare batched ROI tensors
        patches = [item[2] for item in frame_patch_map]
        batch_tensors = self.preprocessor.batch_preprocess_patches(patches)
        
        # Single shared CNN batched forward pass
        all_embeddings = self.encoder.extract_features(batch_tensors)
        
        # Organize embeddings by region across time
        region_embs: Dict[str, List[np.ndarray]] = {r: [] for r in VISUAL_REGIONS}
        for (f_idx, reg, _), emb in zip(frame_patch_map, all_embeddings):
            region_embs[reg].append(emb)
            
        region_embeddings_np = {}
        summary_features = []
        region_anomaly_scores = {}
        
        for reg in VISUAL_REGIONS:
            embs_list = region_embs[reg]
            if len(embs_list) > 0:
                arr = np.stack(embs_list, axis=0) # Shape: (T, feat_dim)
                mean_feat = np.mean(arr, axis=0)
                std_feat = np.std(arr, axis=0)
                max_feat = np.max(arr, axis=0)
                min_feat = np.min(arr, axis=0)
                var_feat = np.var(arr, axis=0)
                
                # Anomaly proxy metric: temporal variance magnitude across ROI embeddings
                anomaly_score = float(np.mean(var_feat))
            else:
                arr = np.zeros((0, self.feature_dim), dtype=np.float32)
                mean_feat = np.zeros((self.feature_dim,), dtype=np.float32)
                std_feat = np.zeros((self.feature_dim,), dtype=np.float32)
                max_feat = np.zeros((self.feature_dim,), dtype=np.float32)
                min_feat = np.zeros((self.feature_dim,), dtype=np.float32)
                var_feat = np.zeros((self.feature_dim,), dtype=np.float32)
                anomaly_score = 0.0
                
            region_embeddings_np[reg] = arr
            region_anomaly_scores[reg] = anomaly_score
            summary_features.extend([mean_feat, std_feat, max_feat, min_feat, var_feat])
            
        feature_vector = np.concatenate([f.ravel() for f in summary_features], axis=0).astype(np.float32)
        
        return {
            "region_embeddings": region_embeddings_np,
            "feature_vector": feature_vector,
            "region_anomaly_scores": region_anomaly_scores
        }
