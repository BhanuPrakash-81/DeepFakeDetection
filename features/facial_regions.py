import numpy as np
from typing import Dict, Tuple, List
from models.landmark_detector import LandmarkDetector

# MediaPipe 478 FaceLandmarker Landmark Indices
LANDMARK_GROUPS: Dict[str, List[int]] = {
    "left_eye": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246, 33],
    "right_eye": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398, 362],
    "nose": [1, 2, 98, 327, 168, 197, 5, 4, 19, 94, 2, 6, 195, 197],
    "mouth": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88],
    "chin": [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 148, 176, 149, 150, 136, 172, 58, 132, 93],
    "left_eyebrow": [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
    "right_eyebrow": [300, 293, 334, 296, 336, 285, 295, 282, 283, 276],
    # Skin ROIs for rPPG (excluding eyes, mouth, hair)
    "forehead": [10, 67, 109, 338, 297, 337, 108, 69, 151, 9, 8, 107],
    "left_cheek": [50, 101, 118, 119, 120, 205, 206, 207, 117, 123, 147, 187, 207, 216],
    "right_cheek": [280, 330, 347, 348, 349, 425, 426, 427, 346, 352, 376, 411, 427, 436]
}

def extract_roi_bboxes(
    landmarks_norm: np.ndarray,
    frame_shape: Tuple[int, int],
    margin: float = 0.15
) -> Dict[str, Tuple[int, int, int, int]]:
    """
    Extracts pixel bounding boxes (xmin, ymin, xmax, ymax) for each facial ROI region.
    frame_shape: (height, width)
    """
    h, w = frame_shape[:2]
    pts_px = LandmarkDetector.get_pixel_landmarks(landmarks_norm, frame_shape)
    
    bboxes = {}
    for region_name, indices in LANDMARK_GROUPS.items():
        region_pts = pts_px[indices]
        xmin, ymin = np.min(region_pts, axis=0)
        xmax, ymax = np.max(region_pts, axis=0)
        
        bw = xmax - xmin
        bh = ymax - ymin
        
        margin_x = int(bw * margin)
        margin_y = int(bh * margin)
        
        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(w, xmax + margin_x)
        crop_ymax = min(h, ymax + margin_y)
        
        # Ensure minimum 5x5 bbox size
        if crop_xmax <= crop_xmin + 5:
            crop_xmax = min(w, crop_xmin + 10)
        if crop_ymax <= crop_ymin + 5:
            crop_ymax = min(h, crop_ymin + 10)
            
        bboxes[region_name] = (crop_xmin, crop_ymin, crop_xmax, crop_ymax)
        
    return bboxes
