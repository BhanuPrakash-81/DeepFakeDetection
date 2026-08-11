import os
import urllib.request
import cv2
import numpy as np
from typing import Optional, List, Tuple, Dict, Any
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.logging import setup_logger

logger = setup_logger("LandmarkDetector")

DEFAULT_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

class LandmarkDetector:
    """
    Pretrained MediaPipe Face Landmarker wrapper.
    Extracts 478 3D facial landmarks per frame.
    Weights are frozen (pretrained TFLite model from Google MediaPipe).
    """
    def __init__(self, model_path: str = "models/face_landmarker.task", model_url: str = DEFAULT_MODEL_URL):
        self.model_path = model_path
        self.model_url = model_url
        self._ensure_model_exists()
        
        base_options = python.BaseOptions(model_asset_path=self.model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        logger.info(f"Initialized MediaPipe Face Landmarker from {self.model_path}")

    def _ensure_model_exists(self):
        """Downloads face_landmarker.task if not present locally."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            logger.info(f"Downloading pretrained MediaPipe Face Landmarker from {self.model_url}...")
            urllib.request.urlretrieve(self.model_url, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

    def detect_landmarks(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """
        Detects facial landmarks on a single BGR frame.
        Returns:
            landmarks: np.ndarray of shape (478, 3) normalized in range [0, 1] for x, y, z.
                       Returns None if no face detected.
        """
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        result = self.landmarker.detect(mp_image)
        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return None
            
        first_face = result.face_landmarks[0]
        coords = np.array([[lm.x, lm.y, lm.z] for lm in first_face], dtype=np.float32)
        return coords

    @staticmethod
    def get_pixel_landmarks(landmarks_norm: np.ndarray, frame_shape: Tuple[int, int]) -> np.ndarray:
        """
        Converts normalized landmark coordinates [0, 1] to pixel coordinates (x_px, y_px).
        frame_shape: (height, width)
        """
        h, w = frame_shape[:2]
        pixel_coords = np.zeros((len(landmarks_norm), 2), dtype=np.int32)
        pixel_coords[:, 0] = np.clip((landmarks_norm[:, 0] * w).astype(np.int32), 0, w - 1)
        pixel_coords[:, 1] = np.clip((landmarks_norm[:, 1] * h).astype(np.int32), 0, h - 1)
        return pixel_coords

    @staticmethod
    def get_face_bbox(landmarks_norm: np.ndarray, frame_shape: Tuple[int, int], margin: float = 0.1) -> Tuple[int, int, int, int]:
        """
        Extracts face bounding box (xmin, ymin, xmax, ymax) in pixel coordinates from landmarks with margin.
        """
        h, w = frame_shape[:2]
        pts_px = LandmarkDetector.get_pixel_landmarks(landmarks_norm, frame_shape)
        
        xmin, ymin = np.min(pts_px, axis=0)
        xmax, ymax = np.max(pts_px, axis=0)
        
        bw = xmax - xmin
        bh = ymax - ymin
        
        margin_x = int(bw * margin)
        margin_y = int(bh * margin)
        
        crop_xmin = max(0, xmin - margin_x)
        crop_ymin = max(0, ymin - margin_y)
        crop_xmax = min(w, xmax + margin_x)
        crop_ymax = min(h, ymax + margin_y)
        
        return crop_xmin, crop_ymin, crop_xmax, crop_ymax
