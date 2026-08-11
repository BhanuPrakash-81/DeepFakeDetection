import cv2
import numpy as np
from typing import Optional, Tuple
from models.landmark_detector import LandmarkDetector
from utils.logging import setup_logger

logger = setup_logger("FaceDetector")

class FaceDetector:
    """
    Unified face detection interface supporting:
    1. 'mediapipe': Uses landmark bounding box (lightweight & fast)
    2. 'opencv': OpenCV Haar cascade detector fallback
    3. 'bypass': Treats full input frame as face (for pre-cropped videos)
    """
    def __init__(self, mode: str = "mediapipe", landmark_detector: Optional[LandmarkDetector] = None):
        self.mode = mode.lower()
        self.landmark_detector = landmark_detector
        
        if self.mode == "opencv":
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.cascade = cv2.CascadeClassifier(cascade_path)

    def detect_face(self, frame_bgr: np.ndarray, landmarks_norm: Optional[np.ndarray] = None) -> Tuple[int, int, int, int]:
        """
        Returns bounding box (xmin, ymin, xmax, ymax) in pixel coordinates.
        """
        h, w = frame_bgr.shape[:2]
        
        if self.mode == "bypass":
            return 0, 0, w, h
            
        if self.mode == "mediapipe":
            if landmarks_norm is not None:
                return LandmarkDetector.get_face_bbox(landmarks_norm, (h, w), margin=0.15)
            elif self.landmark_detector is not None:
                lms = self.landmark_detector.detect_landmarks(frame_bgr)
                if lms is not None:
                    return LandmarkDetector.get_face_bbox(lms, (h, w), margin=0.15)
                    
        if self.mode == "opencv" or (self.mode == "mediapipe" and self.cascade is not None):
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60))
            if len(faces) > 0:
                x, y, bw, bh = faces[0]
                return x, y, x + bw, y + bh

        # Fallback to full frame if detection failed
        return 0, 0, w, h
