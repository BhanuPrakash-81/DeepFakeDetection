import cv2
import os
from typing import Generator, Tuple, Dict, Any
from utils.logging import setup_logger

logger = setup_logger("VideoReader")

class VideoReader:
    """
    Streaming video reader using OpenCV. Reads frames sequentially on demand
    to ensure low memory footprint.
    """
    def __init__(self, video_path: str):
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        self.video_path = video_path

    def get_metadata(self) -> Dict[str, Any]:
        """Reads video metadata (fps, frame_count, width, height, duration)."""
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Unable to open video: {self.video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or fps != fps: # Check 0 or NaN
            fps = 30.0
            
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_sec = frame_count / fps if fps > 0 else 0.0
        cap.release()
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration_sec": duration_sec
        }

    def stream_frames(self) -> Generator[Tuple[int, cv2.Mat], None, None]:
        """
        Yields (frame_index, frame_bgr) frame by frame.
        """
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video stream: {self.video_path}")
            return
            
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            yield frame_idx, frame
            frame_idx += 1
            
        cap.release()
