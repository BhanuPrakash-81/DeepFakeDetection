import numpy as np
import cv2
from typing import Generator, List, Tuple
from video.reader import VideoReader
from utils.logging import setup_logger

logger = setup_logger("VideoSampler")

class VideoSampler:
    """
    Downsamples video frames to a target FPS and caps the maximum number of frames
    to optimize computation and memory usage.
    """
    def __init__(self, target_fps: float = 8.0, max_frames: int = 120):
        self.target_fps = target_fps
        self.max_frames = max_frames

    def sample_video(self, reader: VideoReader) -> Tuple[List[int], List[np.ndarray], float]:
        """
        Samples frames from VideoReader.
        Returns:
            sampled_indices: List of original frame indices sampled
            sampled_frames: List of BGR numpy frame arrays
            effective_fps: Actual sample rate (FPS) of the extracted sequence
        """
        meta = reader.get_metadata()
        orig_fps = meta["fps"]
        
        if orig_fps <= 0:
            orig_fps = 30.0
            
        step = max(1, int(round(orig_fps / self.target_fps)))
        effective_fps = orig_fps / step
        
        sampled_indices = []
        sampled_frames = []
        
        for frame_idx, frame in reader.stream_frames():
            if frame_idx % step == 0:
                sampled_indices.append(frame_idx)
                sampled_frames.append(frame)
                if len(sampled_frames) >= self.max_frames:
                    break
                    
        logger.debug(f"Sampled {len(sampled_frames)} frames from {meta['frame_count']} original frames (Effective FPS: {effective_fps:.2f})")
        return sampled_indices, sampled_frames, effective_fps
