import os
import sys
import time
import queue
import threading
import logging
import collections
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Optional, Tuple, Any

from extract_features import FaceProcessor, SpatialExtractor, TemporalShiftModule, POSrPPGExtractor
from model_and_train import MultimodalGatedAttentionAdapter
from utils.paths import get_checkpoint_path, resolve_path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveInference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Multithreaded Frame Capture Producer
# ==========================================
class WebcamStream(threading.Thread):
    def __init__(self, source: Any = 0, queue_capacity: int = 2):
        super().__init__()
        self.source = source
        self.stopped = False
        self.is_synthetic = False

        if isinstance(source, str) and ("youtube.com" in source.lower() or "youtu.be" in source.lower()):
            logger.info(f"Extracting YouTube stream URL for '{source}' via yt_dlp...")
            try:
                import yt_dlp
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best',
                    'quiet': True,
                    'no_warnings': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(source, download=False)
                    stream_url = info.get('url', None)
                    if not stream_url and 'requested_formats' in info:
                        stream_url = info['requested_formats'][0]['url']
                    title = info.get('title', 'YouTube Stream')
                    if stream_url:
                        logger.info(f"Successfully resolved YouTube stream for: '{title}'")
                        source = stream_url
            except Exception as e:
                logger.error(f"Failed to extract YouTube stream with yt_dlp: {e}")

        self.cap = cv2.VideoCapture(source)
        self.q = queue.Queue(maxsize=queue_capacity)

        if not self.cap.isOpened():
            logger.warning(f"Could not open stream source '{self.source}'. Falling back to synthetic stream.")
            self.is_synthetic = True

    def run(self):
        frame_idx = 0
        while not self.stopped:
            if self.is_synthetic:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cx = 320 + int(30 * np.sin(frame_idx * 0.1))
                cy = 240 + int(10 * np.cos(frame_idx * 0.1))
                cv2.circle(frame, (cx, cy), 90, (180, 160, 140), -1)
                cv2.circle(frame, (cx - 30, cy - 20), 10, (40, 40, 40), -1)
                cv2.circle(frame, (cx + 30, cy - 20), 10, (40, 40, 40), -1)
                cv2.ellipse(frame, (cx, cy + 30), (25, 12), 0, 0, 180, (50, 50, 200), 3)
                frame_idx += 1
                time.sleep(0.033)
            else:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Video stream reached end.")
                    self.stopped = True
                    break

            try:
                self.q.put(frame, timeout=1.0)
            except queue.Full:
                pass

    def read(self) -> Optional[np.ndarray]:
        try:
            return self.q.get(timeout=0.2)
        except queue.Empty:
            return None

    def stop(self):
        self.stopped = True
        if self.cap and self.cap.isOpened():
            self.cap.release()


# ==========================================
# 2. Live Inference Engine & UI Display
# ==========================================
class LiveInferencePipeline:
    def __init__(self, model_path: Optional[str] = None, seq_len: int = 8):
        self.seq_len = seq_len
        self.face_processor = FaceProcessor()
        self.spatial_extractor = SpatialExtractor().to(device)
        self.temporal_extractor = TemporalShiftModule().to(device)
        self.rppg_extractor = POSrPPGExtractor()

        target_ckpt = get_checkpoint_path("attention_adapter.pth") if model_path is None else resolve_path(model_path)
        str_ckpt = str(target_ckpt)

        self.model = MultimodalGatedAttentionAdapter().to(device)
        if os.path.exists(str_ckpt):
            self.model.load_state_dict(torch.load(str_ckpt, map_location=device))
            logger.info(f"Loaded checkpoint from '{str_ckpt}'")
        else:
            logger.warning(f"Checkpoint '{str_ckpt}' not found. Using initialized model weights.")
        self.model.eval()

        self.frame_buffer = collections.deque(maxlen=seq_len)
        self.forehead_buffer = collections.deque(maxlen=seq_len)

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def process_buffer(self) -> Tuple[str, float, np.ndarray]:
        if len(self.frame_buffer) < self.seq_len:
            return "ANALYZING...", 0.0, np.array([0.33, 0.33, 0.34])

        tensor_crops = []
        for fc in self.frame_buffer:
            fc_resized = cv2.resize(fc, (224, 224))
            fc_rgb = cv2.cvtColor(fc_resized, cv2.COLOR_BGR2RGB)
            t_crop = torch.from_numpy(fc_rgb).permute(2, 0, 1).float() / 255.0
            tensor_crops.append(t_crop)

        batch_tensors = torch.stack(tensor_crops).to(device)
        batch_tensors = (batch_tensors - self.mean) / self.std

        with torch.no_grad():
            spatial_feats = self.spatial_extractor(batch_tensors)
            spatial_vec = spatial_feats.mean(dim=0).unsqueeze(0)

            seq_tensor = spatial_feats.unsqueeze(0)
            temporal_vec = self.temporal_extractor(seq_tensor)

            bio_arr = self.rppg_extractor.extract_pos_rppg(list(self.forehead_buffer))
            bio_arr = np.nan_to_num(bio_arr, nan=0.0, posinf=0.0, neginf=0.0)
            bio_vec = torch.from_numpy(bio_arr).unsqueeze(0).float().to(device)

            fused_vector = torch.cat([spatial_vec, temporal_vec, bio_vec], dim=-1)
            fused_vector = torch.nan_to_num(fused_vector, nan=0.0, posinf=0.0, neginf=0.0)

            logits, attn_weights = self.model(fused_vector)
            prob = torch.sigmoid(logits).item()
            weights = attn_weights.squeeze(0).cpu().numpy()

        label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob * 100.0 if label == "FAKE" else (1.0 - prob) * 100.0

        return label, confidence, weights

    def render_overlay(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], label: str, conf: float, weights: np.ndarray) -> np.ndarray:
        display_frame = frame.copy()

        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            color = (0, 0, 255) if label == "FAKE" else ((0, 255, 0) if label == "REAL" else (255, 255, 0))
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(display_frame, f"{label} ({conf:.1f}%)", (xmin, max(20, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        panel_w, panel_h = 320, 130
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        cv2.putText(display_frame, "Dynamic Modality Attention", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        modalities = [("Spatial (S)", weights[0], (255, 120, 0)),
                      ("Temporal (T)", weights[1], (0, 220, 255)),
                      ("Biological (B)", weights[2], (100, 255, 100))]

        y_offset = 50
        for name, weight, bar_color in modalities:
            cv2.putText(display_frame, f"{name}: {weight:.2f}", (20, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            
            bar_x = 150
            max_bar_w = 160
            bar_w = int(weight * max_bar_w)
            cv2.rectangle(display_frame, (bar_x, y_offset), (bar_x + max_bar_w, y_offset + 14), (60, 60, 60), -1)
            cv2.rectangle(display_frame, (bar_x, y_offset), (bar_x + bar_w, y_offset + 14), bar_color, -1)
            y_offset += 25

        return display_frame


def run_live_inference(source: Any = 0, model_path: Optional[str] = None):
    pipeline = LiveInferencePipeline(model_path=model_path)
    stream = WebcamStream(source=source)
    stream.start()
    logger.info("Live Video Inference initialized.")

    prev_time = time.time()
    fps = 0.0
    label, conf, weights = "ANALYZING...", 0.0, np.array([0.33, 0.33, 0.34])
    last_reported_label = None

    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue

        face_crop, forehead_crop, bbox = pipeline.face_processor.extract_face_and_forehead(frame)

        if face_crop is not None:
            pipeline.frame_buffer.append(face_crop)
            pipeline.forehead_buffer.append(forehead_crop)

            if len(pipeline.frame_buffer) == pipeline.seq_len:
                label, conf, weights = pipeline.process_buffer()
                if label != last_reported_label:
                    logger.info(f"Live Detection: {label} (Confidence: {conf:.2f}%) | Weights -> S: {weights[0]:.2f}, T: {weights[1]:.2f}, B: {weights[2]:.2f}")
                    last_reported_label = label

        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-4)
        prev_time = curr_time

        display_frame = pipeline.render_overlay(frame, bbox, label, conf, weights)
        cv2.putText(display_frame, f"Pipeline FPS: {fps:.1f}", (display_frame.shape[1] - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        try:
            cv2.imshow("Multimodal Deepfake Detection - Dynamic Modality Neglect", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break
        except Exception as e:
            # Fallback for headless environments without GUI display
            pass

    stream.stop()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    logger.info(f"Live Video Inference completed. Final Classification: {label} ({conf:.2f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3: Live Video Inference Script")
    parser.add_argument("--source", type=str, default="0", help="Webcam ID or video path")
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_live_inference(source=src, model_path=args.model)
