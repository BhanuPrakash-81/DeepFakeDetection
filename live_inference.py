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
import mediapipe as mp

from extract_features import FaceProcessor, SpatialExtractor, TemporalShiftModule, POSrPPGExtractor
from model_and_train import MultimodalGatedAttentionAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("LiveInference")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. Multithreaded Frame Capture Producer
# ==========================================
class WebcamStream(threading.Thread):
    """
    Decouples webcam frame capture loop from heavy PyTorch inference loop using queue.Queue.
    Ensures zero camera lag regardless of model processing latency.
    """
    def __init__(self, source: Any = 0, queue_capacity: int = 2):
        super().__init__()
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.q = queue.Queue(maxsize=queue_capacity)
        self.stopped = False
        self.is_synthetic = False

        if not self.cap.isOpened():
            logger.warning(f"Could not open camera source '{source}'. Falling back to synthetic stream.")
            self.is_synthetic = True

    def run(self):
        frame_idx = 0
        while not self.stopped:
            if self.is_synthetic:
                # Generate synthetic test frame with animated circle
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cx = 320 + int(30 * np.sin(frame_idx * 0.1))
                cy = 240 + int(10 * np.cos(frame_idx * 0.1))
                cv2.circle(frame, (cx, cy), 90, (180, 160, 140), -1) # face
                cv2.circle(frame, (cx - 30, cy - 20), 10, (40, 40, 40), -1)
                cv2.circle(frame, (cx + 30, cy - 20), 10, (40, 40, 40), -1)
                cv2.ellipse(frame, (cx, cy + 30), (25, 12), 0, 0, 180, (50, 50, 200), 3)
                frame_idx += 1
                time.sleep(0.033) # 30 FPS
            else:
                ret, frame = self.cap.read()
                if not ret:
                    logger.info("Video stream reached end.")
                    self.stopped = True
                    break

            if not self.q.full():
                self.q.put(frame)

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
    """
    Real-time Multimodal Deepfake Inference Pipeline with Dynamic Attention Weight Visualizer.
    """
    def __init__(self, model_path: str = "attention_adapter.pth", seq_len: int = 8):
        self.seq_len = seq_len
        self.face_processor = FaceProcessor()
        self.spatial_extractor = SpatialExtractor().to(device)
        self.temporal_extractor = TemporalShiftModule().to(device)
        self.rppg_extractor = POSrPPGExtractor()

        # Load Trained Gated Attention Model
        self.model = MultimodalGatedAttentionAdapter().to(device)
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info(f"Loaded checkpoint from '{model_path}'")
        else:
            logger.warning(f"Checkpoint '{model_path}' not found. Using initialized model weights.")
        self.model.eval()

        # Rolling Buffer (stores sequence of 8 frames)
        self.frame_buffer = collections.deque(maxlen=seq_len)
        self.forehead_buffer = collections.deque(maxlen=seq_len)

        # PyTorch Normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

    def process_buffer(self) -> Tuple[str, float, np.ndarray]:
        """
        Runs feature extractors and adapter model on rolling 8-frame buffer.
        Returns label, confidence percentage, and dynamic attention weights.
        """
        if len(self.frame_buffer) < self.seq_len:
            return "ANALYZING...", 0.0, np.array([0.33, 0.33, 0.34])

        # Prepare Spatial Tensors
        tensor_crops = []
        for fc in self.frame_buffer:
            fc_resized = cv2.resize(fc, (224, 224))
            fc_rgb = cv2.cvtColor(fc_resized, cv2.COLOR_BGR2RGB)
            t_crop = torch.from_numpy(fc_rgb).permute(2, 0, 1).float() / 255.0
            tensor_crops.append(t_crop)

        batch_tensors = torch.stack(tensor_crops).to(device)
        batch_tensors = (batch_tensors - self.mean) / self.std

        with torch.no_grad():
            # Spatial Feature (EfficientNet-B0)
            spatial_feats = self.spatial_extractor(batch_tensors)
            spatial_vec = spatial_feats.mean(dim=0).unsqueeze(0) # (1, 1280)

            # Temporal Feature (TSM)
            seq_tensor = spatial_feats.unsqueeze(0) # (1, T, 1280)
            temporal_vec = self.temporal_extractor(seq_tensor) # (1, 512)

            # Biological Feature (POS-rPPG)
            bio_arr = self.rppg_extractor.extract_pos_rppg(list(self.forehead_buffer))
            bio_vec = torch.from_numpy(bio_arr).unsqueeze(0).float().to(device) # (1, 32)

            # Fuse Vector
            fused_vector = torch.cat([spatial_vec, temporal_vec, bio_vec], dim=-1) # (1, 1824)

            # Gated Attention Forward Pass
            logits, attn_weights = self.model(fused_vector)
            prob = torch.sigmoid(logits).item()
            weights = attn_weights.squeeze(0).cpu().numpy()

        label = "FAKE" if prob >= 0.5 else "REAL"
        confidence = prob * 100.0 if label == "FAKE" else (1.0 - prob) * 100.0

        return label, confidence, weights

    def render_overlay(self, frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], label: str, conf: float, weights: np.ndarray) -> np.ndarray:
        """
        Draws bounding box, label text, and visual dynamic attention weight bars on frame.
        """
        display_frame = frame.copy()
        h, w, _ = display_frame.shape

        # Draw Bounding Box around Face
        if bbox is not None:
            xmin, ymin, xmax, ymax = bbox
            color = (0, 0, 255) if label == "FAKE" else ((0, 255, 0) if label == "REAL" else (255, 255, 0))
            cv2.rectangle(display_frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(display_frame, f"{label} ({conf:.1f}%)", (xmin, max(20, ymin - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # Render Dynamic Modality Attention Weight Panel on Top Left
        panel_w, panel_h = 320, 130
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (10, 10), (10 + panel_w, 10 + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)

        cv2.putText(display_frame, "Dynamic Modality Attention", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        # Bar Charts for [Spatial, Temporal, Biological]
        modalities = [("Spatial (S)", weights[0], (255, 120, 0)),
                      ("Temporal (T)", weights[1], (0, 220, 255)),
                      ("Biological (B)", weights[2], (100, 255, 100))]

        y_offset = 50
        for name, weight, bar_color in modalities:
            cv2.putText(display_frame, f"{name}: {weight:.2f}", (20, y_offset + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1, cv2.LINE_AA)
            
            # Progress Bar Fill
            bar_x = 150
            max_bar_w = 160
            bar_w = int(weight * max_bar_w)
            cv2.rectangle(display_frame, (bar_x, y_offset), (bar_x + max_bar_w, y_offset + 14), (60, 60, 60), -1)
            cv2.rectangle(display_frame, (bar_x, y_offset), (bar_x + bar_w, y_offset + 14), bar_color, -1)
            y_offset += 25

        return display_frame


def run_live_inference(source: Any = 0, model_path: str = "attention_adapter.pth"):
    """
    Main loop running multithreaded live inference pipeline.
    """
    stream = WebcamStream(source=source)
    stream.start()

    pipeline = LiveInferencePipeline(model_path=model_path)
    logger.info("Live Video Inference initialized. Press 'q' or 'Esc' to exit.")

    prev_time = time.time()
    fps = 0.0

    label, conf, weights = "ANALYZING...", 0.0, np.array([0.33, 0.33, 0.34])

    while not stream.stopped:
        frame = stream.read()
        if frame is None:
            continue

        # Extract Face
        face_crop, forehead_crop, bbox = pipeline.face_processor.extract_face_and_forehead(frame)

        if face_crop is not None:
            pipeline.frame_buffer.append(face_crop)
            pipeline.forehead_buffer.append(forehead_crop)

            # Compute prediction when buffer is full
            if len(pipeline.frame_buffer) == pipeline.seq_len:
                label, conf, weights = pipeline.process_buffer()

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / max(curr_time - prev_time, 1e-4)
        prev_time = curr_time

        # Render UI
        display_frame = pipeline.render_overlay(frame, bbox, label, conf, weights)
        cv2.putText(display_frame, f"Pipeline FPS: {fps:.1f}", (display_frame.shape[1] - 170, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow("Multimodal Deepfake Detection - Dynamic Modality Neglect", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    stream.stop()
    cv2.destroyAllWindows()
    logger.info("Live Video Inference closed.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Task 3: Live Video Inference Script")
    parser.add_argument("--source", type=str, default="0", help="Webcam device ID (e.g. 0) or path to video file")
    parser.add_argument("--model", type=str, default="attention_adapter.pth", help="Path to trained PyTorch checkpoint")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_live_inference(source=src, model_path=args.model)
