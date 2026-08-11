# Dynamic Modality Neglect in Multimodal Deepfake Detection

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-green.svg)](https://opencv.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-Tasks--1.0-blue)](https://developers.google.com/mediapipe)

A production-ready, lightweight PyTorch framework for **Multimodal Deepfake Detection** with real-time video inference and academic evaluation. The core contribution is a **Gated Attention Network** that dynamically evaluates incoming feature vectors across **Spatial**, **Temporal**, and **Biological** modalities to perform **Dynamic Modality Neglect**.

---

## 📌 Abstract

In real-world deployment scenarios, multimodal deepfake detectors face severe noise and degradation. Heavy JPEG compression destroys subtle skin color variations required for Remote Photoplethysmography (rPPG), while spatial blur or frame drops corrupt motion feature continuity. Standard fusion techniques (e.g. concatenation, mean pooling) suffer from severe performance degradation when any single modality becomes unreliable.

This repository introduces a **Dynamic Modality Neglect Gated Attention Network** that evaluates spatial, temporal, and physiological representations in a shared latent space. The network dynamically learns Softmax attention weights $[w_S, w_T, w_B]$ summing to $1.0$. When a modality is corrupted (e.g., JPEG compression destroying rPPG signals), the gating mechanism assigns its attention weight near $0.0$, delegating prediction authority to remaining healthy branches.

```
Incoming Stream ──► [ Spatial (EfficientNet-B0) ] ──► h_S (256D) ┐
                 ──► [ Temporal (TSM)           ] ──► h_T (256D) ├─► Gating Network ──► [w_S, w_T, w_B] ──► Classifier ──► REAL / FAKE
                 ──► [ Biological (POS-rPPG)    ] ──► h_B (256D) ┘
```

---

## 📂 Repository Structure

```
.
├── extract_features.py     # Stage 1: Dataset deduplication, MediaPipe face crop, noise augmentations, & feature fusion (.npz)
├── model_and_train.py      # Stage 2: MultimodalGatedAttentionAdapter PyTorch architecture & training loop
├── live_inference.py       # Stage 3: Multithreaded real-time webcam inference with dynamic attention weight display
├── evaluate_metrics.py     # Stage 4: Academic metric evaluation (AUC-ROC, EER calculation, F1, Precision, Recall, Latency)
├── face_landmarker.task    # Pretrained MediaPipe Face Landmarker model asset
├── requirements.txt        # Python dependency requirements
└── README.md               # Repository documentation
```

---

## 🔬 Pipeline Architecture & Modality Branches

The pipeline extracts representations across three specialized branches before dynamic gating:

1. **Spatial Branch (1280-D)**:  
   - Backbone: Pretrained, frozen **EfficientNet-B0** (`torchvision.models.efficientnet_b0`).
   - Extracts fine-grained spatial compression artifacts, blending boundaries, and facial unnaturalness from 224x224 facial crops.

2. **Temporal Branch (512-D)**:  
   - Backbone: **Temporal Shift Module (TSM)**.
   - Shifts temporal channel slices across consecutive frames to capture inter-frame motion inconsistency without extra computational FLOPs.

3. **Biological Branch (32-D)**:  
   - Algorithm: **Plane-Orthogonal-to-Skin (POS) rPPG**.
   - Tracks blood volume pulse variations from the MediaPipe forehead ROI across temporal frames, extracting heart rate (BPM), spectral energy, SNR, and signal variance.

### Dynamic Gating Formulation
Each raw feature vector $x_S \in \mathbb{R}^{1280}$, $x_T \in \mathbb{R}^{512}$, $x_B \in \mathbb{R}^{32}$ is projected into a shared 256-dimensional latent space:
$$h_S = \text{GELU}(\text{BatchNorm1d}(W_S x_S)), \quad h_T = \text{GELU}(\text{BatchNorm1d}(W_T x_T)), \quad h_B = \text{GELU}(\text{BatchNorm1d}(W_B x_B))$$

The concatenated latent representations $h_{\text{cat}} = [h_S; h_T; h_B] \in \mathbb{R}^{768}$ pass through a 2-layer MLP gating network:
$$[w_S, w_T, w_B] = \text{Softmax}(W_{g2} \cdot \text{GELU}(W_{g1} h_{\text{cat}}))$$

Modulated features $\hat{h}_{\text{fused}} = [w_S \cdot h_S; w_T \cdot h_T; w_B \cdot h_B]$ are evaluated by the final binary classification head.

---

## ⚡ Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/username/multimodal-deepfake-detection.git
   cd multimodal-deepfake-detection
   ```

2. **Set up Virtual Environment**:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Linux/macOS:
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage (The 4 Pipeline Stages)

### Stage 1: Feature Extraction
Discovers and deduplicates raw MP4 videos, applies random JPEG compression & Gaussian blur noise augmentations, extracts spatial/temporal/biological features, and exports compressed `.npz` feature vectors.

```bash
python extract_features.py --data_dir data --out_dir features_out --max_videos 100
```

### Stage 2: Model Training
Trains the `MultimodalGatedAttentionAdapter` using `BCEWithLogitsLoss` and `AdamW` with cosine annealing. Saves the optimal state dict to `attention_adapter.pth`.

```bash
python model_and_train.py --features features_out/extracted_features.npz --save_path attention_adapter.pth --epochs 30
```

### Stage 3: Live Video Inference
Runs real-time webcam inference using a multithreaded producer-consumer architecture (`queue.Queue`). Displays face bounding boxes, classification confidence, and a live visual readout of dynamic attention weights $[w_S, w_T, w_B]$.

```bash
# Test with default webcam (ID 0)
python live_inference.py --source 0 --model attention_adapter.pth

# Test with a video file
python live_inference.py --source video_sample.mp4 --model attention_adapter.pth
```

### Stage 4: Academic Metrics & Evaluation
Evaluates the trained model on test feature vectors. Computes AUC-ROC, explicit Equal Error Rate (EER), F1-Score, Precision, Recall, Confusion Matrix, and latency/FPS throughput benchmarks, saving `roc_curve.png`.

```bash
python evaluate_metrics.py --features features_out/extracted_features.npz --model attention_adapter.pth --roc_plot roc_curve.png
```

---

## 📊 Academic Results Sample Output

```text
======================================================================
      ACADEMIC EVALUATION REPORT: MULTIMODAL DEEPFAKE DETECTION      
======================================================================
 Dataset Size        : 100 video feature samples
 AUC-ROC Score       : 0.9842
 Equal Error Rate    : 3.20% (at threshold tau = 0.4812)
 F1-Score            : 0.9600
 Precision           : 0.9600
 Recall              : 0.9600
----------------------------------------------------------------------
 Mean Attention Weights Allocation (Dynamic Gating):
   Spatial Branch (EfficientNet-B0) : 0.4215
   Temporal Branch (TSM)            : 0.3840
   Biological Branch (POS-rPPG)     : 0.1945
----------------------------------------------------------------------
 Inference Throughput Benchmark:
   Average Latency per Sample      : 0.12 ms
   Inference Throughput (FPS)       : 8333.3 samples/sec
======================================================================
```

---

## 📝 Citation

If you find this codebase or the Dynamic Modality Neglect mechanism useful in your research, please cite our upcoming work:

```bibtex
@inproceedings{deepfake2026dynamic,
  title={Dynamic Modality Neglect in Multimodal Deepfake Detection},
  author={Prakash, Bhanu and Research Collaborators},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for details.
