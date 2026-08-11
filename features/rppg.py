import numpy as np
import cv2
from scipy.signal import butter, filtfilt
from scipy.fft import rfft, rfftfreq
from typing import List, Optional, Dict, Any, Tuple
from features.facial_regions import LANDMARK_GROUPS, extract_roi_bboxes
from video.preprocessing import ImagePreprocessor
from utils.logging import setup_logger

logger = setup_logger("RPPGExtractor")

SKIN_ROIS = ["forehead", "left_cheek", "right_cheek"]

class RPPGExtractor:
    """
    Extracts physiological rPPG pulse signals from skin ROIs using Plane-Orthogonal-to-Skin (POS)
    and Chrominance (CHROM) algorithms, computing spectral, temporal, and spatial consistency features.
    """
    def __init__(
        self,
        algorithm: str = "POS",
        low_cutoff: float = 0.7,
        high_cutoff: float = 3.5,
        filter_order: int = 4
    ):
        self.algorithm = algorithm.upper()
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.filter_order = filter_order

    def _extract_skin_mean_rgb(
        self,
        frame_bgr: np.ndarray,
        landmarks_norm: np.ndarray,
        roi_name: str
    ) -> np.ndarray:
        """
        Extracts mean RGB values inside the skin ROI mask.
        """
        h, w = frame_bgr.shape[:2]
        bboxes = extract_roi_bboxes(landmarks_norm, (h, w), margin=0.0)
        bbox = bboxes[roi_name]
        
        xmin, ymin, xmax, ymax = bbox
        crop_bgr = frame_bgr[ymin:ymax, xmin:xmax]
        
        if crop_bgr.size == 0:
            return np.array([0.0, 0.0, 0.0], dtype=np.float32)
            
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        mean_rgb = np.mean(crop_rgb, axis=(0, 1))
        return mean_rgb

    def _pos_algorithm(self, rgb_signals: np.ndarray) -> np.ndarray:
        """
        Plane-Orthogonal-to-Skin (POS) algorithm.
        rgb_signals: (T, 3) RGB mean intensities.
        """
        T = len(rgb_signals)
        mean_rgb = np.mean(rgb_signals, axis=0, keepdims=True) + 1e-8
        rgb_norm = rgb_signals / mean_rgb
        
        s1 = rgb_norm[:, 1] - rgb_norm[:, 2] # G - B
        s2 = rgb_norm[:, 1] + rgb_norm[:, 2] - 2 * rgb_norm[:, 0] # G + B - 2R
        
        std_s1 = np.std(s1)
        std_s2 = np.std(s2) + 1e-8
        alpha = std_s1 / std_s2
        
        h = s1 + alpha * s2
        return h

    def _chrom_algorithm(self, rgb_signals: np.ndarray) -> np.ndarray:
        """
        Chrominance-based (CHROM) algorithm.
        rgb_signals: (T, 3) RGB mean intensities.
        """
        mean_rgb = np.mean(rgb_signals, axis=0, keepdims=True) + 1e-8
        rgb_norm = rgb_signals / mean_rgb
        
        x = 3.0 * rgb_norm[:, 0] - 2.0 * rgb_norm[:, 1]
        y = 1.5 * rgb_norm[:, 0] + rgb_norm[:, 1] - 1.5 * rgb_norm[:, 2]
        
        std_x = np.std(x)
        std_y = np.std(y) + 1e-8
        alpha = std_x / std_y
        
        s = x - alpha * y
        return s

    def _bandpass_filter(self, signal: np.ndarray, fps: float) -> np.ndarray:
        """Applies Butterworth bandpass filter."""
        T = len(signal)
        nyquist = 0.5 * fps
        low = self.low_cutoff / nyquist
        high = self.high_cutoff / nyquist
        
        if low <= 0 or high >= 1.0 or low >= high or T < 8:
            return signal - np.mean(signal)
            
        order = min(self.filter_order, max(1, T // 4))
        try:
            b, a = butter(order, [low, high], btype='bandpass')
            filtered = filtfilt(b, a, signal)
            return filtered
        except Exception:
            return signal - np.mean(signal)

    def extract_rppg_features(
        self,
        frames_bgr: List[np.ndarray],
        landmarks_seq: List[Optional[np.ndarray]],
        fps: float = 8.0
    ) -> Dict[str, Any]:
        """
        Extracts rPPG physiological signals and features across facial skin ROIs.
        Handles failures gracefully without crashing.
        """
        try:
            # Collect RGB signals per ROI across valid frames
            roi_rgb_series: Dict[str, List[np.ndarray]] = {r: [] for r in SKIN_ROIS}
            
            for frame, lms in zip(frames_bgr, landmarks_seq):
                if lms is None:
                    continue
                for roi in SKIN_ROIS:
                    mean_rgb = self._extract_skin_mean_rgb(frame, lms, roi)
                    roi_rgb_series[roi].append(mean_rgb)
                    
            # Verify minimum frame length
            min_T = min([len(v) for v in roi_rgb_series.values()]) if roi_rgb_series else 0
            if min_T < 8:
                # Return graceful fallback default features
                return self._fallback_features()
                
            roi_pulses = {}
            roi_bpms = {}
            roi_snrs = {}
            roi_powers = {}

            for roi in SKIN_ROIS:
                rgb_arr = np.stack(roi_rgb_series[roi], axis=0) # (T, 3)
                if self.algorithm == "CHROM":
                    raw_pulse = self._chrom_algorithm(rgb_arr)
                else:
                    raw_pulse = self._pos_algorithm(rgb_arr)
                    
                pulse = self._bandpass_filter(raw_pulse, fps)
                roi_pulses[roi] = pulse
                
                # Spectral Analysis
                freqs = rfftfreq(len(pulse), 1.0 / fps)
                power_spec = np.abs(rfft(pulse)) ** 2
                
                valid_mask = (freqs >= self.low_cutoff) & (freqs <= self.high_cutoff)
                if np.any(valid_mask):
                    valid_freqs = freqs[valid_mask]
                    valid_power = power_spec[valid_mask]
                    
                    peak_idx = np.argmax(valid_power)
                    best_freq = valid_freqs[peak_idx]
                    bpm = float(best_freq * 60.0)
                    total_power = float(np.sum(valid_power))
                    
                    # SNR estimation around peak frequency
                    peak_mask = (freqs >= best_freq - 0.1) & (freqs <= best_freq + 0.1)
                    signal_energy = np.sum(power_spec[peak_mask])
                    noise_energy = max(total_power - signal_energy, 1e-6)
                    snr = float(10.0 * np.log10(signal_energy / noise_energy))
                else:
                    bpm = 70.0
                    snr = 0.0
                    total_power = 0.0
                    
                roi_bpms[roi] = bpm
                roi_snrs[roi] = snr
                roi_powers[roi] = total_power

            # Cross-Region Physiological Consistency (Spatial Correlation)
            forehead_p = roi_pulses["forehead"]
            left_cheek_p = roi_pulses["left_cheek"]
            right_cheek_p = roi_pulses["right_cheek"]
            
            def safe_corr(a, b):
                if np.std(a) < 1e-6 or np.std(b) < 1e-6:
                    return 0.0
                corr = np.corrcoef(a, b)[0, 1]
                return float(corr) if not np.isnan(corr) else 0.0

            corr_fl = safe_corr(forehead_p, left_cheek_p)
            corr_fr = safe_corr(forehead_p, right_cheek_p)
            corr_lr = safe_corr(left_cheek_p, right_cheek_p)
            
            mean_consistency = float((corr_fl + corr_fr + corr_lr) / 3.0)
            bpm_std = float(np.std([roi_bpms[r] for r in SKIN_ROIS]))
            
            # Form 1D feature vector
            feature_list = []
            for roi in SKIN_ROIS:
                feature_list.extend([
                    roi_bpms[roi],
                    roi_snrs[roi],
                    roi_powers[roi],
                    float(np.std(roi_pulses[roi])),
                    float(np.var(roi_pulses[roi]))
                ])
            feature_list.extend([corr_fl, corr_fr, corr_lr, mean_consistency, bpm_std])
            
            feature_vector = np.array(feature_list, dtype=np.float32)
            
            # Map normalized rPPG consistency score (0..1)
            consistency_score = max(0.0, min(1.0, (mean_consistency + 1.0) / 2.0))
            
            return {
                "feature_vector": feature_vector,
                "rppg_consistency": float(consistency_score),
                "estimated_bpm": float(roi_bpms["forehead"]),
                "roi_bpms": roi_bpms
            }

        except Exception as e:
            logger.warning(f"rPPG extraction encountered error: {e}. Using fallback features.")
            return self._fallback_features()

    def _fallback_features(self) -> Dict[str, Any]:
        """Returns zero fallback feature vector when rPPG fails."""
        # 3 ROIs * 5 stats + 5 spatial consistency stats = 20 features
        feature_vec = np.zeros((20,), dtype=np.float32)
        return {
            "feature_vector": feature_vec,
            "rppg_consistency": 0.5,
            "estimated_bpm": 0.0,
            "roi_bpms": {r: 0.0 for r in SKIN_ROIS}
        }
