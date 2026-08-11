import numpy as np
from scipy.signal import savgol_filter
from scipy.fft import rfft, rfftfreq
from typing import List, Optional, Dict, Any, Tuple
from features.facial_regions import LANDMARK_GROUPS
from utils.logging import setup_logger

logger = setup_logger("BiomechanicsExtractor")

BIOMECHANIC_REGIONS = [
    "mouth", "chin", "left_eye", "right_eye", "left_eyebrow", "right_eyebrow"
]

class BiomechanicsExtractor:
    """
    Computes mathematical kinematic motion features (Displacement, Velocity, Acceleration, Jerk)
    and frequency-domain metrics from 3D facial landmark sequences over time.
    """
    def __init__(self, smoothing_window: int = 7, poly_order: int = 2):
        self.smoothing_window = smoothing_window
        self.poly_order = poly_order

    def _impute_missing_landmarks(self, landmarks_seq: List[Optional[np.ndarray]]) -> np.ndarray:
        """
        Interpolates missing frames where landmark detection failed.
        Returns array of shape (T, 478, 3).
        """
        T = len(landmarks_seq)
        if T == 0:
            return np.zeros((0, 478, 3), dtype=np.float32)
            
        # Find first valid frame
        first_valid = None
        for lms in landmarks_seq:
            if lms is not None:
                first_valid = lms
                break
                
        if first_valid is None:
            # Entire video has no landmarks
            return np.zeros((T, 478, 3), dtype=np.float32)
            
        clean_seq = []
        last_valid = first_valid
        for lms in landmarks_seq:
            if lms is not None:
                last_valid = lms
                clean_seq.append(lms)
            else:
                clean_seq.append(last_valid)
                
        return np.stack(clean_seq, axis=0) # (T, 478, 3)

    def _smooth_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Applies Savitzky-Golay filter to smooth (T, N, 3) coordinates.
        """
        T = trajectory.shape[0]
        if T < 4:
            return trajectory
            
        win = self.smoothing_window
        if win >= T:
            win = T if T % 2 != 0 else T - 1
        if win <= self.poly_order:
            win = self.poly_order + 2 if (self.poly_order + 2) % 2 != 0 else self.poly_order + 3
        if win > T:
            return trajectory
            
        try:
            smoothed = savgol_filter(trajectory, window_length=win, polyorder=self.poly_order, axis=0)
            return smoothed
        except Exception:
            return trajectory

    def compute_stats(self, signal: np.ndarray, fps: float) -> List[float]:
        """
        Computes 7 statistical metrics for a 1D kinematic signal:
        [mean, std, max, min, rms, variance, spectral_energy]
        """
        if len(signal) == 0:
            return [0.0] * 7
            
        mean_val = float(np.mean(signal))
        std_val = float(np.std(signal))
        max_val = float(np.max(signal))
        min_val = float(np.min(signal))
        rms_val = float(np.sqrt(np.mean(signal ** 2)))
        var_val = float(np.var(signal))
        
        # FFT spectral energy
        if len(signal) >= 4:
            fft_vals = np.abs(rfft(signal - mean_val))
            spectral_energy = float(np.sum(fft_vals ** 2) / len(fft_vals))
        else:
            spectral_energy = 0.0
            
        return [mean_val, std_val, max_val, min_val, rms_val, var_val, spectral_energy]

    def extract_features(
        self,
        landmarks_seq: List[Optional[np.ndarray]],
        fps: float = 8.0
    ) -> Dict[str, Any]:
        """
        Extracts biomechanical kinematic feature vector across sampled frames.
        """
        dt = 1.0 / max(fps, 1.0)
        lms_arr = self._impute_missing_landmarks(landmarks_seq) # (T, 478, 3)
        T = lms_arr.shape[0]
        
        if T == 0 or np.all(lms_arr == 0):
            # Fallback zero vector (6 regions * 4 derivative types * 7 stats = 168 features)
            feature_vec = np.zeros((168,), dtype=np.float32)
            return {
                "feature_vector": feature_vec,
                "biomechanics_scores": {"mouth_motion_anomaly": 0.0, "eye_motion_anomaly": 0.0}
            }

        # Scale normalization relative to inter-pupillary distance (left eye center to right eye center)
        left_eye_center = np.mean(lms_arr[:, LANDMARK_GROUPS["left_eye"], :2], axis=1) # (T, 2)
        right_eye_center = np.mean(lms_arr[:, LANDMARK_GROUPS["right_eye"], :2], axis=1) # (T, 2)
        ipd = np.mean(np.linalg.norm(left_eye_center - right_eye_center, axis=1))
        scale_factor = 1.0 / max(ipd, 1e-4)

        all_stats = []
        region_anomalies = {}

        for reg in BIOMECHANIC_REGIONS:
            indices = LANDMARK_GROUPS[reg]
            # Centroid trajectory of the region over time
            reg_traj = np.mean(lms_arr[:, indices, :], axis=1) * scale_factor # (T, 3)
            
            # Smooth trajectory
            reg_traj_smoothed = self._smooth_trajectory(reg_traj)
            
            # Kinematic derivatives
            # 1. Displacement relative to initial frame and consecutive frame
            step_disp = np.linalg.norm(np.diff(reg_traj_smoothed, axis=0, prepend=reg_traj_smoothed[:1]), axis=1)
            
            # 2. Velocity (1st derivative)
            vel = np.gradient(reg_traj_smoothed, dt, axis=0)
            vel_mag = np.linalg.norm(vel, axis=1)
            
            # 3. Acceleration (2nd derivative)
            acc = np.gradient(vel, dt, axis=0)
            acc_mag = np.linalg.norm(acc, axis=1)
            
            # 4. Jerk (3rd derivative)
            jerk = np.gradient(acc, dt, axis=0)
            jerk_mag = np.linalg.norm(jerk, axis=1)
            
            # Compute statistical metrics for each kinematic derivative
            disp_stats = self.compute_stats(step_disp, fps)
            vel_stats = self.compute_stats(vel_mag, fps)
            acc_stats = self.compute_stats(acc_mag, fps)
            jerk_stats = self.compute_stats(jerk_mag, fps)
            
            all_stats.extend(disp_stats + vel_stats + acc_stats + jerk_stats)
            
            # Compute anomaly proxy metric (RMS jerk & acceleration magnitude)
            anomaly_score = float(jerk_stats[4] * 0.5 + acc_stats[4] * 0.5)
            region_anomalies[f"{reg}_motion_anomaly"] = anomaly_score

        feature_vec = np.array(all_stats, dtype=np.float32)
        
        mouth_score = region_anomalies.get("mouth_motion_anomaly", 0.0)
        eye_score = (region_anomalies.get("left_eye_motion_anomaly", 0.0) + region_anomalies.get("right_eye_motion_anomaly", 0.0)) / 2.0
        
        return {
            "feature_vector": feature_vec,
            "biomechanics_scores": {
                "mouth_motion_anomaly": float(mouth_score),
                "eye_motion_anomaly": float(eye_score)
            }
        }
