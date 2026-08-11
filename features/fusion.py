import os
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from typing import Dict, Any, Tuple, Optional, List
from utils.logging import setup_logger

logger = setup_logger("FeatureFuser")

class FeatureFuser:
    """
    Fuses Visual, Biomechanical, and Physiological feature vectors into a clean,
    normalized representation, managing missing value imputation, scaling,
    and ablation experiment feature selection (Exp A through Exp E).
    """
    def __init__(self, imputation_strategy: str = "median", with_scaling: bool = True):
        self.imputer = SimpleImputer(strategy=imputation_strategy, fill_value=0.0)
        self.scaler = StandardScaler() if with_scaling else None
        self.is_fitted = False

    def build_raw_feature_vector(
        self,
        visual_dict: Dict[str, Any],
        biomech_dict: Dict[str, Any],
        rppg_dict: Dict[str, Any],
        experiment_mode: str = "exp_E"
    ) -> np.ndarray:
        """
        Concatenates features based on experiment configuration:
        - exp_A: Whole-face / generic visual features only
        - exp_B: Facial parts visual features only (eyes + nose + mouth + chin)
        - exp_C: Facial parts + biomechanics
        - exp_D: Facial parts + rPPG
        - exp_E: Full model (Facial parts + biomechanics + rPPG)
        """
        vis_vec = visual_dict.get("feature_vector", np.array([], dtype=np.float32))
        bio_vec = biomech_dict.get("feature_vector", np.array([], dtype=np.float32))
        rppg_vec = rppg_dict.get("feature_vector", np.array([], dtype=np.float32))

        # Handle NaNs or Infs in raw features
        vis_vec = np.nan_to_num(vis_vec, nan=0.0, posinf=0.0, neginf=0.0)
        bio_vec = np.nan_to_num(bio_vec, nan=0.0, posinf=0.0, neginf=0.0)
        rppg_vec = np.nan_to_num(rppg_vec, nan=0.0, posinf=0.0, neginf=0.0)

        exp_mode = experiment_mode.lower()
        
        if exp_mode in ["exp_a", "visual_face_only"]:
            # Uses subset (first 1/5th) of visual feature vector representing whole face/eyes
            sub_len = max(1, len(vis_vec) // 5)
            fused = vis_vec[:sub_len]
        elif exp_mode in ["exp_b", "visual_parts_only"]:
            fused = vis_vec
        elif exp_mode in ["exp_c", "visual_parts_biomechanics"]:
            fused = np.concatenate([vis_vec, bio_vec], axis=0)
        elif exp_mode in ["exp_d", "visual_parts_rppg"]:
            fused = np.concatenate([vis_vec, rppg_vec], axis=0)
        elif exp_mode in ["exp_e", "full_model"]:
            fused = np.concatenate([vis_vec, bio_vec, rppg_vec], axis=0)
        else:
            raise ValueError(f"Unknown experiment mode: {experiment_mode}")

        return fused.astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fits imputer and scaler on dataset matrix X and transforms it."""
        X_imp = self.imputer.fit_transform(X)
        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X_imp)
        else:
            X_scaled = X_imp
        self.is_fitted = True
        return X_scaled

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms feature matrix X using fitted imputer and scaler."""
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        if not self.is_fitted:
            # Fallback if not pre-fitted
            X_imp = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            return X_imp
            
        X_imp = self.imputer.transform(X)
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X_imp)
        else:
            X_scaled = X_imp
        return X_scaled

    def save(self, filepath: str):
        """Saves fitted imputer and scaler."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({"imputer": self.imputer, "scaler": self.scaler, "is_fitted": self.is_fitted}, filepath)
        logger.info(f"Saved FeatureFuser scaler/imputer to {filepath}")

    def load(self, filepath: str):
        """Loads fitted imputer and scaler."""
        data = joblib.load(filepath)
        self.imputer = data["imputer"]
        self.scaler = data["scaler"]
        self.is_fitted = data["is_fitted"]
        logger.info(f"Loaded FeatureFuser from {filepath}")
