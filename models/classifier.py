import os
import joblib
import json
import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional, List
from utils.logging import setup_logger

logger = setup_logger("Classifier")

class BaseClassifier(ABC):
    """Abstract base class for modular classifiers."""
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray):
        pass

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Returns array of shape (N, 2) with [prob_real, prob_fake]."""
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns binary predictions (0 for REAL, 1 for FAKE)."""
        pass

    @abstractmethod
    def get_feature_importances(self) -> Optional[np.ndarray]:
        pass

    @abstractmethod
    def save(self, filepath: str):
        pass

    @abstractmethod
    def load(self, filepath: str):
        pass

class PyTorchAdapterWrapper(BaseClassifier):
    """Wrapper for LightweightAnatomicalAdapter PyTorch model."""
    def __init__(self, input_dim: int = 1824, params: Optional[Dict[str, Any]] = None):
        from models.adapter import LightweightAnatomicalAdapter
        params = params or {}
        hidden_dim = params.get("hidden_dim", 256)
        proj_dim = params.get("proj_dim", 128)
        dropout = params.get("dropout", 0.3)
        
        self.device = torch.device("cpu")
        self.model = LightweightAnatomicalAdapter(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            proj_dim=proj_dim,
            dropout=dropout
        ).to(self.device)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        from training.train_adapter import train_adapter_pipeline
        # Training handled by train_adapter_pipeline
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        self.model.eval()
        with torch.no_grad():
            tensor_x = torch.tensor(X, dtype=torch.float32).to(self.device)
            _, logits = self.model(tensor_x)
            prob_fake = torch.sigmoid(logits).cpu().numpy()
            
        if prob_fake.ndim == 0:
            prob_fake = np.array([prob_fake])
            
        prob_real = 1.0 - prob_fake
        probas = np.column_stack([prob_real, prob_fake])
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        probas = self.predict_proba(X)
        return (probas[:, 1] >= 0.5).astype(int)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        return None

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        torch.save(self.model.state_dict(), filepath)
        logger.info(f"Saved PyTorch Adapter weights to {filepath}")

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            logger.warning(f"File {filepath} not found for loading PyTorch Adapter.")
            return
            
        state_dict = torch.load(filepath, map_location=self.device)
        # Infer input dimension from first linear layer weights
        if "fc1.weight" in state_dict:
            in_dim = state_dict["fc1.weight"].shape[1]
            from models.adapter import LightweightAnatomicalAdapter
            self.model = LightweightAnatomicalAdapter(input_dim=in_dim).to(self.device)
            
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.is_fitted = True
        logger.info(f"Loaded PyTorch Adapter weights from {filepath}")

class XGBoostClassifierWrapper(BaseClassifier):
    """XGBoost Classifier implementation."""
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        import xgboost as xgb
        self.params = params or {
            "n_estimators": 150,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "random_state": 42
        }
        self.model = xgb.XGBClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info(f"Training XGBoost classifier on matrix shape X={X.shape}, y={y.shape}...")
        self.model.fit(X, y)
        self.is_fitted = True
        logger.info("XGBoost training complete.")

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        probas = self.model.predict_proba(X)
        return probas

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        if not self.is_fitted:
            return None
        return self.model.feature_importances_

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        if filepath.endswith(".json"):
            self.model.save_model(filepath)
        else:
            joblib.dump(self.model, filepath)
        logger.info(f"Saved XGBoost model to {filepath}")

    def load(self, filepath: str):
        import xgboost as xgb
        if filepath.endswith(".json"):
            self.model = xgb.XGBClassifier()
            self.model.load_model(filepath)
        else:
            self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded XGBoost model from {filepath}")

class SklearnClassifierWrapper(BaseClassifier):
    """Wrapper for scikit-learn classifiers (LogisticRegression, RandomForest, MLP)."""
    def __init__(self, classifier_type: str = "logistic_regression", params: Optional[Dict[str, Any]] = None):
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neural_network import MLPClassifier
        
        self.classifier_type = classifier_type
        params = params or {}
        
        if classifier_type == "logistic_regression":
            self.model = LogisticRegression(**params)
        elif classifier_type == "random_forest":
            self.model = RandomForestClassifier(**params)
        elif classifier_type == "mlp":
            self.model = MLPClassifier(**params)
        else:
            raise ValueError(f"Unsupported classifier type: {classifier_type}")
            
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray):
        logger.info(f"Training {self.classifier_type} on X={X.shape}, y={y.shape}...")
        self.model.fit(X, y)
        self.is_fitted = True

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return self.model.predict(X)

    def get_feature_importances(self) -> Optional[np.ndarray]:
        if hasattr(self.model, "feature_importances_"):
            return self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            return np.abs(self.model.coef_).ravel()
        return None

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        logger.info(f"Saved {self.classifier_type} model to {filepath}")

    def load(self, filepath: str):
        self.model = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Loaded {self.classifier_type} model from {filepath}")
