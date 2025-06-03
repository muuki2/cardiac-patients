"""Base model class for consistent interface."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config):
        self.config = config
        self.model = None
        self.is_fitted = False
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def get_model_name(self) -> str:
        """Get the model name."""
        return self.__class__.__name__.replace('Model', '')
    
    def get_training_history(self) -> Dict[str, Any]:
        """Get training history if available."""
        return self.training_history