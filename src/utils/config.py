"""Configuration settings for the machine learning pipeline."""

import torch
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """Configuration class for ML pipeline."""
    
    # Data settings
    data_path: str = "data/heart.csv"
    target_column: str = "HeartDisease"
    test_size: float = 0.2
    val_size: float = 0.2
    random_state: int = 42
    
    # Feature engineering
    scale_features: bool = True
    
    # Model settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Neural Network settings
    hidden_size: int = 64
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    early_stopping_patience: int = 10
    
    # Cross-validation
    cv_folds: int = 5
    
    # Visualization
    figsize: tuple = (12, 8)
    random_seed: int = 42
    
    def __post_init__(self):
        """Set random seeds for reproducibility."""
        import numpy as np
        import random
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)