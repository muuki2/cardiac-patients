"""
Configuration Management Module

This module provides centralized configuration management for the heart disease prediction
ML pipeline. It includes all hyperparameters, data settings, and model configurations 
required for reproducible machine learning experiments.

Classes:
    Config: Main configuration dataclass containing all settings for the ML pipeline

Key Features:
    - Centralized parameter management
    - Automatic random seed setting for reproducibility
    - Device detection (CPU/GPU) for neural networks
    - Cross-validation and model hyperparameter settings
    - Data preprocessing configurations

Usage:
    >>> from utils.config import Config
    >>> config = Config()
    >>> print(f"Using device: {config.device}")
    >>> print(f"Random seed: {config.random_seed}")
"""

import torch
from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class Config:
    """
    Configuration class for ML pipeline containing all hyperparameters and settings.
    
    This dataclass centralizes all configuration parameters used throughout the 
    heart disease prediction pipeline, including data processing, model training,
    and evaluation settings.
    
    Attributes:
        data_path (str): Path to the heart disease dataset CSV file
        target_column (str): Name of the target variable column
        test_size (float): Proportion of data reserved for testing (0.0-1.0)
        val_size (float): Proportion of data reserved for validation (0.0-1.0)
        random_state (int): Random seed for sklearn operations
        scale_features (bool): Whether to apply StandardScaler to features
        device (str): Computing device for PyTorch ('cuda' or 'cpu')
        hidden_size (int): Number of neurons in neural network hidden layers
        dropout_rate (float): Dropout probability for neural network regularization
        learning_rate (float): Learning rate for neural network optimization
        epochs (int): Maximum number of training epochs for neural networks
        batch_size (int): Batch size for neural network training
        early_stopping_patience (int): Epochs to wait before early stopping
        cv_folds (int): Number of cross-validation folds
        figsize (tuple): Default figure size for visualizations
        random_seed (int): Master random seed for reproducibility
    
    Methods:
        __post_init__(): Automatically sets random seeds for reproducibility
    
    Example:
        >>> config = Config()
        >>> config.epochs = 200  # Modify default settings
        >>> config.learning_rate = 0.0001
    """
    
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
        """
        Post-initialization method to set random seeds for reproducibility.
        
        This method is automatically called after the dataclass is initialized.
        It sets random seeds for all major libraries used in the pipeline:
        - PyTorch (both CPU and CUDA)
        - NumPy
        - Python's random module
        
        This ensures reproducible results across different runs of the pipeline.
        """
        import numpy as np
        import random
        
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.random_seed)