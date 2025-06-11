"""
Base Model Interface Module

This module defines the abstract base class that provides a consistent interface
for all machine learning models in the heart disease prediction pipeline.

Classes:
    BaseModel: Abstract base class defining the standard interface for all models

Key Features:
    - Consistent interface across different model types
    - Abstract methods ensuring implementation of core functionality
    - Training history tracking
    - Standardized model naming convention

Design Pattern:
    Uses the Template Method pattern to define the skeleton of model operations
    while allowing subclasses to implement specific algorithms.
"""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Tuple


class BaseModel(ABC):
    """
    Abstract base class for all machine learning models in the pipeline.
    
    This class defines the standard interface that all models must implement,
    ensuring consistency across different model types (traditional ML and deep learning).
    
    The class follows the Template Method design pattern, providing a common
    structure while allowing specific implementations in subclasses.
    
    Attributes:
        config: Configuration object containing hyperparameters and settings
        model: The actual model instance (e.g., sklearn model, PyTorch model)
        is_fitted (bool): Flag indicating whether the model has been trained
        training_history (dict): Dictionary storing training metrics and information
    
    Abstract Methods:
        fit: Train the model on provided data
        predict: Make binary predictions
        predict_proba: Predict class probabilities
    
    Concrete Methods:
        get_model_name: Extract and return the model name
        get_training_history: Return the training history dictionary
    
    Example:
        >>> class MyModel(BaseModel):
        ...     def fit(self, X_train, y_train, X_val=None, y_val=None):
        ...         # Implementation here
        ...         pass
        ...     def predict(self, X):
        ...         # Implementation here
        ...         pass
        ...     def predict_proba(self, X):
        ...         # Implementation here
        ...         pass
    """
    
    def __init__(self, config):
        """
        Initialize the base model with configuration.
        
        Args:
            config: Configuration object containing hyperparameters and settings
        """
        self.config = config
        self.model = None
        self.is_fitted = False
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the model on the provided training data.
        
        This method must be implemented by all subclasses to define how the
        specific model type should be trained.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_val: Optional validation feature matrix
            y_val: Optional validation target vector
            
        Returns:
            Dictionary containing training history and metrics
            
        Raises:
            NotImplementedError: If not implemented by subclass
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions on the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions array of shape (n_samples,)
            Values should be 0 (no heart disease) or 1 (heart disease)
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If model has not been fitted
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities for the provided data.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, 2)
            First column: probability of class 0 (no heart disease)
            Second column: probability of class 1 (heart disease)
            
        Raises:
            NotImplementedError: If not implemented by subclass
            ValueError: If model has not been fitted
        """
        pass
    
    def get_model_name(self) -> str:
        """
        Extract and return a clean model name from the class name.
        
        Removes the 'Model' suffix from class names for cleaner display
        in visualizations and reports.
        
        Returns:
            Clean model name string (e.g., 'LogisticRegression' from 'LogisticRegressionModel')
            
        Example:
            >>> model = LogisticRegressionModel(config)
            >>> print(model.get_model_name())  # Output: 'LogisticRegression'
        """
        return self.__class__.__name__.replace('Model', '')
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Return the training history dictionary.
        
        The training history contains model-specific information such as:
        - Best hyperparameters (for grid search models)
        - Cross-validation scores
        - Training/validation losses (for neural networks)
        - Training/validation accuracies (for neural networks)
        
        Returns:
            Dictionary containing training history and metrics
            
        Example:
            >>> model.fit(X_train, y_train)
            >>> history = model.get_training_history()
            >>> print(history.keys())
        """
        return self.training_history