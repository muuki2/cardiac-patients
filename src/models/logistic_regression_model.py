"""
Logistic Regression Model Implementation Module

This module implements a Logistic Regression model with automated hyperparameter tuning
for the heart disease prediction pipeline. It extends the BaseModel interface and provides
comprehensive grid search optimization for finding optimal model parameters.

Classes:
    LogisticRegressionModel: Logistic regression with automated hyperparameter tuning
                           using cross-validation and grid search

Key Features:
    - Automated hyperparameter optimization using GridSearchCV
    - Cross-validation for robust model selection
    - Support for different regularization types (L1, L2)
    - Multiple solver options for different dataset characteristics
    - ROC-AUC optimization for binary classification
    - Comprehensive training history tracking

Design Pattern:
    Extends the BaseModel abstract class following the Template Method pattern,
    implementing specific training and prediction methods for logistic regression
    while maintaining consistent interface across all models.

Usage:
    >>> from models.logistic_regression_model import LogisticRegressionModel
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> model = LogisticRegressionModel(config)
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from .base_model import BaseModel


class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression model with comprehensive hyperparameter tuning.
    
    This class implements a logistic regression classifier with automated
    hyperparameter optimization using grid search and cross-validation.
    It explores different regularization strengths, penalty types, and
    solvers to find the optimal configuration for the dataset.
    
    The model is particularly well-suited for binary classification tasks
    with interpretable feature coefficients and probabilistic outputs.
    It handles both L1 and L2 regularization and supports multiple
    optimization algorithms.
    
    Attributes:
        config: Configuration object containing training parameters
        model: Fitted GridSearchCV object containing the best estimator
        is_fitted (bool): Flag indicating whether the model has been trained
        training_history (dict): Dictionary containing training results and best parameters
        param_grid (dict): Hyperparameter search space definition
    
    Hyperparameter Search Space:
        - C: Regularization strength [0.001, 0.01, 0.1, 1, 10, 100]
        - penalty: Regularization type ['l1', 'l2']
        - solver: Optimization algorithm ['liblinear', 'saga']
        - max_iter: Maximum iterations [1000, 2000]
    
    Example:
        >>> config = Config()
        >>> model = LogisticRegressionModel(config)
        >>> 
        >>> # Train with automatic hyperparameter tuning
        >>> history = model.fit(X_train, y_train, X_val, y_val)
        >>> print(f"Best parameters: {history['best_params']}")
        >>> print(f"Best CV score: {history['best_score']:.4f}")
        >>> 
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
        >>> 
        >>> # Get model name for reporting
        >>> print(f"Model: {model.get_model_name()}")
    """
    
    def __init__(self, config):
        """
        Initialize the LogisticRegressionModel with configuration and hyperparameter grid.
        
        Sets up the hyperparameter search space optimized for binary classification
        tasks with various regularization options and solver configurations.
        
        Args:
            config: Configuration object containing random seed, CV folds,
                   and other training parameters
        """
        super().__init__(config)
        self.param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train logistic regression with comprehensive hyperparameter tuning.
        
        Performs grid search with cross-validation to find optimal hyperparameters,
        optimizing for ROC-AUC score which is appropriate for binary classification
        tasks with potential class imbalance.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_val: Optional validation features (not used in grid search but kept for consistency)
            y_val: Optional validation targets (not used in grid search but kept for consistency)
            
        Returns:
            Dictionary containing:
                - 'best_params': Best hyperparameters found by grid search
                - 'best_score': Best cross-validation ROC-AUC score
                - 'cv_results': Complete cross-validation results
                
        Raises:
            ValueError: If training data is empty or has incompatible shapes
            
        Note:
            Validation data (X_val, y_val) is not used during training as grid search
            uses cross-validation on training data. This maintains consistency with
            other models that may use validation data.
            
        Example:
            >>> model = LogisticRegressionModel(config)
            >>> history = model.fit(X_train, y_train)
            Best parameters: {'C': 1, 'penalty': 'l2', 'solver': 'liblinear', 'max_iter': 1000}
            Best cross-validation score: 0.8542
        """
        
        # Create base model
        base_model = LogisticRegression(random_state=self.config.random_state)
        
        # Grid search with cross-validation
        self.model = GridSearchCV(
            base_model, 
            self.param_grid, 
            cv=self.config.cv_folds,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Store training information
        self.training_history = {
            'best_params': self.model.best_params_,
            'best_score': self.model.best_score_,
            'cv_results': self.model.cv_results_
        }
        
        print(f"Best parameters: {self.model.best_params_}")
        print(f"Best cross-validation score: {self.model.best_score_:.4f}")
        
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the fitted logistic regression model.
        
        Uses the best estimator found during hyperparameter tuning to make
        binary class predictions.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions array of shape (n_samples,)
            Values are 0 (no heart disease) or 1 (heart disease)
            
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"Predicted classes: {np.unique(predictions, return_counts=True)}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the fitted logistic regression model.
        
        Returns the probability estimates for both classes, providing insight
        into prediction confidence and enabling threshold tuning.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, 2) where:
                - Column 0: Probability of class 0 (no heart disease)
                - Column 1: Probability of class 1 (heart disease)
                
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Note:
            Probabilities are calibrated by default in logistic regression,
            making them suitable for direct interpretation and threshold tuning.
            
        Example:
            >>> probabilities = model.predict_proba(X_test)
            >>> positive_probs = probabilities[:, 1]  # Probability of heart disease
            >>> high_risk_patients = X_test[positive_probs > 0.8]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)