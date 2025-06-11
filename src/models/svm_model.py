"""
Support Vector Machine Model Implementation Module

This module implements a Support Vector Machine (SVM) classifier with comprehensive
hyperparameter tuning for the heart disease prediction pipeline. It provides automated
optimization of kernel parameters, regularization strength, and other SVM-specific
hyperparameters using cross-validation.

Classes:
    SVMModel: SVM classifier with automated hyperparameter tuning using GridSearchCV

Key Features:
    - Multiple kernel support (RBF, Linear, Polynomial)
    - Automated hyperparameter optimization with grid search
    - Probabilistic predictions for classification confidence
    - Cross-validation for robust model selection
    - ROC-AUC optimization for imbalanced datasets
    - Comprehensive parameter space exploration
    - Automatic gamma scaling for different data distributions

Design Pattern:
    Extends BaseModel following the Template Method pattern, implementing
    SVM-specific training and prediction methods while maintaining consistency
    with other models in the pipeline.

Usage:
    >>> from models.svm_model import SVMModel
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> model = SVMModel(config)
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    >>> probabilities = model.predict_proba(X_test)
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from .base_model import BaseModel


class SVMModel(BaseModel):
    """
    Support Vector Machine model with comprehensive hyperparameter tuning.
    
    This class implements an SVM classifier with automated hyperparameter
    optimization exploring different kernel functions, regularization strengths,
    and kernel-specific parameters. It is particularly effective for complex
    non-linear classification problems.
    
    SVMs are powerful for binary classification tasks, especially with:
    - High-dimensional data
    - Non-linear decision boundaries (with appropriate kernels)
    - Robust performance with limited training data
    - Good generalization capabilities
    
    Attributes:
        config: Configuration object containing training parameters
        model: Fitted GridSearchCV object containing the best SVM estimator
        is_fitted (bool): Flag indicating whether the model has been trained
        training_history (dict): Dictionary containing training results and best parameters
        param_grid (dict): Hyperparameter search space definition
    
    Hyperparameter Search Space:
        - C: Regularization parameter [0.1, 1, 10, 100]
        - kernel: Kernel function ['rbf', 'linear', 'poly']
        - gamma: Kernel coefficient ['scale', 'auto', 0.001, 0.01, 0.1, 1]
    
    Kernel Functions:
        - RBF (Radial Basis Function): Good for non-linear patterns
        - Linear: Efficient for linearly separable data
        - Polynomial: Captures polynomial relationships
    
    Example:
        >>> config = Config()
        >>> model = SVMModel(config)
        >>> 
        >>> # Train with automatic hyperparameter tuning
        >>> history = model.fit(X_train, y_train, X_val, y_val)
        >>> print(f"Best kernel: {history['best_params']['kernel']}")
        >>> print(f"Best C: {history['best_params']['C']}")
        >>> 
        >>> # Make predictions with probability estimates
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
        >>> confidence = np.max(probabilities, axis=1)  # Prediction confidence
    """
    
    def __init__(self, config):
        """
        Initialize the SVMModel with configuration and hyperparameter grid.
        
        Sets up a comprehensive hyperparameter search space covering different
        kernel functions and their associated parameters for optimal performance
        across various data patterns.
        
        Args:
            config: Configuration object containing random seed, CV folds,
                   and other training parameters
        """
        super().__init__(config)
        self.param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train SVM with comprehensive hyperparameter tuning using grid search.
        
        Performs exhaustive grid search over kernel functions, regularization
        parameters, and kernel coefficients to find the optimal SVM configuration.
        Uses ROC-AUC scoring for optimization, which is appropriate for binary
        classification with potential class imbalance.
        
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
                - 'support_vectors_info': Information about support vectors
                
        Raises:
            ValueError: If training data is empty or has incompatible shapes
            
        Note:
            Grid search explores all parameter combinations, which can be
            computationally intensive. Consider reducing the parameter grid
            for very large datasets.
            
        Example:
            >>> model = SVMModel(config)
            >>> history = model.fit(X_train, y_train)
            Starting hyperparameter tuning for SVM...
            Search space: 192 combinations
            Best parameters: {'C': 10, 'gamma': 0.1, 'kernel': 'rbf'}
            Best cross-validation ROC-AUC: 0.8654
        """
        
        # Create base model
        base_model = SVC(
            random_state=self.config.random_state,
            probability=True  # Enable probability predictions
        )
        
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
        Make binary predictions using the fitted SVM model.
        
        Uses the best estimator found during hyperparameter tuning to make
        binary class predictions based on the learned decision boundary.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions array of shape (n_samples,)
            Values are 0 (no heart disease) or 1 (heart disease)
            
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Note:
            SVM predictions are based on the sign of the decision function.
            Points on one side of the hyperplane are classified as one class,
            points on the other side as the other class.
            
        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"Predicted classes: {np.unique(predictions, return_counts=True)}")
            >>> print(f"Prediction distribution: {np.bincount(predictions)}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the fitted SVM model.
        
        Returns probability estimates for both classes using Platt scaling,
        which calibrates the SVM decision function to produce probability-like
        outputs suitable for confidence estimation and threshold tuning.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, 2) where:
                - Column 0: Probability of class 0 (no heart disease)
                - Column 1: Probability of class 1 (heart disease)
                
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Note:
            SVM probability estimates are computed using Platt scaling by default,
            which fits a sigmoid function to the SVM decision function outputs.
            These probabilities may be less well-calibrated than those from
            inherently probabilistic models like logistic regression.
            
        Example:
            >>> probabilities = model.predict_proba(X_test)
            >>> positive_probs = probabilities[:, 1]  # Probability of heart disease
            >>> confident_predictions = X_test[positive_probs > 0.9]  # High confidence cases
            >>> uncertain_cases = X_test[(positive_probs > 0.4) & (positive_probs < 0.6)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)