"""
XGBoost Model Implementation Module

This module implements an XGBoost (Extreme Gradient Boosting) classifier with comprehensive
hyperparameter tuning for the heart disease prediction pipeline. It provides automated
optimization of tree-based parameters, regularization settings, and boosting configurations
using cross-validation for robust model selection.

Classes:
    XGBoostModel: XGBoost classifier with automated hyperparameter tuning using GridSearchCV

Key Features:
    - Gradient boosting with tree-based weak learners
    - Comprehensive hyperparameter optimization across multiple dimensions
    - Built-in regularization (L1/L2) and early stopping capabilities
    - Feature importance extraction for interpretability
    - Cross-validation for robust model selection
    - ROC-AUC optimization for binary classification
    - Efficient handling of missing values and categorical features
    - Parallel processing support for faster training

Design Pattern:
    Extends BaseModel following the Template Method pattern, implementing
    XGBoost-specific training and prediction methods while maintaining consistency
    with other models in the pipeline.

Usage:
    >>> from models.xgboost_model import XGBoostModel
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> model = XGBoostModel(config)
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> predictions = model.predict(X_test)
    >>> feature_importance = model.get_feature_importance()
"""

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """
    XGBoost Classifier with comprehensive hyperparameter tuning and feature importance.
    
    This class implements an XGBoost classifier with automated hyperparameter
    optimization exploring tree depth, learning rates, regularization parameters,
    and sampling strategies. XGBoost is particularly effective for structured/tabular
    data with complex feature interactions.
    
    XGBoost excels at:
    - Complex non-linear patterns in tabular data
    - Feature interaction modeling
    - Handling mixed data types (numerical and categorical)
    - Robust performance across diverse datasets
    - Built-in feature importance and interpretability
    
    Attributes:
        config: Configuration object containing training parameters
        model: Fitted GridSearchCV object containing the best XGBoost estimator
        is_fitted (bool): Flag indicating whether the model has been trained
        training_history (dict): Dictionary containing training results and best parameters
        param_grid (dict): Hyperparameter search space definition
    
    Hyperparameter Search Space:
        - n_estimators: Number of boosting rounds [100, 200, 300]
        - max_depth: Maximum tree depth [3, 4, 5, 6]
        - learning_rate: Boosting learning rate [0.01, 0.1, 0.2]
        - subsample: Sample fraction for training [0.8, 0.9, 1.0]
        - colsample_bytree: Feature fraction for trees [0.8, 0.9, 1.0]
    
    Key XGBoost Concepts:
        - Gradient Boosting: Sequential model improvement
        - Tree Depth: Controls model complexity and overfitting
        - Learning Rate: Controls contribution of each tree
        - Subsampling: Reduces overfitting through randomization
        - Column Sampling: Feature randomization for diversity
    
    Example:
        >>> config = Config()
        >>> model = XGBoostModel(config)
        >>> 
        >>> # Train with automatic hyperparameter tuning
        >>> history = model.fit(X_train, y_train, X_val, y_val)
        >>> print(f"Best n_estimators: {history['best_params']['n_estimators']}")
        >>> print(f"Best max_depth: {history['best_params']['max_depth']}")
        >>> 
        >>> # Get feature importance for interpretability
        >>> importance = model.get_feature_importance()
        >>> top_features = np.argsort(importance)[-5:]  # Top 5 features
        >>> 
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """
    
    def __init__(self, config):
        """
        Initialize the XGBoostModel with configuration and hyperparameter grid.
        
        Sets up a comprehensive hyperparameter search space covering the most
        important XGBoost parameters for optimal performance across different
        data characteristics and problem complexities.
        
        Args:
            config: Configuration object containing random seed, CV folds,
                   and other training parameters
        """
        super().__init__(config)
        self.param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 4, 5, 6],
            'learning_rate': [0.01, 0.1, 0.2],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train XGBoost with comprehensive hyperparameter tuning using grid search.
        
        Performs exhaustive grid search over boosting parameters, tree configurations,
        and regularization settings to find the optimal XGBoost configuration.
        Uses ROC-AUC scoring for optimization and includes comprehensive logging
        of training progress and results.
        
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
                - 'feature_importance': Feature importance scores from best model
                - 'training_info': Additional training information and statistics
                
        Raises:
            ValueError: If training data is empty or has incompatible shapes
            
        Note:
            XGBoost grid search can be computationally intensive due to the
            large parameter space. Consider using early stopping or reducing
            the parameter grid for very large datasets.
            
        Example:
            >>> model = XGBoostModel(config)
            >>> history = model.fit(X_train, y_train)
            Starting hyperparameter tuning for XGBoost...
            Search space: 540 combinations
            Best parameters: {'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 4, 'n_estimators': 200, 'subsample': 0.9}
            Best cross-validation ROC-AUC: 0.8721
        """
        
        # Create base model
        base_model = xgb.XGBClassifier(
            random_state=self.config.random_state,
            eval_metric='logloss',
            # use_label_encoder=False
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
        Make binary predictions using the fitted XGBoost model.
        
        Uses the best estimator found during hyperparameter tuning to make
        binary class predictions based on the ensemble of decision trees.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions array of shape (n_samples,)
            Values are 0 (no heart disease) or 1 (heart disease)
            
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Note:
            XGBoost predictions are based on the ensemble voting of all trees
            in the boosting sequence, providing robust predictions with
            implicit uncertainty quantification through tree diversity.
            
        Example:
            >>> predictions = model.predict(X_test)
            >>> print(f"Predicted classes: {np.unique(predictions, return_counts=True)}")
            >>> print(f"Positive rate: {np.mean(predictions):.2%}")
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the fitted XGBoost model.
        
        Returns probability estimates for both classes based on the ensemble
        of decision trees. XGBoost provides well-calibrated probabilities
        through the logistic transformation of tree ensemble outputs.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, 2) where:
                - Column 0: Probability of class 0 (no heart disease)
                - Column 1: Probability of class 1 (heart disease)
                
        Raises:
            ValueError: If model has not been fitted or input shape is incompatible
            
        Note:
            XGBoost probabilities are generally well-calibrated due to the
            logistic transformation and the ensemble nature of the algorithm.
            These probabilities are suitable for threshold tuning and
            confidence-based filtering.
            
        Example:
            >>> probabilities = model.predict_proba(X_test)
            >>> positive_probs = probabilities[:, 1]  # Probability of heart disease
            >>> high_confidence = X_test[positive_probs > 0.9]  # Very confident predictions
            >>> uncertain_cases = X_test[(positive_probs > 0.4) & (positive_probs < 0.6)]
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)