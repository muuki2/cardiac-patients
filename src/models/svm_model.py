"""SVM model implementation."""

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any
from .base_model import BaseModel


class SVMModel(BaseModel):
    """Support Vector Machine with hyperparameter tuning."""
    
    def __init__(self, config):
        super().__init__(config)
        self.param_grid = {
            'C': [0.1, 1, 10, 100],
            'kernel': ['rbf', 'linear', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1]
        }
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train SVM with grid search."""
        
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
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        return self.model.predict_proba(X)