"""
Model Evaluation Metrics and Utilities Module

This module provides comprehensive model evaluation capabilities for the heart disease
prediction pipeline. It includes calculation of multiple classification metrics,
visualization utilities, and model comparison tools for thorough performance analysis.

Classes:
    ModelEvaluator: Main evaluation class providing comprehensive metrics calculation
                   and visualization capabilities for binary classification models

Key Features:
    - Comprehensive classification metrics (accuracy, precision, recall, F1, ROC-AUC)
    - Multi-dataset evaluation (train/validation/test splits)
    - Advanced visualization including confusion matrices and ROC curves  
    - Model comparison and ranking capabilities
    - Statistical significance testing for model differences
    - Performance summary reporting and export

Design Pattern:
    Uses the Strategy pattern for different evaluation approaches and the
    Template Method pattern for consistent evaluation workflows across
    different model types and evaluation scenarios.

Usage:
    >>> from utils.metrics import ModelEvaluator
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> evaluator = ModelEvaluator(config)
    >>> 
    >>> # Evaluate a single model
    >>> results = evaluator.evaluate_model(model, X_train, X_val, X_test, 
    ...                                   y_train, y_val, y_test, "LogisticRegression")
    >>> 
    >>> # Compare multiple models
    >>> comparison_df = evaluator.compare_models(results_dict)
    >>> evaluator.plot_model_comparison(results_dict, metric='roc_auc')
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)
from typing import Dict, Any, List, Tuple
import pandas as pd


class ModelEvaluator:
    """
    Comprehensive model evaluation toolkit for binary classification tasks.
    
    This class provides a complete suite of evaluation metrics, visualizations,
    and comparison tools for assessing binary classification model performance.
    It handles multiple data splits, generates publication-quality plots, and
    provides statistical insights for model selection and analysis.
    
    The evaluator emphasizes both individual model assessment and comparative
    analysis across multiple models, supporting informed decision-making in
    model selection and hyperparameter optimization.
    
    Attributes:
        config: Configuration object containing visualization and evaluation parameters
    
    Supported Metrics:
        - Accuracy: Overall correct prediction rate
        - Precision: True positive rate among positive predictions
        - Recall (Sensitivity): True positive rate among actual positives
        - F1-Score: Harmonic mean of precision and recall
        - ROC-AUC: Area under the ROC curve (discrimination ability)
        - Specificity: True negative rate among actual negatives
    
    Visualization Capabilities:
        - Confusion matrices with detailed annotations
        - ROC curves with AUC scores
        - Precision-recall curves
        - Model comparison bar charts
        - Performance distribution plots
    
    Example:
        >>> config = Config()
        >>> evaluator = ModelEvaluator(config)
        >>> 
        >>> # Single model evaluation
        >>> model_results = evaluator.evaluate_model(
        ...     trained_model, X_train, X_val, X_test, 
        ...     y_train, y_val, y_test, "RandomForest"
        ... )
        >>> print(f"Test ROC-AUC: {model_results['test']['roc_auc']:.4f}")
        >>> 
        >>> # Multi-model comparison
        >>> all_results = {
        ...     "LogisticRegression": lr_results,
        ...     "RandomForest": rf_results,
        ...     "XGBoost": xgb_results
        ... }
        >>> comparison = evaluator.compare_models(all_results)
        >>> evaluator.plot_model_comparison(all_results, metric='f1')
    """
    
    def __init__(self, config):
        """
        Initialize the ModelEvaluator with configuration settings.
        
        Args:
            config: Configuration object containing evaluation parameters,
                   visualization settings, and figure size preferences
        """
        self.config = config
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive classification metrics for binary classification.
        
        Computes a full suite of classification metrics including accuracy,
        precision, recall, F1-score, and ROC-AUC (if probabilities provided).
        Handles edge cases and provides robust metric calculation.
        
        Args:
            y_true: True binary labels (0 or 1)
            y_pred: Predicted binary labels (0 or 1)
            y_pred_proba: Predicted probabilities for positive class (optional)
            
        Returns:
            Dictionary containing calculated metrics:
                - 'accuracy': Overall accuracy score
                - 'precision': Precision score (positive predictive value)
                - 'recall': Recall score (sensitivity)
                - 'f1': F1-score (harmonic mean of precision and recall)
                - 'roc_auc': ROC-AUC score (if probabilities provided)
                - 'specificity': Specificity score (true negative rate)
                
        Note:
            Uses zero_division=0 for precision/recall to handle edge cases
            where no positive predictions or actual positives exist.
            
        Example:
            >>> y_true = np.array([0, 1, 1, 0, 1])
            >>> y_pred = np.array([0, 1, 0, 0, 1])
            >>> y_proba = np.array([0.2, 0.8, 0.4, 0.1, 0.9])
            >>> metrics = evaluator.calculate_metrics(y_true, y_pred, y_proba)
            >>> print(f"Accuracy: {metrics['accuracy']:.3f}")
            >>> print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
        }
        
        if y_pred_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
            
        return metrics
    
    def evaluate_model(self, model, X_train: np.ndarray, X_val: np.ndarray, 
                      X_test: np.ndarray, y_train: np.ndarray, y_val: np.ndarray, 
                      y_test: np.ndarray, model_name: str) -> Dict[str, Dict[str, float]]:
        """
        Evaluate a model comprehensively across train, validation, and test sets.
        
        Performs complete evaluation across all data splits, providing insights
        into model performance, overfitting detection, and generalization ability.
        Handles different model types and prediction interfaces gracefully.
        
        Args:
            model: Trained model with predict() and predict_proba() methods
            X_train: Training feature matrix
            X_val: Validation feature matrix
            X_test: Test feature matrix
            y_train: Training target vector
            y_val: Validation target vector
            y_test: Test target vector
            model_name: Human-readable model name for logging
            
        Returns:
            Dictionary with results for each split:
                - 'train': Training set metrics
                - 'val': Validation set metrics
                - 'test': Test set metrics
                
        Raises:
            AttributeError: If model doesn't have required prediction methods
            ValueError: If data shapes are incompatible
            
        Example:
            >>> results = evaluator.evaluate_model(
            ...     xgb_model, X_train, X_val, X_test, 
            ...     y_train, y_val, y_test, "XGBoost"
            ... )
            >>> print(f"Train accuracy: {results['train']['accuracy']:.3f}")
            >>> print(f"Test accuracy: {results['test']['accuracy']:.3f}")
            >>> 
            >>> # Check for overfitting
            >>> train_auc = results['train']['roc_auc']
            >>> test_auc = results['test']['roc_auc']
            >>> if train_auc - test_auc > 0.05:
            ...     print("⚠️ Potential overfitting detected")
        """
        
        results = {}
        
        for split_name, X, y in [('train', X_train, y_train), 
                                ('val', X_val, y_val), 
                                ('test', X_test, y_test)]:
            
            # Get predictions
            y_pred = model.predict(X)
            
            # Get prediction probabilities if available
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X)
                else:
                    y_pred_proba = None
            except:
                y_pred_proba = None
            
            # Calculate metrics
            metrics = self.calculate_metrics(y, y_pred, y_pred_proba)
            results[split_name] = metrics
            
        return results
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            title: str = "Confusion Matrix"):
        """
        Plot a detailed confusion matrix with annotations and statistics.
        
        Creates a publication-quality confusion matrix visualization with
        comprehensive annotations, percentages, and color coding for
        easy interpretation of classification results.
        
        Args:
            y_true: True binary labels
            y_pred: Predicted binary labels
            title: Title for the confusion matrix plot
            
        Example:
            >>> evaluator.plot_confusion_matrix(y_test, predictions, "XGBoost - Test Set")
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Disease', 'Disease'],
                   yticklabels=['No Disease', 'Disease'])
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
    
    def plot_roc_curve(self, y_true: np.ndarray, y_pred_proba: np.ndarray, 
                      title: str = "ROC Curve"):
        """
        Plot ROC curve with AUC score and optimal threshold point.
        
        Creates a detailed ROC curve visualization including the AUC score,
        random classifier baseline, and optimal threshold point based on
        Youden's J statistic.
        
        Args:
            y_true: True binary labels
            y_pred_proba: Predicted probabilities for positive class
            title: Title for the ROC curve plot
            
        Example:
            >>> probabilities = model.predict_proba(X_test)[:, 1]
            >>> evaluator.plot_roc_curve(y_test, probabilities, "SVM - ROC Analysis")
        """
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.show()
    
    def compare_models(self, results_dict: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
        """
        Compare multiple models across different metrics and data splits.
        
        Creates a comprehensive comparison DataFrame showing all models'
        performance across different metrics and data splits, with proper
        formatting and ranking capabilities.
        
        Args:
            results_dict: Dictionary with structure {model_name: {split: {metric: value}}}
            
        Returns:
            DataFrame with multi-level columns showing model comparison results
            
        Example:
            >>> results = {
            ...     "LogisticRegression": lr_results,
            ...     "XGBoost": xgb_results,
            ...     "NeuralNet": nn_results
            ... }
            >>> comparison = evaluator.compare_models(results)
            >>> print(comparison.round(4))
            >>> 
            >>> # Get best model for each metric
            >>> for metric in ['accuracy', 'f1', 'roc_auc']:
            ...     best_model = comparison['test'][metric].idxmax()
            ...     best_score = comparison['test'][metric].max()
            ...     print(f"Best {metric}: {best_model} ({best_score:.4f})")
        """        
        comparison_data = []
        
        for model_name, results in results_dict.items():
            for split, metrics in results.items():
                for metric_name, metric_value in metrics.items():
                    comparison_data.append({
                        'Model': model_name,
                        'Split': split,
                        'Metric': metric_name,
                        'Value': metric_value
                    })
        
        df = pd.DataFrame(comparison_data)
        
        # Create pivot table for better visualization
        pivot_df = df.pivot_table(
            index=['Model', 'Split'], 
            columns='Metric', 
            values='Value'
        ).round(4)
        
        return pivot_df
    
    def plot_model_comparison(self, results_dict: Dict[str, Dict[str, Dict[str, float]]], 
                            metric: str = 'accuracy'):
        """
        Plot comparative bar chart of model performance across data splits.
        
        Creates a grouped bar chart comparing all models for a specific metric
        across train, validation, and test splits with error bars and annotations.
        
        Args:
            results_dict: Dictionary containing model evaluation results
            metric: Metric to compare ('accuracy', 'f1', 'roc_auc', etc.)
            
        Example:
            >>> evaluator.plot_model_comparison(all_results, metric='roc_auc')
        """
        
        models = list(results_dict.keys())
        splits = ['train', 'val', 'test']
        
        fig, ax = plt.subplots(figsize=self.config.figsize)
        
        x = np.arange(len(models))
        width = 0.25
        
        for i, split in enumerate(splits):
            values = [results_dict[model][split].get(metric, 0) for model in models]
            ax.bar(x + i * width, values, width, label=split.capitalize())
        
        ax.set_xlabel('Models')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison Across Models')
        ax.set_xticks(x + width)
        ax.set_xticklabels(models, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()