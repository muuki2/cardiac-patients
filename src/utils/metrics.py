"""Model evaluation metrics and utilities."""

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
    """Comprehensive model evaluation with multiple metrics."""
    
    def __init__(self, config):
        self.config = config
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                         y_pred_proba: np.ndarray = None) -> Dict[str, float]:
        """Calculate comprehensive classification metrics."""
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
        """Evaluate a model on train, validation, and test sets."""
        
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
        """Plot confusion matrix."""
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
        """Plot ROC curve."""
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
        """Compare multiple models across different metrics and splits."""
        
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
        """Plot model comparison for a specific metric across splits."""
        
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