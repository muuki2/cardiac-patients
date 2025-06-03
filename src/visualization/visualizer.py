"""Visualization utilities for model results and analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve


class ModelVisualizer:
    """Comprehensive visualization utilities for model analysis."""
    
    def __init__(self, config):
        self.config = config
        sns.set_style("whitegrid")
        
    def plot_training_comparison(self, results_dict: Dict[str, Dict[str, Dict[str, float]]]) -> None:
        """Create comprehensive training results comparison."""
        
        # Extract metrics for plotting
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        splits = ['train', 'val', 'test']
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Prepare data for plotting
            x = np.arange(len(models))
            width = 0.25
            
            for j, split in enumerate(splits):
                values = []
                for model in models:
                    if split in results_dict[model] and metric in results_dict[model][split]:
                        values.append(results_dict[model][split][metric])
                    else:
                        values.append(0)
                
                ax.bar(x + j * width, values, width, label=split.capitalize(), alpha=0.8)
            
            ax.set_xlabel('Models')
            ax.set_ylabel(metric.upper())
            ax.set_title(f'{metric.upper()} Comparison')
            ax.set_xticks(x + width)
            ax.set_xticklabels(models, rotation=45)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1)
            
        # Remove the last subplot if there are 6 subplots but only 5 metrics
        if len(metrics) < len(axes):
            fig.delaxes(axes[-1])
            
        plt.tight_layout()
        plt.show()
        
    def plot_confusion_matrices(self, models_dict: Dict, X_test: np.ndarray, 
                               y_test: np.ndarray) -> None:
        """Plot confusion matrices for all models."""
        
        n_models = len(models_dict)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 6 * n_rows))
        
        # Handle different subplot configurations
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, (model_name, model) in enumerate(models_dict.items()):
            ax = axes[i]
            
            # Get predictions
            y_pred = model.predict(X_test)
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=['No Disease', 'Disease'],
                       yticklabels=['No Disease', 'Disease'],
                       ax=ax)
            ax.set_title(f'{model_name} - Confusion Matrix')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Remove empty subplots
        if n_models < len(axes):
            for i in range(n_models, len(axes)):
                fig.delaxes(axes[i])
                
        plt.tight_layout()
        plt.show()
        
    def plot_roc_curves(self, models_dict: Dict, X_test: np.ndarray, 
                       y_test: np.ndarray) -> None:
        """Plot ROC curves for all models on the same plot."""
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            # Get prediction probabilities
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                else:
                    continue
                    
                # Calculate ROC curve
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                
                # Calculate AUC
                from sklearn.metrics import auc
                roc_auc = auc(fpr, tpr)
                
                # Plot ROC curve
                plt.plot(fpr, tpr, linewidth=2, 
                        label=f'{model_name} (AUC = {roc_auc:.3f})')
                        
            except Exception as e:
                print(f"Could not plot ROC curve for {model_name}: {e}")
                continue
        
        # Plot random classifier line
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_precision_recall_curves(self, models_dict: Dict, X_test: np.ndarray, 
                                    y_test: np.ndarray) -> None:
        """Plot Precision-Recall curves for all models."""
        
        plt.figure(figsize=(10, 8))
        
        for model_name, model in models_dict.items():
            try:
                # Get prediction probabilities
                if hasattr(model, 'predict_proba'):
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, 'decision_function'):
                    y_pred_proba = model.decision_function(X_test)
                else:
                    continue
                    
                # Calculate Precision-Recall curve
                precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
                
                # Calculate average precision
                from sklearn.metrics import average_precision_score
                avg_precision = average_precision_score(y_test, y_pred_proba)
                
                # Plot PR curve
                plt.plot(recall, precision, linewidth=2,
                        label=f'{model_name} (AP = {avg_precision:.3f})')
                        
            except Exception as e:
                print(f"Could not plot PR curve for {model_name}: {e}")
                continue
        
        # Plot baseline
        baseline = y_test.sum() / len(y_test)
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline (AP = {baseline:.3f})')
        
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
    def plot_feature_importance_comparison(self, models_dict: Dict, 
                                         feature_names: List[str]) -> None:
        """Plot feature importance comparison for models that support it."""
        
        importance_data = {}
        
        for model_name, model in models_dict.items():
            try:
                if hasattr(model, 'get_feature_importance'):
                    importance_data[model_name] = model.get_feature_importance()
                elif hasattr(model.model, 'feature_importances_'):
                    importance_data[model_name] = model.model.feature_importances_
                elif hasattr(model.model, 'best_estimator_'):
                    if hasattr(model.model.best_estimator_, 'feature_importances_'):
                        importance_data[model_name] = model.model.best_estimator_.feature_importances_
                    elif hasattr(model.model.best_estimator_, 'coef_'):
                        importance_data[model_name] = np.abs(model.model.best_estimator_.coef_[0])
            except:
                continue
                
        if not importance_data:
            print("No models with feature importance available.")
            return
            
        # Create subplot for each model
        n_models = len(importance_data)
        n_cols = 2
        n_rows = (n_models + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 6 * n_rows))
        
        # Fix axes handling for different configurations
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1 and n_cols > 1:
            axes = list(axes)
        elif n_rows > 1:
            axes = axes.flatten()
        else:
            axes = [axes]
        
        for i, (model_name, importance) in enumerate(importance_data.items()):
            ax = axes[i]
            
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot top 10 features
            top_n = min(10, len(feature_names))
            y_pos = np.arange(top_n)
            
            ax.barh(y_pos, importance[indices[:top_n]])
            ax.set_yticks(y_pos)
            ax.set_yticklabels([feature_names[i] for i in indices[:top_n]])
            ax.invert_yaxis()
            ax.set_xlabel('Importance')
            ax.set_title(f'{model_name} - Feature Importance (Top {top_n})')
            ax.grid(True, alpha=0.3)
        
        # Remove empty subplots
        if n_models < len(axes):
            for i in range(n_models, len(axes)):
                fig.delaxes(axes[i])
                
        plt.tight_layout()
        plt.show()
        
    def plot_model_performance_radar(self, results_dict: Dict[str, Dict[str, Dict[str, float]]], 
                                   split: str = 'test') -> None:
        """Create a radar chart comparing model performance metrics."""
        
        models = list(results_dict.keys())
        metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        
        # Prepare data
        data = []
        for model in models:
            model_scores = []
            for metric in metrics:
                if split in results_dict[model] and metric in results_dict[model][split]:
                    model_scores.append(results_dict[model][split][metric])
                else:
                    model_scores.append(0)
            data.append(model_scores)
        
        # Number of metrics
        N = len(metrics)
        
        # Compute angle for each metric
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Complete the circle
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Colors for different models
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, (model, scores) in enumerate(zip(models, data)):
            # Complete the circle
            scores += scores[:1]
            
            # Plot
            ax.plot(angles, scores, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, scores, alpha=0.25, color=colors[i])
        
        # Add metric labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([metric.upper() for metric in metrics])
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'])
        ax.grid(True)
        
        # Add legend and title
        plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        plt.title(f'Model Performance Comparison ({split.upper()} Set)', 
                 size=16, fontweight='bold', pad=20)
        
        plt.tight_layout()
        plt.show()
        
    def plot_loss_curves(self, neural_net_model) -> None:
        """Plot training and validation loss curves for neural network."""
        
        if not hasattr(neural_net_model, 'training_history'):
            print("No training history available for neural network.")
            return
            
        history = neural_net_model.training_history
        
        if not history:
            print("Training history is empty.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs = range(1, len(history['train_losses']) + 1)
        
        # Plot losses
        ax1.plot(epochs, history['train_losses'], 'b-', label='Training Loss', linewidth=2)
        if history.get('val_losses'):
            ax1.plot(epochs, history['val_losses'], 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Model Loss Over Time')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot accuracies
        ax2.plot(epochs, history['train_accuracies'], 'b-', label='Training Accuracy', linewidth=2)
        if history.get('val_accuracies'):
            ax2.plot(epochs, history['val_accuracies'], 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Model Accuracy Over Time')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    def create_comprehensive_report(self, results_dict: Dict, models_dict: Dict, 
                                  X_test: np.ndarray, y_test: np.ndarray,
                                  feature_names: List[str]) -> None:
        """Create a comprehensive visualization report."""
        
        print("=== COMPREHENSIVE MODEL EVALUATION REPORT ===\n")
        
        # 1. Performance comparison
        print("1. Model Performance Comparison")
        self.plot_training_comparison(results_dict)
        
        # 2. Confusion matrices
        print("\n2. Confusion Matrices")
        self.plot_confusion_matrices(models_dict, X_test, y_test)
        
        # 3. ROC curves
        print("\n3. ROC Curves Comparison")
        self.plot_roc_curves(models_dict, X_test, y_test)
        
        # 4. Precision-Recall curves
        print("\n4. Precision-Recall Curves")
        self.plot_precision_recall_curves(models_dict, X_test, y_test)
        
        # 5. Feature importance (if available)
        print("\n5. Feature Importance Analysis")
        self.plot_feature_importance_comparison(models_dict, feature_names)
        
        # 6. Radar chart
        print("\n6. Performance Radar Chart")
        self.plot_model_performance_radar(results_dict, split='test')
        
        # 7. Neural network training curves (if available)
        for model_name, model in models_dict.items():
            if 'Neural' in model_name or 'NN' in model_name:
                print(f"\n7. {model_name} Training Curves")
                self.plot_loss_curves(model)
                break
        
        print("\n=== REPORT COMPLETE ===")
        
    def save_all_plots(self, results_dict: Dict, models_dict: Dict, 
                      X_test: np.ndarray, y_test: np.ndarray,
                      feature_names: List[str], save_dir: str = "plots/") -> None:
        """Save all plots to files."""
        
        import os
        os.makedirs(save_dir, exist_ok=True)
        
        # Set matplotlib to use non-interactive backend for saving
        original_backend = plt.get_backend()
        plt.switch_backend('Agg')
        
        try:
            # Performance comparison
            self.plot_training_comparison(results_dict)
            plt.savefig(f"{save_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC curves
            self.plot_roc_curves(models_dict, X_test, y_test)
            plt.savefig(f"{save_dir}/roc_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Precision-Recall curves
            self.plot_precision_recall_curves(models_dict, X_test, y_test)
            plt.savefig(f"{save_dir}/precision_recall_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Radar chart
            self.plot_model_performance_radar(results_dict, split='test')
            plt.savefig(f"{save_dir}/performance_radar.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"All plots saved to {save_dir}")
            
        finally:
            # Restore original backend
            plt.switch_backend(original_backend)