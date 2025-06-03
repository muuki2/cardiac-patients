"""MLflow utilities for experiment management and comparison."""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np


class MLflowExperimentManager:
    """Utilities for managing MLflow experiments and comparing results."""
    
    def __init__(self, experiment_name: str = "heart_disease_neural_net"):
        self.experiment_name = experiment_name
        
    def get_experiment_runs(self, max_results: int = 100) -> pd.DataFrame:
        """Get all runs from the experiment."""
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                print(f"Experiment '{self.experiment_name}' not found")
                return pd.DataFrame()
            
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                max_results=max_results,
                order_by=["metrics.val_roc_auc DESC"]
            )
            
            return runs
            
        except Exception as e:
            print(f"Error retrieving runs: {e}")
            return pd.DataFrame()
    
    def get_best_runs(self, metric: str = "val_roc_auc", top_k: int = 5) -> pd.DataFrame:
        """Get top-k best runs based on a metric."""
        runs = self.get_experiment_runs()
        
        if runs.empty:
            return runs
        
        metric_col = f"metrics.{metric}"
        if metric_col in runs.columns:
            best_runs = runs.nlargest(top_k, metric_col)
            return best_runs
        else:
            print(f"Metric '{metric}' not found in runs")
            return pd.DataFrame()
    
    def compare_hyperparameters(self, top_k: int = 10) -> None:
        """Create visualizations comparing hyperparameters."""
        runs = self.get_best_runs(top_k=top_k)
        
        if runs.empty:
            print("No runs found for comparison")
            return
        
        # Extract hyperparameter columns
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        metric_cols = [col for col in runs.columns if col.startswith('metrics.val_')]
        
        if not param_cols:
            print("No hyperparameters found in runs")
            return
        
        # Create subplots for each hyperparameter
        n_params = len(param_cols)
        fig, axes = plt.subplots(2, (n_params + 1) // 2, figsize=(15, 10))
        axes = axes.flatten() if n_params > 1 else [axes]
        
        for i, param_col in enumerate(param_cols):
            if i >= len(axes):
                break
                
            param_name = param_col.replace('params.', '')
            
            # Handle different parameter types
            param_values = runs[param_col].dropna()
            
            if param_values.dtype == 'object':
                # Categorical parameters
                value_counts = param_values.value_counts()
                axes[i].bar(range(len(value_counts)), value_counts.values)
                axes[i].set_xticks(range(len(value_counts)))
                axes[i].set_xticklabels(value_counts.index, rotation=45)
            else:
                # Numerical parameters
                axes[i].hist(param_values.astype(float), bins=min(10, len(param_values)), alpha=0.7)
            
            axes[i].set_title(f'Distribution of {param_name}')
            axes[i].set_xlabel(param_name)
            axes[i].set_ylabel('Count')
        
        # Hide unused subplots
        for i in range(len(param_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.show()
    
    def plot_hyperparameter_importance(self, metric: str = "val_roc_auc") -> None:
        """Plot hyperparameter importance based on correlation with target metric."""
        runs = self.get_experiment_runs()
        
        if runs.empty:
            print("No runs found")
            return
        
        metric_col = f"metrics.{metric}"
        if metric_col not in runs.columns:
            print(f"Metric '{metric}' not found")
            return
        
        # Get numeric hyperparameters
        param_cols = [col for col in runs.columns if col.startswith('params.')]
        
        correlations = {}
        for param_col in param_cols:
            param_name = param_col.replace('params.', '')
            
            # Try to convert to numeric
            try:
                param_values = pd.to_numeric(runs[param_col], errors='coerce')
                if not param_values.isna().all():
                    corr = param_values.corr(runs[metric_col])
                    if not np.isnan(corr):
                        correlations[param_name] = abs(corr)
            except:
                continue
        
        if not correlations:
            print("No numeric hyperparameters found for correlation analysis")
            return
        
        # Plot correlations
        sorted_correlations = dict(sorted(correlations.items(), key=lambda x: x[1], reverse=True))
        
        plt.figure(figsize=(10, 6))
        plt.barh(list(sorted_correlations.keys()), list(sorted_correlations.values()))
        plt.xlabel(f'Absolute Correlation with {metric}')
        plt.title('Hyperparameter Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_metric_trends(self, metrics: List[str] = None) -> None:
        """Plot trends of metrics across runs."""
        if metrics is None:
            metrics = ["val_accuracy", "val_precision", "val_recall", "val_f1", "val_roc_auc"]
        
        runs = self.get_experiment_runs()
        
        if runs.empty:
            print("No runs found")
            return
        
        # Sort runs by start time
        runs = runs.sort_values('start_time')
        
        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            metric_col = f"metrics.{metric}"
            if metric_col in runs.columns:
                metric_values = runs[metric_col].dropna()
                axes[i].plot(range(len(metric_values)), metric_values, 'o-', alpha=0.7)
                axes[i].set_title(f'{metric} Trend Across Runs')
                axes[i].set_xlabel('Run Number')
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
            else:
                axes[i].text(0.5, 0.5, f"Metric '{metric}' not found", 
                           ha='center', va='center', transform=axes[i].transAxes)
        
        plt.tight_layout()
        plt.show()
    
    def generate_experiment_report(self) -> str:
        """Generate a comprehensive experiment report."""
        runs = self.get_experiment_runs()
        
        if runs.empty:
            return "No runs found in experiment"
        
        # Basic statistics
        total_runs = len(runs)
        successful_runs = len(runs.dropna(subset=['metrics.val_roc_auc']))
        
        # Best performance
        best_run = runs.nlargest(1, 'metrics.val_roc_auc').iloc[0] if not runs.empty else None
        
        report = f"""
        MLflow Experiment Report: {self.experiment_name}
        ================================================
        
        Summary:
        - Total runs: {total_runs}
        - Successful runs: {successful_runs}
        - Success rate: {successful_runs/total_runs*100:.1f}%
        
        Best Performance:
        """
        
        if best_run is not None:
            report += f"""
        - Run ID: {best_run['run_id']}
        - Validation ROC-AUC: {best_run['metrics.val_roc_auc']:.4f}
        - Validation Accuracy: {best_run.get('metrics.val_accuracy', 'N/A')}
        - Best Hyperparameters:
        """
            
            # Add best hyperparameters
            param_cols = [col for col in runs.columns if col.startswith('params.')]
            for param_col in param_cols:
                param_name = param_col.replace('params.', '')
                param_value = best_run[param_col]
                report += f"    - {param_name}: {param_value}\n"
        
        return report
    
    def cleanup_failed_runs(self) -> None:
        """Delete runs that failed or have no metrics."""
        runs = self.get_experiment_runs()
        
        if runs.empty:
            print("No runs found")
            return
        
        # Find runs with no validation metrics
        failed_runs = runs[runs['metrics.val_roc_auc'].isna()]
        
        print(f"Found {len(failed_runs)} failed runs")
        
        for _, run in failed_runs.iterrows():
            try:
                mlflow.delete_run(run['run_id'])
                print(f"Deleted run {run['run_id']}")
            except Exception as e:
                print(f"Error deleting run {run['run_id']}: {e}")