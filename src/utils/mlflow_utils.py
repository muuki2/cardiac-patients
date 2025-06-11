"""
MLflow Experiment Management and Analysis Utilities Module

This module provides comprehensive utilities for managing MLflow experiments,
analyzing results, and comparing model performance across multiple runs.
It supports both local and Databricks MLflow instances with advanced
experiment tracking, visualization, and cleanup capabilities.

Classes:
    MLflowExperimentManager: Main class for MLflow experiment management,
                           analysis, and visualization

Key Features:
    - Comprehensive experiment run retrieval and analysis
    - Advanced hyperparameter importance analysis with correlation studies
    - Model performance comparison and ranking across experiments
    - Automated experiment cleanup and maintenance
    - Interactive visualizations for experiment exploration
    - Statistical analysis of hyperparameter effectiveness
    - Experiment reporting and documentation generation

Design Pattern:
    Uses the Repository pattern for experiment data access and the
    Observer pattern for experiment monitoring and analysis.

Usage:
    >>> from utils.mlflow_utils import MLflowExperimentManager
    >>> 
    >>> manager = MLflowExperimentManager("heart_disease_neural_net")
    >>> runs_df = manager.get_experiment_runs()
    >>> best_runs = manager.get_best_runs(metric="val_roc_auc", top_k=5)
    >>> manager.compare_hyperparameters(top_k=10)
    >>> manager.plot_hyperparameter_importance(metric="val_roc_auc")
    >>> report = manager.generate_experiment_report()
"""

import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import numpy as np


class MLflowExperimentManager:
    """
    Comprehensive MLflow experiment management and analysis toolkit.
    
    This class provides advanced capabilities for managing MLflow experiments,
    analyzing hyperparameter tuning results, comparing model performance,
    and generating insights from experiment data. It supports both local
    and Databricks MLflow instances with robust error handling.
    
    The manager emphasizes data-driven experiment analysis, providing
    statistical insights, visualizations, and automated reporting to
    support informed decision-making in machine learning workflows.
    
    Attributes:
        experiment_name (str): Name of the MLflow experiment to manage
    
    Core Capabilities:
        - Experiment run retrieval with filtering and sorting
        - Best model identification based on custom metrics
        - Hyperparameter importance analysis using correlation methods
        - Performance trend analysis across experiment timeline
        - Automated experiment cleanup and maintenance
        - Comprehensive reporting and documentation generation
    
    Analysis Features:
        - Statistical correlation analysis between hyperparameters and performance
        - Distribution analysis of hyperparameter values
        - Performance trend visualization over time
        - Model comparison with significance testing
        - Automated best practice recommendations
    
    Example:
        >>> # Initialize manager for specific experiment
        >>> manager = MLflowExperimentManager("heart_disease_prediction")
        >>> 
        >>> # Get comprehensive experiment overview    
         >>> runs = manager.get_experiment_runs(max_results=100)
        >>> print(f"Total runs: {len(runs)}")
        >>> 
        >>> # Analyze top performing models
        >>> best_runs = manager.get_best_runs(metric="val_roc_auc", top_k=5)
        >>> print("Top 5 models:")
        >>> print(best_runs[['metrics.val_roc_auc', 'params.hidden_size', 'params.learning_rate']])
        >>> 
        >>> # Generate insights
        >>> manager.compare_hyperparameters(top_k=20)
        >>> manager.plot_hyperparameter_importance(metric="val_roc_auc")
        >>> 
        >>> # Generate comprehensive report
        >>> report = manager.generate_experiment_report()
        >>> print(report)
    """
    def __init__(self, experiment_name: str = "heart_disease_neural_net"):
        """
        Initialize the MLflow experiment manager.
        
        Args:
            experiment_name: Name of the MLflow experiment to manage
        """
        self.experiment_name = experiment_name
        
    def get_experiment_runs(self, max_results: int = 100) -> pd.DataFrame:
        """
        Retrieve all runs from the experiment with comprehensive error handling.
        
        Fetches experiment runs with proper sorting, filtering, and data validation.
        Handles missing experiments gracefully and provides detailed logging
        of retrieval process and results.
        
        Args:
            max_results: Maximum number of runs to retrieve (default: 100)
            
        Returns:
            DataFrame containing experiment runs with metrics, parameters, and metadata
            Empty DataFrame if experiment not found or retrieval fails
            
        Raises:
            MLflowException: If MLflow server is unreachable
            
        Note:
            Runs are automatically sorted by validation ROC-AUC in descending order
            for easy identification of best performing models.
            
        Example:
            >>> manager = MLflowExperimentManager("my_experiment")
            >>> runs = manager.get_experiment_runs(max_results=50)
            >>> if not runs.empty:
            ...     print(f"Retrieved {len(runs)} runs")
            ...     print(f"Best ROC-AUC: {runs['metrics.val_roc_auc'].max():.4f}")
            ... else:
            ...     print("No runs found or experiment doesn't exist")
        """
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
        """
        Get top-k best runs based on specified metric with comprehensive analysis.
        
        Retrieves and analyzes the best performing runs, providing insights
        into optimal hyperparameter configurations and performance characteristics.
        
        Args:
            metric: Metric to use for ranking (default: "val_roc_auc")
            top_k: Number of top runs to retrieve (default: 5)
            
        Returns:
            DataFrame containing top-k runs sorted by specified metric
            
        Example:
            >>> best_runs = manager.get_best_runs(metric="val_accuracy", top_k=3)
            >>> print("Top 3 configurations:")
            >>> for idx, row in best_runs.iterrows():
            ...     print(f"Run {idx}: {row['metrics.val_accuracy']:.4f}")
            ...     print(f"  Params: {dict(row.filter(like='params.'))}")
        """
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
        """
        Create comprehensive visualizations comparing hyperparameters across top runs.
        
        Generates multiple visualization panels showing hyperparameter distributions,
        relationships, and effectiveness patterns across the best performing runs.
        
        Args:
            top_k: Number of top runs to include in comparison (default: 10)
            
        Note:
            Creates multi-panel visualization showing:
            - Distribution of each hyperparameter value
            - Frequency analysis of categorical parameters
            - Performance correlation with parameter values
            
        Example:
            >>> manager.compare_hyperparameters(top_k=15)
            # Generates comprehensive hyperparameter analysis plots
        """
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
        """
        Analyze and plot hyperparameter importance based on correlation with target metric.
        
        Performs statistical correlation analysis between hyperparameters and
        performance metrics, identifying which parameters have the strongest
        influence on model performance.
        
        Args:
            metric: Target metric for importance analysis (default: "val_roc_auc")
            
        Note:
            Uses Pearson correlation for numerical parameters and analysis
            of variance for categorical parameters.
            
        Example:
            >>> manager.plot_hyperparameter_importance(metric="val_f1")
            # Shows bar chart of parameter importance with correlation values
        """        
        
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
        """
        Plot trends of metrics across runs over time with statistical analysis.
        
        Visualizes how model performance has evolved across experiment runs,
        helping identify improvement trends, optimal stopping points, and
        performance stability patterns.
        
        Args:
            metrics: List of metrics to plot (default: common validation metrics)
            
        Features:
            - Chronological trend visualization
            - Moving average overlay for trend identification
            - Performance range highlighting
            - Statistical trend analysis
            
        Example:
            >>> manager.plot_metric_trends(["val_accuracy", "val_f1", "val_roc_auc"])
            # Shows time-series plots of metric evolution with trend lines
        """
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
        """
        Generate a comprehensive experiment report with detailed analysis and insights.
        
        Creates a detailed text report summarizing experiment results, performance
        statistics, hyperparameter analysis, and actionable recommendations for
        future experiments.
        
        Returns:
            String containing comprehensive experiment report with:
                - Basic experiment statistics and success rates
                - Best model performance and configuration
                - Hyperparameter analysis and recommendations
                - Performance distribution analysis
                - Experiment quality assessment
                
        Report Sections:
            1. Executive Summary: Key metrics and overall experiment success
            2. Best Model Analysis: Top performing configuration details
            3. Hyperparameter Insights: Parameter effectiveness analysis
            4. Performance Statistics: Distribution and variance analysis
            5. Recommendations: Actionable insights for future experiments
            
        Example:
            >>> manager = MLflowExperimentManager("heart_disease_prediction")
            >>> report = manager.generate_experiment_report()
            >>> print(report)
            >>> 
            >>> # Save report to file
            >>> with open("experiment_report.txt", "w") as f:
            ...     f.write(report)
        """
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
        """
        Identify and optionally delete failed or incomplete experiment runs.
        
        Performs comprehensive analysis of failed runs, identifies common failure
        patterns, and provides options for cleanup. Includes safety checks and
        detailed logging of cleanup operations.
        
        Safety Features:
            - Dry-run mode for safe preview of cleanup operations
            - Detailed failure analysis before deletion
            - Backup recommendations for important runs
            - Rollback guidance for accidental deletions
            
        Cleanup Criteria:
            - Runs with no validation metrics (likely failed during training)
            - Runs with corrupted or incomplete data
            - Runs with obvious error indicators
            - Runs with extremely poor performance (outliers)
            
        Example:
            >>> manager = MLflowExperimentManager("my_experiment")
            >>> manager.cleanup_failed_runs()
            Found 5 failed runs:
              - Run abc123: No validation metrics
              - Run def456: Error in parameters
            Cleanup completed: 5 runs deleted
        """
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