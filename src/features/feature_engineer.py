"""
Feature Engineering and Exploratory Data Analysis Module

This module provides comprehensive feature engineering and exploratory data analysis
utilities for the heart disease prediction pipeline. It handles all aspects of data
exploration, visualization, and feature analysis to support informed modeling decisions.

Classes:
    FeatureEngineer: Main class providing EDA capabilities, feature analysis,
                    and visualization utilities for understanding data patterns

Key Features:
    - Comprehensive exploratory data analysis with automated visualizations
    - Statistical feature importance analysis using multiple methods
    - Correlation analysis and visualization
    - Categorical and numerical feature relationship analysis
    - Feature distribution analysis and outlier detection
    - Interactive plotting with detailed insights

Design Pattern:
    Uses the Strategy pattern for different analysis types, allowing flexible
    combination of analysis methods while maintaining consistent interfaces.

Usage:
    >>> from features.feature_engineer import FeatureEngineer
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> engineer = FeatureEngineer(config)
    >>> engineer.perform_eda(df)
    >>> importance_results = engineer.feature_importance_analysis(X, y, feature_names)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class FeatureEngineer:
    """
    Comprehensive feature engineering and exploratory data analysis toolkit.
    
    This class provides a complete suite of tools for understanding data patterns,
    relationships, and feature importance. It generates automated visualizations
    and statistical analyses to support data-driven modeling decisions.
    
    The class emphasizes visual exploration combined with statistical rigor,
    providing both intuitive plots and quantitative measures for feature analysis.
    
    Attributes:
        config: Configuration object containing visualization and analysis parameters
    
    Methods:
        perform_eda: Complete exploratory data analysis with visualizations
        analyze_categorical_features: Specialized analysis for categorical variables
        feature_importance_analysis: Statistical feature importance evaluation
        
    Example:
        >>> config = Config()
        >>> engineer = FeatureEngineer(config)
        >>> 
        >>> # Perform comprehensive EDA
        >>> engineer.perform_eda(df)
        >>> 
        >>> # Analyze feature importance
        >>> importance_results = engineer.feature_importance_analysis(
        ...     X, y, feature_names
        ... )
        >>> 
        >>> # Specialized categorical analysis
        >>> engineer.analyze_categorical_features(df)
    """
    
    def __init__(self, config):
        """
        Initialize the FeatureEngineer with configuration settings.
        
        Args:
            config: Configuration object containing visualization parameters,
                   figure sizes, and analysis settings
        """
        self.config = config
        
    def perform_eda(self, df: pd.DataFrame) -> None:
        """
        Perform comprehensive exploratory data analysis with automated visualizations.
        
        Conducts a full EDA workflow including data shape analysis, missing value
        detection, target distribution analysis, feature distributions, correlation
        analysis, and feature-target relationships. Generates multiple publication-
        quality visualizations for data understanding.
        
        Args:
            df: pandas DataFrame to analyze
            
        Raises:
            ValueError: If DataFrame is empty or target column is missing
            
        Note:
            This method generates multiple plots and prints extensive statistics.
            Ensure sufficient display space and consider running in sections for
            large datasets.
            
        Example:
            >>> engineer.perform_eda(df)
            === EXPLORATORY DATA ANALYSIS ===
            
            Dataset shape: (1025, 12)
            Data types:
            Age                 int64
            Sex                object
            ChestPainType      object
            ...
        """
        print("=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic information
        print(f"\nDataset shape: {df.shape}")
        print(f"\nData types:\n{df.dtypes}")
        print(f"\nMissing values:\n{df.isnull().sum()}")
        
        # Target distribution
        target_col = self.config.target_column
        print(f"\nTarget distribution:")
        print(df[target_col].value_counts())
        print(f"Target proportion: {df[target_col].mean():.3f}")
        
        # Statistical summary
        print(f"\nStatistical summary:")
        print(df.describe())
        
        # Visualizations
        self._plot_target_distribution(df)
        self._plot_feature_distributions(df)
        self._plot_correlation_matrix(df)
        self._plot_feature_target_relationships(df)
        
    def _plot_target_distribution(self, df: pd.DataFrame) -> None:
        """
        Plot target variable distribution with both count and percentage views.
        
        Creates a two-panel visualization showing the absolute counts and
        relative proportions of target classes to assess class balance.
        
        Args:
            df: DataFrame containing the target variable
        """
        plt.figure(figsize=(10, 4))
        
        plt.subplot(1, 2, 1)
        df[self.config.target_column].value_counts().plot(kind='bar')
        plt.title('Heart Disease Distribution (Count)')
        plt.xlabel('Heart Disease')
        plt.ylabel('Count')
        plt.xticks([0, 1], ['No Disease', 'Disease'], rotation=0)
        
        plt.subplot(1, 2, 2)
        df[self.config.target_column].value_counts(normalize=True).plot(kind='pie', autopct='%1.1f%%')
        plt.title('Heart Disease Distribution (Percentage)')
        plt.ylabel('')
        
        plt.tight_layout()
        plt.show()
        
    def _plot_feature_distributions(self, df: pd.DataFrame) -> None:
        """
        Plot distribution histograms for all numerical features.
        
        Creates a grid of histograms showing the distribution of each numerical
        feature to identify skewness, outliers, and distribution patterns.
        
        Args:
            df: DataFrame containing numerical features to plot
        """
        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != self.config.target_column]
        
        n_features = len(numerical_features)
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(16, 4 * n_rows))
        
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            plt.hist(df[feature], bins=30, alpha=0.7, edgecolor='black')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Frequency')
            
        plt.tight_layout()
        plt.show()
        
    def _plot_correlation_matrix(self, df: pd.DataFrame) -> None:
        """
        Plot correlation matrix heatmap for numerical features.
        
        Creates a correlation heatmap with enhanced visualization including
        color coding, annotations, and masking of upper triangle for clarity.
        
        Args:
            df: DataFrame containing numerical features for correlation analysis
        """
        # Select only numerical columns
        numerical_df = df.select_dtypes(include=[np.number])
        
        plt.figure(figsize=(12, 10))
        correlation_matrix = numerical_df.corr()
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Feature Correlation Matrix')
        plt.tight_layout()
        plt.show()
        
    def _plot_feature_target_relationships(self, df: pd.DataFrame) -> None:
        """
        Plot relationships between numerical features and target variable.
        
        Creates box plots showing the distribution of each numerical feature
        stratified by target class to identify discriminative features.
        
        Args:
            df: DataFrame containing features and target variable
        """
        numerical_features = df.select_dtypes(include=[np.number]).columns
        numerical_features = [col for col in numerical_features if col != self.config.target_column]
        
        n_features = len(numerical_features)
        n_cols = 3
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 5 * n_rows))
        
        for i, feature in enumerate(numerical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Box plot for each class
            df.boxplot(column=feature, by=self.config.target_column, ax=plt.gca())
            plt.title(f'{feature} by Heart Disease')
            plt.suptitle('')  # Remove automatic title
            
        plt.tight_layout()
        plt.show()
        
    def analyze_categorical_features(self, df: pd.DataFrame) -> None:
        """
        Analyze categorical features and their relationship with target variable.
        
        Creates stacked bar plots showing the proportion of target classes within
        each category of categorical features to identify important patterns.
        
        Args:
            df: DataFrame containing categorical features and target variable
            
        Note:
            This method is designed to work with the original DataFrame before
            categorical encoding to show meaningful category names.
        """
        categorical_features = df.select_dtypes(include=['object']).columns
        categorical_features = [col for col in categorical_features if col != self.config.target_column]
        
        if len(categorical_features) == 0:
            print("No categorical features found after preprocessing.")
            return
            
        n_features = len(categorical_features)
        n_cols = 2
        n_rows = (n_features + n_cols - 1) // n_cols
        
        plt.figure(figsize=(15, 6 * n_rows))
        
        for i, feature in enumerate(categorical_features, 1):
            plt.subplot(n_rows, n_cols, i)
            
            # Create cross-tabulation
            ct = pd.crosstab(df[feature], df[self.config.target_column], normalize='index')
            ct.plot(kind='bar', ax=plt.gca())
            plt.title(f'{feature} vs Heart Disease')
            plt.xlabel(feature)
            plt.ylabel('Proportion')
            plt.legend(['No Disease', 'Disease'])
            plt.xticks(rotation=45)
            
        plt.tight_layout()
        plt.show()
        
    def feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series, 
                                  feature_names: List[str]) -> Dict[str, np.ndarray]:
        """
        Analyze feature importance using multiple statistical methods.
        
        Applies both F-statistic (ANOVA) and mutual information methods to
        assess feature importance, providing complementary perspectives on
        feature relevance for the target variable.
        
        Args:
            X: Feature matrix (preprocessed)
            y: Target variable series
            feature_names: List of feature names corresponding to X columns
            
        Returns:
            Dictionary containing:
                - 'f_scores': F-statistic scores for each feature
                - 'mi_scores': Mutual information scores for each feature
                - 'importance_df': DataFrame with ranked feature importance
                
        Raises:
            ValueError: If X and y have incompatible shapes or feature_names length mismatch
            
        Example:
            >>> importance_results = engineer.feature_importance_analysis(X, y, feature_names)
            >>> print(importance_results['importance_df'].head())
        """
        print("=== FEATURE IMPORTANCE ANALYSIS ===")
        
        # Statistical feature selection
        f_selector = SelectKBest(score_func=f_classif, k='all')
        f_selector.fit(X, y)
        f_scores = f_selector.scores_
        
        # Mutual information
        mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
        mi_selector.fit(X, y)
        mi_scores = mi_selector.scores_
        
        # Create feature importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'F_Score': f_scores,
            'Mutual_Information': mi_scores
        })
        
        # Normalize scores for comparison
        importance_df['F_Score_Normalized'] = (importance_df['F_Score'] - importance_df['F_Score'].min()) / \
                                             (importance_df['F_Score'].max() - importance_df['F_Score'].min())
        importance_df['MI_Normalized'] = (importance_df['Mutual_Information'] - importance_df['Mutual_Information'].min()) / \
                                        (importance_df['Mutual_Information'].max() - importance_df['Mutual_Information'].min())
        
        # Sort by F-score
        importance_df = importance_df.sort_values('F_Score', ascending=False)
        
        print("Feature Importance Ranking:")
        print(importance_df)
        
        # Plot feature importance
        self._plot_feature_importance(importance_df)
        
        return {
            'f_scores': f_scores,
            'mi_scores': mi_scores,
            'importance_df': importance_df
        }
        
    def _plot_feature_importance(self, importance_df: pd.DataFrame) -> None:
        """
        Plot comprehensive feature importance visualization.
        
        Creates a multi-panel visualization showing F-scores, mutual information,
        and combined importance scores with statistical significance indicators.
        
        Args:
            importance_df: DataFrame containing feature importance metrics
        """
        plt.figure(figsize=(15, 8))
        
        plt.subplot(1, 2, 1)
        plt.barh(range(len(importance_df)), importance_df['F_Score_Normalized'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Normalized F-Score')
        plt.title('Feature Importance (F-Score)')
        plt.gca().invert_yaxis()
        
        plt.subplot(1, 2, 2)
        plt.barh(range(len(importance_df)), importance_df['MI_Normalized'])
        plt.yticks(range(len(importance_df)), importance_df['Feature'])
        plt.xlabel('Normalized Mutual Information')
        plt.title('Feature Importance (Mutual Information)')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        plt.show()