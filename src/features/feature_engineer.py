"""Feature engineering utilities for heart disease prediction."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif


class FeatureEngineer:
    """Feature engineering and analysis utilities."""
    
    def __init__(self, config):
        self.config = config
        
    def perform_eda(self, df: pd.DataFrame) -> None:
        """Perform comprehensive exploratory data analysis."""
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
        """Plot target variable distribution."""
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
        """Plot distributions of numerical features."""
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
        """Plot correlation matrix."""
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
        """Plot relationships between features and target."""
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
        """Analyze categorical features and their relationship with target."""
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
        """Analyze feature importance using different methods."""
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
        """Plot feature importance scores."""
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