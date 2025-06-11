"""
Data Loading and Preprocessing Module

This module provides comprehensive data loading, preprocessing, and preparation utilities
for the heart disease prediction ML pipeline. It handles all aspects of data preparation
from raw CSV loading to ready-to-use datasets for both traditional ML and deep learning models.

Classes:
    DataLoader: Main class handling all data operations including loading, preprocessing,
               splitting, scaling, and PyTorch dataset creation

Key Features:
    - Automated data loading with summary statistics
    - Categorical variable encoding with label preservation
    - Stratified train/validation/test splitting
    - Feature scaling with StandardScaler
    - PyTorch DataLoader creation for neural networks
    - Comprehensive data validation and logging

Design Pattern:
    Uses the Builder pattern approach where each method builds upon the previous
    data processing step, maintaining state and allowing for flexible pipeline construction.

Usage:
    >>> from data.data_loader import DataLoader
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> loader = DataLoader(config)
    >>> df = loader.load_data()
    >>> df_processed = loader.preprocess_data(df)
    >>> X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df_processed)
    >>> X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(X_train, X_val, X_test)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class DataLoader:
    """
    Comprehensive data loading and preprocessing handler for ML pipeline.
    
    This class manages the entire data preparation workflow from raw CSV files
    to ML-ready datasets. It provides methods for loading, preprocessing, splitting,
    scaling, and creating PyTorch DataLoaders while maintaining data integrity
    and providing detailed logging of each step.
    
    The class follows a sequential processing approach where each method builds
    upon the previous step, allowing for flexible pipeline construction and
    easy debugging of data issues.
    
    Attributes:
        config: Configuration object containing data processing parameters
        scaler (StandardScaler): Fitted scaler for feature normalization
        label_encoders (Dict[str, LabelEncoder]): Dictionary of fitted label encoders
                                                 for categorical features
        feature_names (List[str]): List of feature column names after preprocessing
    
    Example:
        >>> config = Config()
        >>> loader = DataLoader(config)
        >>> 
        >>> # Complete data preparation pipeline
        >>> df = loader.load_data()
        >>> df_processed = loader.preprocess_data(df)
        >>> splits = loader.split_data(df_processed)
        >>> X_train, X_val, X_test, y_train, y_val, y_test = splits
        >>> X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(
        ...     X_train, X_val, X_test
        ... )
        >>> 
        >>> # For neural networks
        >>> dataloaders = loader.create_torch_dataloaders(
        ...     X_train_scaled, X_val_scaled, X_test_scaled,
        ...     y_train, y_val, y_test
        ... )
    """
    
    def __init__(self, config):
        """
        Initialize the DataLoader with configuration settings.
        
        Args:
            config: Configuration object containing data processing parameters
                   including file paths, split ratios, scaling options, etc.
        """
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load the heart disease dataset from CSV file with validation.
        
        Loads the dataset and provides immediate feedback on data shape,
        target distribution, and basic statistics to help identify potential
        data issues early in the pipeline.
        
        Args:
            file_path: Optional path to CSV file. If None, uses config.data_path
            
        Returns:
            pandas.DataFrame: Loaded dataset with all original columns and data types
            
        Raises:
            FileNotFoundError: If the specified file path doesn't exist
            pd.errors.EmptyDataError: If the CSV file is empty
            ValueError: If the target column is not found in the dataset
            
        Example:
            >>> loader = DataLoader(config)
            >>> df = loader.load_data("data/heart.csv")
            Loaded dataset with shape: (1025, 12)
            Target distribution:
            0    508
            1    517
            Name: HeartDisease, dtype: int64
        """
        if file_path is None:
            file_path = self.config.data_path
            
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Target distribution:\n{df[self.config.target_column].value_counts()}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the dataset by encoding categorical variables and validation.
        
        Identifies categorical columns and applies label encoding while preserving
        the encoders for potential inverse transformation. Handles the target column
        appropriately by excluding it from categorical encoding.
        
        Args:
            df: Raw pandas DataFrame to preprocess
            
        Returns:
            pandas.DataFrame: Preprocessed dataset with encoded categorical variables
            
        Raises:
            ValueError: If the dataset is empty or target column is missing
            
        Note:
            Label encoders are stored in self.label_encoders for later use in
            feature interpretation or inverse transformation.
            
        Example:
            >>> df_processed = loader.preprocess_data(df)
            Categorical columns found: ['Sex', 'ChestPainType', 'RestingECG']
        """
        if df.empty:
            raise ValueError("Cannot preprocess empty dataset")
        
        df_processed = df.copy()
        
        # Identify categorical columns
        categorical_columns = df_processed.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != self.config.target_column]
        
        print(f"Categorical columns found: {list(categorical_columns)}")
        
        # Encode categorical variables
        for col in categorical_columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col])
            self.label_encoders[col] = le
            
        return df_processed
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
                                                    pd.Series, pd.Series, pd.Series]:
        """
        Split dataset into stratified train, validation, and test sets.
        
        Performs a two-step stratified split to ensure balanced class distribution
        across all splits. First splits into (train+val) vs test, then splits
        (train+val) into train vs validation.
        
        Args:
            df: Preprocessed pandas DataFrame ready for splitting
            
        Returns:
            Tuple containing:
                - X_train (pd.DataFrame): Training features
                - X_val (pd.DataFrame): Validation features  
                - X_test (pd.DataFrame): Test features
                - y_train (pd.Series): Training targets
                - y_val (pd.Series): Validation targets
                - y_test (pd.Series): Test targets
                
        Raises:
            ValueError: If dataset is empty or target column missing
            
        Note:
            Feature names are stored in self.feature_names for later reference.
            All splits maintain the original class distribution through stratification.
            
        Example:
            >>> X_train, X_val, X_test, y_train, y_val, y_test = loader.split_data(df_processed)
            Train set size: 656
            Validation set size: 164
            Test set size: 205
        """
        if df.empty:
            raise ValueError("Cannot split empty dataset")
        X = df.drop(columns=[self.config.target_column])
        y = df[self.config.target_column]
        
        self.feature_names = X.columns.tolist()
        
        # First split: train + val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = self.config.val_size / (1 - self.config.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size_adjusted,
            random_state=self.config.random_state,
            stratify=y_temp
        )
        
        print(f"Train set size: {len(X_train)}")
        print(f"Validation set size: {len(X_val)}")
        print(f"Test set size: {len(X_test)}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame, 
                      X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Scale features using StandardScaler with proper train/val/test handling.
        
        Applies feature scaling if enabled in configuration. Fits the scaler on
        training data only and transforms all splits to prevent data leakage.
        Returns numpy arrays ready for ML model consumption.
        
        Args:
            X_train: Training features DataFrame
            X_val: Validation features DataFrame  
            X_test: Test features DataFrame
            
        Returns:
            Tuple of scaled feature arrays:
                - X_train_scaled (np.ndarray): Scaled training features
                - X_val_scaled (np.ndarray): Scaled validation features
                - X_test_scaled (np.ndarray): Scaled test features
                
        Note:
            The fitted scaler is stored in self.scaler for later use.
            If scaling is disabled in config, returns original values as numpy arrays.
            
        Example:
            >>> X_train_scaled, X_val_scaled, X_test_scaled = loader.scale_features(
            ...     X_train, X_val, X_test
            ... )
            Features scaled using StandardScaler
        """
        
        if self.config.scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            print("Features scaled using StandardScaler")
            return X_train_scaled, X_val_scaled, X_test_scaled
        else:
            return X_train.values, X_val.values, X_test.values
    
    def create_torch_dataloaders(self, X_train: np.ndarray, X_val: np.ndarray, 
                                X_test: np.ndarray, y_train: pd.Series, 
                                y_val: pd.Series, y_test: pd.Series) -> Dict[str, TorchDataLoader]:
        """
        Create PyTorch DataLoaders for neural network training and evaluation.
        
        Converts numpy arrays and pandas series to PyTorch tensors and creates
        DataLoader objects with appropriate batch sizes and shuffling settings
        for neural network training.
        
        Args:
            X_train: Training features array
            X_val: Validation features array
            X_test: Test features array
            y_train: Training targets series
            y_val: Validation targets series
            y_test: Test targets series
            
        Returns:
            Dictionary containing PyTorch DataLoaders:
                - 'train': Training DataLoader (shuffled)
                - 'val': Validation DataLoader (not shuffled)
                - 'test': Test DataLoader (not shuffled)
                
        Note:
            Training DataLoader is shuffled for better gradient updates.
            Validation and test DataLoaders maintain order for reproducible evaluation.
            
        Example:
            >>> dataloaders = loader.create_torch_dataloaders(
            ...     X_train_scaled, X_val_scaled, X_test_scaled,
            ...     y_train, y_val, y_test
            ... )
            >>> train_loader = dataloaders['train']
            >>> for batch_X, batch_y in train_loader:
            ...     # Neural network training loop
            ...     pass
        """
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        y_train_tensor = torch.FloatTensor(y_train.values)
        y_val_tensor = torch.FloatTensor(y_val.values)
        y_test_tensor = torch.FloatTensor(y_test.values)
        
        # Create datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        # Create dataloaders
        dataloaders = {
            'train': TorchDataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True),
            'val': TorchDataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False),
            'test': TorchDataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False)
        }
        
        return dataloaders