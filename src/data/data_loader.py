"""Data loading and preprocessing utilities."""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, TensorDataset


class DataLoader:
    """Handles data loading, preprocessing, and splitting."""
    
    def __init__(self, config):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """Load the heart disease dataset."""
        if file_path is None:
            file_path = self.config.data_path
            
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        print(f"Target distribution:\n{df[self.config.target_column].value_counts()}")
        
        return df
    
    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess the data by encoding categorical variables."""
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
        """Split data into train, validation, and test sets."""
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
        """Scale features using StandardScaler."""
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
        """Create PyTorch DataLoaders for neural network training."""
        
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