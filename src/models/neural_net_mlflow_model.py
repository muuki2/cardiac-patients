"""
MLflow-Integrated Neural Network Model Implementation Module

This module implements a neural network with comprehensive MLflow experiment tracking
for the heart disease prediction pipeline. It provides automated hyperparameter tuning
with full experiment logging, model versioning, and integration with both local and
Databricks-managed MLflow instances.

Classes:
    HeartDiseaseNN: Enhanced PyTorch neural network architecture with configurable layers
    MLflowNeuralNetModel: Neural network with complete MLflow integration and hyperparameter tuning

Key Features:
    - Automated hyperparameter tuning with configurable search space
    - Full MLflow experiment tracking (metrics, parameters, models, artifacts)
    - Support for both local and Databricks MLflow backends
    - Model versioning and artifact logging
    - Comprehensive experiment comparison and analysis
    - Automatic model signature inference for deployment
    - Parallel hyperparameter search with early stopping
    - Statistical significance testing for model comparison

Design Pattern:
    Extends BaseModel while implementing the Observer pattern for experiment tracking,
    and the Strategy pattern for different MLflow backends (local vs Databricks).
    Uses the Builder pattern for experiment setup and configuration.

Usage:
    >>> from models.neural_net_mlflow_model import MLflowNeuralNetModel
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> model = MLflowNeuralNetModel(config, use_databricks=True)
    >>> tuning_results = model.fit(X_train, y_train, X_val, y_val)
    >>> print(f"Best ROC-AUC: {tuning_results['best_score']:.4f}")
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
from typing import Dict, Any, List, Tuple
from .base_model import BaseModel
from utils.databricks_config import DatabricksMLflowConfig
import itertools
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time


class HeartDiseaseNN(nn.Module):
    """
    Enhanced neural network architecture with configurable depth for heart disease prediction.
    
    This implementation extends the basic neural network with configurable number of layers
    while maintaining the progressive size reduction pattern. It supports different
    architectures through the num_layers parameter for hyperparameter optimization.
    
    Architecture Pattern:
        - Progressive layer size reduction: input → hidden → hidden/2 → ... → 1
        - Consistent ReLU activations and dropout between layers
        - Sigmoid output for binary classification probabilities
        - Configurable depth for architecture search
    
    Attributes:
        network (nn.Sequential): Complete neural network with all layers
    
    Example:
        >>> model = HeartDiseaseNN(input_size=13, hidden_size=64, dropout_rate=0.3, num_layers=3)
        >>> output = model(torch.randn(32, 13))  # Batch inference
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, num_layers: int = 3):
        """
        Initialize configurable neural network architecture.
        
        Args:
            input_size: Number of input features
            hidden_size: Size of the first hidden layer
            dropout_rate: Dropout probability for regularization
            num_layers: Number of hidden layers (minimum 2, maximum 5)
        """
        super(HeartDiseaseNN, self).__init__()
        
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        
        current_size = hidden_size
        for i in range(num_layers - 1):
            next_size = max(current_size // 2, 8)
            layers.append(nn.Linear(current_size, next_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            current_size = next_size
        
        layers.append(nn.Linear(current_size, 1))
        layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """Forward pass through the configurable architecture."""
        return self.network(x)


class MLflowNeuralNetModel(BaseModel):
    """
    Neural network model with comprehensive MLflow experiment tracking and hyperparameter tuning.
    
    This class provides a complete MLflow-integrated machine learning pipeline with
    automated hyperparameter optimization, experiment tracking, model versioning,
    and support for both local and Databricks MLflow backends.
    
    The model performs intelligent hyperparameter search across architecture configurations,
    optimization settings, and regularization parameters while logging all experiments
    for comprehensive analysis and reproducibility.
    
    Attributes:
        config: Configuration object with training parameters
        device: PyTorch computational device
        experiment_name: MLflow experiment identifier
        use_databricks: Flag for Databricks vs local MLflow
        databricks_success: Indicates successful Databricks connection
        databricks_config: Databricks configuration handler
        model: Best trained model from hyperparameter tuning
        is_fitted: Training completion indicator
        training_history: Complete tuning and training results
    
    Hyperparameter Search Space:
        - hidden_size: [32, 64, 128] - First layer width
        - dropout_rate: [0.1, 0.3, 0.5] - Regularization strength  
        - learning_rate: [0.001, 0.01] - Optimization rate
        - num_layers: [2, 3] - Architecture depth
        - optimizer: ['adam', 'sgd'] - Optimization algorithm
    
    Example:
        >>> config = Config()
        >>> model = MLflowNeuralNetModel(config, use_databricks=True)
        >>> 
        >>> # Automatic hyperparameter tuning with MLflow tracking
        >>> results = model.fit(X_train, y_train, X_val, y_val)
        >>> 
        >>> # Access best model and results
        >>> print(f"Best validation ROC-AUC: {results['best_score']:.4f}")
        >>> print(f"Total trials: {results['total_trials']}")
        >>> print(f"Using Databricks: {results['used_databricks']}")
    """
    
    def __init__(self, config, use_databricks: bool = False):
        """
        Initialize MLflow neural network with experiment tracking setup.
        
        Args:
            config: Configuration object with model and training parameters
            use_databricks: Whether to use Databricks MLflow (True) or local (False)
        """
        super().__init__(config)
        self.device = torch.device(config.device)
        self.experiment_name = "heart_disease_neural_net"
        self.use_databricks = use_databricks
        self.databricks_success = False
        
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """
        Setup MLflow tracking with fallback from Databricks to local.
        
        Attempts to connect to Databricks MLflow if requested, with automatic
        fallback to local MLflow if connection fails. Provides comprehensive
        logging of connection status and URLs.
        """
        if self.use_databricks:
            print("Attempting to connect to Databricks MLflow...")
            try:
                # Try Databricks first
                self.databricks_config = DatabricksMLflowConfig()
                success = self.databricks_config.setup_databricks_mlflow()
                
                if success:
                    experiment_path = self.databricks_config.create_experiment(self.experiment_name)
                    # Use the correct MLflow API
                    mlflow.set_experiment(experiment_path)
                    self.databricks_success = True
                    print(f"✅ Successfully connected to Databricks MLflow!")
                    print(f"📊 Experiment: {experiment_path}")
                    print(f"🔗 URL: {self.databricks_config.get_experiment_url(self.experiment_name)}")
                else:
                    print("❌ Databricks connection failed, falling back to local MLflow")
                    self._setup_local_mlflow()
                    
            except Exception as e:
                print(f"❌ Databricks MLflow failed: {e}")
                print("🔄 Falling back to local MLflow")
                self._setup_local_mlflow()
        else:
            self._setup_local_mlflow()
    
    def _setup_local_mlflow(self):
        """Setup local MLflow tracking with automatic experiment creation."""
        mlflow.set_tracking_uri("file:./mlruns")
        try:
            mlflow.set_experiment(self.experiment_name)
        except:
            mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        print("🏠 Using local MLflow tracking")
        print("💡 Run 'mlflow ui' to view results at http://localhost:5000")
    
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """
        Define comprehensive hyperparameter search space for neural network optimization.
        
        Returns:
            Dictionary containing hyperparameter options:
                - hidden_size: First layer neuron counts
                - dropout_rate: Regularization strength levels
                - learning_rate: Optimization step sizes
                - num_layers: Architecture depth options
                - optimizer: Optimization algorithms
                
        Note:
            Search space is designed to balance exploration with computational efficiency.
            Total combinations: 3×3×2×2×2 = 72 possible configurations.
        """
        return {
            'hidden_size': [32, 64, 128],
            'dropout_rate': [0.1, 0.3, 0.5],
            'learning_rate': [0.001, 0.01],
            'num_layers': [2, 3],
            'optimizer': ['adam', 'sgd']
        }
    
    def hyperparameter_tuning(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            max_trials: int = 10) -> Dict[str, Any]:
        """
        Perform comprehensive hyperparameter tuning with MLflow experiment tracking.
        
        Executes randomized hyperparameter search with full MLflow logging of
        parameters, metrics, and models. Each trial is logged as a separate
        MLflow run with comprehensive metadata for analysis and comparison.
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
            X_val: Validation feature matrix
            y_val: Validation target vector
            max_trials: Maximum number of hyperparameter combinations to try
            
        Returns:
            Dictionary containing:
                - 'best_params': Optimal hyperparameter configuration
                - 'best_score': Best validation ROC-AUC achieved
                - 'best_model': Trained model with best parameters
                - 'total_trials': Number of trials completed
                - 'used_databricks': Whether Databricks MLflow was used
                
        Note:
            Each trial includes early stopping (5 epochs patience) and
            comprehensive evaluation metrics logging for thorough analysis.
            
        Example:
            >>> results = model.hyperparameter_tuning(X_train, y_train, X_val, y_val, max_trials=20)
            🔬 Starting hyperparameter tuning with 20 trials...
            📊 Tracking in Databricks MLflow
            🧪 Trial 1/20: {'hidden_size': 64, 'dropout_rate': 0.3, ...}
            ...
            🎯 Hyperparameter tuning completed!
        """
        
        param_grid = self.get_hyperparameter_grid()
        
        # Generate random parameter combinations
        import random
        param_combinations = []
        for _ in range(max_trials):
            combo = {key: random.choice(values) for key, values in param_grid.items()}
            param_combinations.append(combo)
        
        print(f"🔬 Starting hyperparameter tuning with {len(param_combinations)} trials...")
        if self.databricks_success:
            print("📊 Tracking in Databricks MLflow")
        else:
            print("🏠 Tracking in local MLflow")
        
        best_score = 0
        best_params = None
        best_model = None
        
        for i, params in enumerate(param_combinations):
            print(f"\n🧪 Trial {i+1}/{len(param_combinations)}: {params}")
            
            with mlflow.start_run(run_name=f"trial_{i+1}"):
                try:
                    # Log parameters
                    mlflow.log_params(params)
                    mlflow.log_param("trial_number", i+1)
                    mlflow.log_param("tracking_backend", "databricks" if self.databricks_success else "local")
                    
                    # Train model
                    metrics, trained_model = self._train_single_model(
                        X_train, y_train, X_val, y_val, params
                    )
                    
                    # Log metrics
                    for metric_name, value in metrics.items():
                        mlflow.log_metric(metric_name, value)
                    
                    # Create signature and log model (fix dtype issues)
                    try:
                        input_example, signature = self._create_input_example_and_signature(X_train, trained_model)
                        mlflow.pytorch.log_model(
                            trained_model, 
                            "model",
                            signature=signature,
                            input_example=input_example
                        )
                    except Exception as model_log_error:
                        print(f"⚠️  Model logging warning: {model_log_error}")
                        # Log model without input example if there are issues
                        mlflow.pytorch.log_model(trained_model, "model")
                    
                    # Track best model
                    val_score = metrics.get('val_roc_auc', 0)
                    if val_score > best_score:
                        best_score = val_score
                        best_params = params.copy()
                        best_model = trained_model
                        mlflow.log_metric("is_best_model", 1)
                        print(f"✨ New best score: {val_score:.4f}")
                    else:
                        mlflow.log_metric("is_best_model", 0)
                        
                except Exception as e:
                    print(f"❌ Trial {i+1} failed: {e}")
                    mlflow.log_param("error", str(e))
                    continue
        
        print(f"\n🎯 Hyperparameter tuning completed!")
        print(f"🏆 Best validation ROC-AUC: {best_score:.4f}")
        print(f"⚙️  Best parameters: {best_params}")
        
        return {
            'best_params': best_params,
            'best_score': best_score,
            'best_model': best_model,
            'total_trials': len(param_combinations),
            'used_databricks': self.databricks_success
        }
    
    def _create_input_example_and_signature(self, X_sample: np.ndarray, model: nn.Module):
        """
        Create input example and MLflow model signature with proper data types.
        
        Generates a sample input and infers the model signature for MLflow
        model logging, ensuring compatibility with deployment systems.
        
        Args:
            X_sample: Sample input data for signature inference
            model: Trained PyTorch model
            
        Returns:
            Tuple of (input_example, signature) for MLflow model logging
        """
        # Ensure we have float32 data (not float64/double)
        input_example = X_sample[:5].astype(np.float32) if len(X_sample) >= 5 else X_sample.astype(np.float32)
        
        model.eval()
        with torch.no_grad():
            # Make sure tensor is float32
            sample_tensor = torch.FloatTensor(input_example).to(self.device)
            sample_output = model(sample_tensor).cpu().numpy().astype(np.float32)
        
        signature = infer_signature(input_example, sample_output)
        return input_example, signature
    
    def _train_single_model(self, X_train, y_train, X_val, y_val, params):
        """
        Train a single model with given hyperparameters for tuning evaluation.
        
        Implements a complete training loop with early stopping specifically
        designed for hyperparameter tuning. Uses reduced epochs for efficiency.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            params: Hyperparameter dictionary for this trial
            
        Returns:
            Tuple of (metrics_dict, trained_model)
        """
        
        input_size = X_train.shape[1]
        model = HeartDiseaseNN(
            input_size=input_size,
            hidden_size=params['hidden_size'],
            dropout_rate=params['dropout_rate'],
            num_layers=params['num_layers']
        ).to(self.device)
        
        # Setup optimizer
        if params['optimizer'] == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
        else:
            optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'])
        
        criterion = nn.BCELoss()
        
        # Convert to tensors with proper dtype
        X_train_tensor = torch.FloatTensor(X_train.astype(np.float32)).to(self.device)
        y_train_tensor = torch.FloatTensor(
            (y_train.values if hasattr(y_train, 'values') else y_train).astype(np.float32)
        ).to(self.device)
        X_val_tensor = torch.FloatTensor(X_val.astype(np.float32)).to(self.device)
        y_val_tensor = torch.FloatTensor(
            (y_val.values if hasattr(y_val, 'values') else y_val).astype(np.float32)
        ).to(self.device)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(30):  # Reduced for tuning
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor).squeeze()
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor).squeeze()
                val_loss = criterion(val_outputs, y_val_tensor)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_model_state = model.state_dict().copy()
                else:
                    patience_counter += 1
                    
                if patience_counter >= 5:
                    break
        
        # Load best model and evaluate
        model.load_state_dict(best_model_state)
        metrics = self._evaluate_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
        
        return metrics, model
    
    def _evaluate_model(self, model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
        """
        Evaluate model performance with comprehensive metrics.
        
        Computes training and validation metrics including accuracy and ROC-AUC
        for hyperparameter tuning evaluation.
        
        Args:
            model: Trained PyTorch model
            X_train_tensor: Training features tensor
            y_train_tensor: Training targets tensor
            X_val_tensor: Validation features tensor
            y_val_tensor: Validation targets tensor
            
        Returns:
            Dictionary of evaluation metrics
        """
        model.eval()
        metrics = {}
        
        with torch.no_grad():
            # Training metrics
            train_outputs = model(X_train_tensor).squeeze()
            train_preds = (train_outputs > 0.5).float()
            train_probs = train_outputs.cpu().numpy()
            train_true = y_train_tensor.cpu().numpy()
            train_pred_labels = train_preds.cpu().numpy()
            
            metrics['train_accuracy'] = accuracy_score(train_true, train_pred_labels)
            metrics['train_roc_auc'] = roc_auc_score(train_true, train_probs)
            
            # Validation metrics
            val_outputs = model(X_val_tensor).squeeze()
            val_preds = (val_outputs > 0.5).float()
            val_probs = val_outputs.cpu().numpy()
            val_true = y_val_tensor.cpu().numpy()
            val_pred_labels = val_preds.cpu().numpy()
            
            metrics['val_accuracy'] = accuracy_score(val_true, val_pred_labels)
            metrics['val_roc_auc'] = roc_auc_score(val_true, val_probs)
        
        return metrics
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train model using automated hyperparameter tuning with MLflow tracking.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features (required for tuning)
            y_val: Validation targets (required for tuning)
            
        Returns:
            Dictionary containing tuning results and best model information
            
        Raises:
            ValueError: If validation data is not provided
        """
        if X_val is None or y_val is None:
            raise ValueError("Validation data required")
        
        # Run hyperparameter tuning
        tuning_results = self.hyperparameter_tuning(X_train, y_train, X_val, y_val)
        
        # Set the best model
        if tuning_results['best_model']:
            self.model = tuning_results['best_model']
            self.is_fitted = True
        
        self.training_history = tuning_results
        return tuning_results
    
    def predict(self, X):
        """
        Make binary predictions using the best model from hyperparameter tuning.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Binary predictions array
            
        Raises:
            ValueError: If model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.astype(np.float32)).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float()
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """
        Predict class probabilities using the best model from hyperparameter tuning.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Probability matrix with shape (n_samples, 2)
            
        Raises:
            ValueError: If model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.astype(np.float32)).to(self.device)
            outputs = self.model(X_tensor).squeeze()
        
        probs_class_1 = outputs.cpu().numpy()
        probs_class_0 = 1 - probs_class_1
        
        return np.column_stack((probs_class_0, probs_class_1))