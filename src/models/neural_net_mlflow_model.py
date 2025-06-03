"""Neural Network model with MLflow integration."""

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
    """Neural Network architecture for heart disease classification."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float, num_layers: int = 3):
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
        return self.network(x)


class MLflowNeuralNetModel(BaseModel):
    """Neural Network with MLflow tracking (local or Databricks)."""
    
    def __init__(self, config, use_databricks: bool = False):
        super().__init__(config)
        self.device = torch.device(config.device)
        self.experiment_name = "heart_disease_neural_net"
        self.use_databricks = use_databricks
        self.databricks_success = False
        
        self._setup_mlflow()
        
    def _setup_mlflow(self):
        """Setup MLflow tracking."""
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
        """Setup local MLflow."""
        mlflow.set_tracking_uri("file:./mlruns")
        try:
            mlflow.set_experiment(self.experiment_name)
        except:
            mlflow.create_experiment(self.experiment_name)
            mlflow.set_experiment(self.experiment_name)
        print("🏠 Using local MLflow tracking")
        print("💡 Run 'mlflow ui' to view results at http://localhost:5000")
    
    def get_hyperparameter_grid(self) -> Dict[str, List]:
        """Define hyperparameter search space."""
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
        """Perform hyperparameter tuning with MLflow tracking."""
        
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
        """Create input example and signature with proper data types."""
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
        """Train a single model with given parameters."""
        
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
        """Evaluate model."""
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
        """Train with hyperparameter tuning."""
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
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.astype(np.float32)).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float()
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, X):
        """Predict probabilities."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted first")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X.astype(np.float32)).to(self.device)
            outputs = self.model(X_tensor).squeeze()
        
        probs_class_1 = outputs.cpu().numpy()
        probs_class_0 = 1 - probs_class_1
        
        return np.column_stack((probs_class_0, probs_class_1))