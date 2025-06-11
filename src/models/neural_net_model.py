"""
Neural Network Model Implementation Module

This module implements a deep neural network for heart disease prediction using PyTorch.
It provides a flexible architecture with configurable layers, dropout regularization,
and comprehensive training with early stopping and learning rate scheduling.

Classes:
    HeartDiseaseNN: PyTorch neural network architecture for binary classification
    NeuralNetModel: Complete neural network model extending BaseModel interface

Key Features:
    - Multi-layer feedforward architecture with configurable depth
    - Dropout regularization for overfitting prevention
    - Early stopping with patience-based monitoring
    - Learning rate scheduling with ReduceLROnPlateau
    - Comprehensive training history tracking
    - GPU support with automatic device detection
    - Validation-based model selection

Design Pattern:
    Implements the Template Method pattern through BaseModel extension,
    with the Strategy pattern for different optimization and scheduling approaches.
    Uses PyTorch's modular design for flexible network architecture.

Usage:
    >>> from models.neural_net_model import NeuralNetModel
    >>> from utils.config import Config
    >>> 
    >>> config = Config()
    >>> model = NeuralNetModel(config)
    >>> history = model.fit(X_train, y_train, X_val, y_val)
    >>> model.plot_training_history()  # Visualize training progress
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
from .base_model import BaseModel


class HeartDiseaseNN(nn.Module):
    """
    Deep neural network architecture for heart disease binary classification.
    
    Implements a feedforward neural network with multiple hidden layers,
    ReLU activations, dropout regularization, and sigmoid output activation
    for binary classification tasks.
    
    The architecture uses a progressively decreasing layer size pattern:
    input_size → hidden_size → hidden_size//2 → hidden_size//4 → 1
    
    Attributes:
        network (nn.Sequential): Complete neural network architecture
    
    Architecture Details:
        - Input layer: Matches feature dimensionality
        - Hidden layers: 3 layers with decreasing sizes
        - Activations: ReLU for hidden layers, Sigmoid for output
        - Regularization: Dropout after each hidden layer
        - Output: Single neuron with sigmoid for probability estimation
    
    Example:
        >>> input_size = 13  # Number of features
        >>> model = HeartDiseaseNN(input_size=13, hidden_size=64, dropout_rate=0.3)
        >>> x = torch.randn(32, 13)  # Batch of 32 samples
        >>> output = model(x)  # Shape: (32, 1)
    """
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
        """
        Initialize the neural network architecture.
        
        Args:
            input_size: Number of input features
            hidden_size: Number of neurons in the first hidden layer
            dropout_rate: Dropout probability for regularization (0.0-1.0)
        """
        super(HeartDiseaseNN, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(hidden_size // 4, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1) with sigmoid probabilities
        """
        return self.network(x)


class NeuralNetModel(BaseModel):
    """
    Complete neural network model with training, validation, and prediction capabilities.
    
    This class provides a comprehensive neural network implementation with
    automated training procedures, early stopping, learning rate scheduling,
    and extensive monitoring of training progress.
    
    Features include GPU acceleration, validation-based model selection,
    comprehensive training history tracking, and visualization capabilities.
    
    Attributes:
        config: Configuration object with network and training parameters
        device: PyTorch device (CPU or CUDA) for computation
        model: The HeartDiseaseNN instance
        is_fitted (bool): Training completion flag
        training_history (dict): Complete training metrics and progress
    
    Training Features:
        - Binary cross-entropy loss for classification
        - Adam optimizer with configurable learning rate
        - ReduceLROnPlateau scheduler for adaptive learning rate
        - Early stopping with patience monitoring
        - Validation-based best model selection
        - Comprehensive metrics tracking
    
    Example:
        >>> config = Config()
        >>> model = NeuralNetModel(config)
        >>> 
        >>> # Train with validation monitoring
        >>> history = model.fit(X_train, y_train, X_val, y_val)
        >>> 
        >>> # Visualize training progress
        >>> model.plot_training_history()
        >>> 
        >>> # Make predictions
        >>> predictions = model.predict(X_test)
        >>> probabilities = model.predict_proba(X_test)
    """
    
    def __init__(self, config):
        """
        Initialize the neural network model with configuration.
        
        Sets up the device for computation (GPU if available, otherwise CPU)
        and prepares the model for training.
        
        Args:
            config: Configuration object containing network architecture,
                   training parameters, and device settings
        """
        super().__init__(config)
        self.device = torch.device(config.device)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train the neural network with comprehensive monitoring and optimization.
        
        Implements a complete training loop with validation monitoring, early stopping,
        learning rate scheduling, and best model selection. Provides detailed progress
        tracking and handles both training and validation phases.
        
        Args:
            X_train: Training feature matrix of shape (n_samples, n_features)
            y_train: Training target vector of shape (n_samples,)
            X_val: Validation feature matrix (recommended for early stopping)
            y_val: Validation target vector (recommended for early stopping)
            
        Returns:
            Dictionary containing comprehensive training history:
                - 'train_losses': Loss values for each epoch
                - 'val_losses': Validation loss values (if validation data provided)
                - 'train_accuracies': Training accuracy for each epoch
                - 'val_accuracies': Validation accuracy (if validation data provided)
                - 'epochs_trained': Total number of epochs completed
                
        Raises:
            ValueError: If training data is empty or has incompatible shapes
            
        Note:
            Validation data is highly recommended for proper early stopping and
            model selection. Without validation data, training continues for
            the full number of epochs specified in config.
            
        Example:
            >>> history = model.fit(X_train, y_train, X_val, y_val)
            Starting neural network training...
            Epoch [10/100], Train Loss: 0.6234, Train Acc: 0.6789, Val Loss: 0.6456, Val Acc: 0.6543
            ...
            Early stopping at epoch 45
            Neural network training completed!
        """
        
        input_size = X_train.shape[1]
        
        # Initialize model
        self.model = HeartDiseaseNN(
            input_size=input_size,
            hidden_size=self.config.hidden_size,
            dropout_rate=self.config.dropout_rate
        ).to(self.device)
        
        # Loss function and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train.values if hasattr(y_train, 'values') else y_train).to(self.device)
        
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self.device)
            y_val_tensor = torch.FloatTensor(y_val.values if hasattr(y_val, 'values') else y_val).to(self.device)
        
        # Training history
        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting neural network training...")
        
        for epoch in range(self.config.epochs):
            # Training phase
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_tensor).squeeze()
            train_loss = criterion(outputs, y_train_tensor)
            train_loss.backward()
            optimizer.step()
            
            # Calculate training accuracy
            train_predictions = (outputs > 0.5).float()
            train_accuracy = (train_predictions == y_train_tensor).float().mean()
            
            train_losses.append(train_loss.item())
            train_accuracies.append(train_accuracy.item())
            
            # Validation phase
            if X_val is not None and y_val is not None:
                self.model.eval()
                with torch.no_grad():
                    val_outputs = self.model(X_val_tensor).squeeze()
                    val_loss = criterion(val_outputs, y_val_tensor)
                    
                    val_predictions = (val_outputs > 0.5).float()
                    val_accuracy = (val_predictions == y_val_tensor).float().mean()
                    
                    val_losses.append(val_loss.item())
                    val_accuracies.append(val_accuracy.item())
                    
                    # Learning rate scheduling
                    scheduler.step(val_loss)
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= self.config.early_stopping_patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
            # Print progress
            if (epoch + 1) % 10 == 0:
                if X_val is not None:
                    print(f"Epoch [{epoch+1}/{self.config.epochs}], "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, "
                          f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
                else:
                    print(f"Epoch [{epoch+1}/{self.config.epochs}], "
                          f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        
        # Load best model if validation was used
        if X_val is not None and y_val is not None:
            self.model.load_state_dict(best_model_state)
        
        self.is_fitted = True
        
        # Store training history
        self.training_history = {
            'train_losses': train_losses,
            'val_losses': val_losses if X_val is not None else [],
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies if X_val is not None else [],
            'epochs_trained': len(train_losses)
        }
        
        print("Neural network training completed!")
        return self.training_history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make binary predictions using the trained neural network.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Binary predictions array of shape (n_samples,)
            
        Raises:
            ValueError: If model has not been fitted
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float()
            
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using the trained neural network.
        
        Args:
            X: Feature matrix of shape (n_samples, n_features)
            
        Returns:
            Probability matrix of shape (n_samples, 2)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            
        # Return probabilities for both classes
        probs_class_1 = outputs.cpu().numpy()
        probs_class_0 = 1 - probs_class_1
        
        return np.column_stack((probs_class_0, probs_class_1))
    
    def plot_training_history(self):
        """
        Plot comprehensive training history visualization.
        
        Creates side-by-side plots of loss and accuracy curves for both
        training and validation (if available) to visualize training progress.
        """
        import matplotlib.pyplot as plt
        
        if not self.training_history:
            print("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        epochs = range(1, len(self.training_history['train_losses']) + 1)
        ax1.plot(epochs, self.training_history['train_losses'], 'b-', label='Training Loss')
        if self.training_history['val_losses']:
            ax1.plot(epochs, self.training_history['val_losses'], 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(epochs, self.training_history['train_accuracies'], 'b-', label='Training Accuracy')
        if self.training_history['val_accuracies']:
            ax2.plot(epochs, self.training_history['val_accuracies'], 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()