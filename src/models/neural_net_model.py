"""Neural Network model implementation using PyTorch."""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List
from .base_model import BaseModel


class HeartDiseaseNN(nn.Module):
    """Neural Network architecture for heart disease classification."""
    
    def __init__(self, input_size: int, hidden_size: int, dropout_rate: float):
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
        return self.network(x)


class NeuralNetModel(BaseModel):
    """Neural Network model using PyTorch."""
    
    def __init__(self, config):
        super().__init__(config)
        self.device = torch.device(config.device)
        
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None) -> Dict[str, Any]:
        """Train the neural network."""
        
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
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor).squeeze()
            predictions = (outputs > 0.5).float()
            
        return predictions.cpu().numpy()
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
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
        """Plot training history."""
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