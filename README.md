# Heart Disease Prediction ML Pipeline

A comprehensive machine learning project for predicting heart disease using multiple algorithms, featuring a modular architecture, extensive visualization, and experiment tracking with MLflow.

**Student:** Murat Kolic

## Table of Contents

- [Overview](#overview)
- [Features](#features) 
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Results & Visualization](#results--visualization)
- [MLflow Integration](#mlflow-integration)
- [Databricks Integration](#databricks-integration)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete machine learning pipeline for heart disease prediction, comparing the performance of multiple algorithms including Logistic Regression, XGBoost, Support Vector Machines, and Neural Networks. The project emphasizes:

- **Modular Design**: Clean, reusable code architecture
- **Comprehensive Analysis**: Extensive data exploration and feature engineering
- **Multiple Models**: Implementation of 4 different ML algorithms
- **Experiment Tracking**: Integration with MLflow for model versioning
- **Rich Visualizations**: Detailed plots and model interpretability
- **Reproducibility**: Fixed random seeds and configuration management

## Features

-  **Modular Architecture**: Separated concerns across data loading, feature engineering, modeling, and visualization
-  **Comprehensive EDA**: Detailed exploratory data analysis with statistical insights
-  **Multiple ML Models**: Logistic Regression, XGBoost, SVM, and Neural Networks
-  **Advanced Visualizations**: Feature importance, confusion matrices, ROC curves, and more
-  **Experiment Tracking**: MLflow integration for model versioning and comparison
-  **Configurable Pipeline**: Centralized configuration management
-  **Cross-Validation**: Stratified k-fold cross-validation for robust evaluation
-  **GPU Support**: Automatic device detection for neural network training

##  Project Structure
- **cardiac-patients/**
  - **data/**
    - `heart.csv`
  - **src/**
    - **data/** — `data_loader.py`
    - **features/** — `feature_engineer.py`
    - **models/**
      - `logistic_regression_model.py`
      - `svm_model.py`
      - `xgboost_model.py`
      - `neural_net_model.py`
    - **utils/** — `mlflow_utils.py`
    - **visualization/** — `visualizer.py`
  - `main.ipynb`
  - `requirements.txt`








##  Dataset

The project uses the Heart Disease dataset with **918 patient records** and **11 clinical features**:

| Feature | Type | Description |
|---------|------|-------------|
| `Age` | Numeric | Patient age in years |
| `Sex` | Categorical | Gender (M/F) |
| `ChestPainType` | Categorical | Type of chest pain (ATA, NAP, ASY, TA) |
| `RestingBP` | Numeric | Resting blood pressure |
| `Cholesterol` | Numeric | Serum cholesterol level |
| `FastingBS` | Binary | Fasting blood sugar > 120 mg/dl |
| `RestingECG` | Categorical | Resting ECG results |
| `MaxHR` | Numeric | Maximum heart rate achieved |
| `ExerciseAngina` | Binary | Exercise-induced angina |
| `Oldpeak` | Numeric | ST depression induced by exercise |
| `ST_Slope` | Categorical | Slope of peak exercise ST segment |
| `HeartDisease` | Binary | **Target variable** (0: No disease, 1: Disease) |

##  Models Implemented

### 1. **Logistic Regression**
- Linear classification with L2 regularization
- Feature scaling with StandardScaler
- Probability calibration for reliable predictions

### 2. **XGBoost**
- Gradient boosting with tree-based learners
- Hyperparameter tuning with GridSearchCV
- Feature importance analysis

### 3. **Support Vector Machine (SVM)**
- RBF kernel for non-linear classification
- Grid search for optimal C and gamma parameters
- Robust to outliers and high-dimensional data

### 4. **Neural Network**
- Multi-layer perceptron with dropout regularization
- Early stopping to prevent overfitting
- PyTorch implementation with GPU support
- Batch normalization and adaptive learning rate

##  Installation

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Setup

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/cardiac-patients.git
cd cardiac-patients
```

2. **Create virtual environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter:**
```bash
jupyter notebook main.ipynb
```

##  Usage

### Quick Start

1. **Open the main notebook:**
```bash
jupyter notebook main.ipynb
```

2. **Run all cells** to execute the complete pipeline:
   - Data loading and exploration
   - Feature engineering
   - Model training and evaluation
   - Visualization and results

### Programmatic Usage

```python
from src.utils.config import Config
from src.data.data_loader import DataLoader
from src.models import XGBoostModel

# Initialize configuration
config = Config()

# Load data
data_loader = DataLoader(config)
X_train, X_test, y_train, y_test = data_loader.get_train_test_split()

# Train model
model = XGBoostModel(config)
model.fit(X_train, y_train)

# Evaluate
accuracy = model.score(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Configuration

The project uses a centralized configuration system in `src/utils/config.py`:

```python
@dataclass
class Config:
    # Data settings
    data_path: str = "data/heart.csv"
    target_column: str = "HeartDisease"
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Model settings
    cv_folds: int = 5
    random_seed: int = 42
    
    # Neural Network settings
    hidden_size: int = 64
    dropout_rate: float = 0.3
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
```

### Key Parameters:
- **Data Split**: 60% Train, 20% Validation, 20% Test
- **Cross-Validation**: 5-fold stratified CV
- **Random Seed**: 42 (for reproducibility)
- **Device**: Automatic GPU/CPU detection

##  Results & Visualization

The project generates comprehensive visualizations:

### Data Exploration
- Distribution plots for all features
- Correlation heatmaps
- Target variable analysis
- Statistical summaries

### Model Performance
- Confusion matrices for all models
- ROC curves and AUC scores
- Precision-Recall curves
- Feature importance plots
- Learning curves for neural networks

### Model Comparison
- Cross-validation score comparisons
- Performance metrics table
- Hyperparameter sensitivity analysis

##  MLflow Integration

The project includes comprehensive MLflow integration for experiment tracking:

- **Experiment Logging**: Automatic logging of parameters, metrics, and artifacts
- **Model Versioning**: Track different model versions and configurations
- **Artifact Storage**: Save trained models, plots, and evaluation results
- **Experiment Comparison**: Compare different runs and hyperparameters

### MLflow UI
```bash
mlflow ui
```
Navigate to `http://localhost:5000` to view experiments.

## Databricks Integration

This project includes comprehensive **Databricks MLflow** integration for enterprise-scale experiment tracking and model management. The integration provides seamless connectivity between local development and Databricks cloud environments.

### Key Features

- **Secure Authentication**: Multiple authentication methods (environment variables, tokens, CLI profiles)
- **Automatic Fallback**: Graceful fallback from Databricks to local MLflow if connection fails
- **User-Aware Experiments**: Intelligent experiment path generation based on user context
- **Direct Web Access**: Automatic generation of Databricks workspace URLs
- **Connection Testing**: Built-in validation and health checks

### Setup & Configuration

#### Environment Variables
Set your Databricks credentials as environment variables:

```bash
export DATABRICKS_HOST="https://your-workspace.cloud.databricks.com"
export DATABRICKS_TOKEN="dapi1234567890abcdef..."
```

#### Programmatic Usage
```python
from src.utils.databricks_config import DatabricksMLflowConfig

# Initialize Databricks configuration
config = DatabricksMLflowConfig()

# Setup connection
if config.setup_databricks_mlflow():
    print("Connected to Databricks MLflow!")
    
    # Create experiment
    experiment_path = config.create_experiment("heart_disease_ml")
    print(f"Experiment: {experiment_path}")
    
    # Get web URL
    url = config.get_experiment_url("heart_disease_ml")
    print(f"View at: {url}")
else:
    print("Databricks connection failed - using local MLflow")
```

### Integration Architecture

Local Development → Databricks MLflow → Databricks Workspace
↓ ↓ ↓
main.ipynb → Experiment Tracking → Web UI Access
↓ ↓ ↓
Model Training → Metrics & Artifacts → Collaboration





### Experiment Organization

The system automatically organizes experiments using intelligent path generation:

- **User Workspace**: `/Users/{username}/heart_disease_ml`
- **Shared Workspace**: `/Shared/mlflow_experiments/heart_disease_ml`
- **Fallback Paths**: Multiple fallback strategies for different permission scenarios

### Neural Network Integration

The project's neural network model (`neural_net_mlflow_model.py`) includes built-in Databricks support:

```python
# Enable Databricks tracking
model = MLflowNeuralNetModel(config, use_databricks=True)

# Train with automatic Databricks logging
results = model.fit(X_train, y_train, X_val, y_val)
print(f"Using Databricks: {results['used_databricks']}")
```

### Benefits

- **Enterprise Scale**: Leverage Databricks' managed MLflow for team collaboration
- **Centralized Tracking**: All experiments stored in a single, accessible location  
- **Security**: Enterprise-grade security and access controls
- **Performance**: High-performance infrastructure for large-scale experiments
- **Collaboration**: Easy sharing and comparison of experiments across team members

### Fallback Strategy

If Databricks connection fails, the system automatically falls back to local MLflow:

1. **Primary**: Attempt Databricks connection
2. **Validation**: Test connection with API call
3. **Fallback**: Switch to local MLflow if Databricks unavailable
4. **Logging**: Comprehensive logging of connection status and fallback reasons

This ensures your ML pipeline runs smoothly regardless of network connectivity or Databricks availability.

##  Technologies Used

### Core ML/Data Science
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms and utilities
- **xgboost**: Gradient boosting framework
- **pytorch**: Deep learning framework

### Visualization
- **matplotlib**: Base plotting library
- **seaborn**: Statistical data visualization
- **plotly**: Interactive visualizations (if applicable)

### Experiment Tracking
- **mlflow**: ML lifecycle management
- **mlflow-skinny**: Lightweight MLflow client

### Development
- **jupyter**: Interactive development environment
- **python-dotenv**: Environment variable management

### Optional Integrations
- **databricks-sdk**: Databricks platform integration
- **lightgbm**: Alternative gradient boosting (if used)

##  Workflow

1. **Data Loading**: Load and validate the heart disease dataset
2. **Exploratory Analysis**: Comprehensive statistical and visual analysis
3. **Feature Engineering**: Handle categorical variables, scaling, and selection
4. **Model Training**: Train multiple ML models with cross-validation
5. **Evaluation**: Comprehensive performance evaluation and comparison
6. **Visualization**: Generate plots for model interpretation
7. **Experiment Tracking**: Log results to MLflow for reproducibility

## Performance Metrics

All models are evaluated using:
- **Accuracy**: Overall prediction accuracy
- **Precision**: True positive rate for each class
- **Recall**: Sensitivity for detecting heart disease
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Cross-Validation**: 5-fold stratified validation scores


## Contact

**Murat Kolic**
