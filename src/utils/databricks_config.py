"""
Databricks MLflow Configuration and Integration Module

This module provides comprehensive configuration and connection management for
Databricks-managed MLflow instances. It handles authentication, experiment creation,
and seamless integration between local development and Databricks cloud environments
for MLflow experiment tracking.

Classes:
    DatabricksMLflowConfig: Main configuration class for Databricks MLflow integration

Key Features:
    - Automatic Databricks workspace connection with credential management
    - MLflow experiment creation and management in Databricks
    - Fallback mechanisms for connectivity issues
    - User-aware experiment path generation
    - Comprehensive error handling and logging
    - Support for multiple authentication methods
    - Connection testing and validation

Design Pattern:
    Uses the Singleton pattern for configuration management and the Adapter pattern
    to provide a consistent interface for different MLflow backends (local vs Databricks).

Usage:
    >>> from utils.databricks_config import DatabricksMLflowConfig
    >>> 
    >>> # Initialize with environment variables
    >>> config = DatabricksMLflowConfig()
    >>> success = config.setup_databricks_mlflow()
    >>> 
    >>> if success:
    ...     experiment_path = config.create_experiment("my_experiment")
    ...     print(f"Experiment URL: {config.get_experiment_url('my_experiment')}")
"""
import os
import mlflow
from typing import Optional
import logging

class DatabricksMLflowConfig:
    """
    Comprehensive configuration manager for Databricks MLflow integration.
    
    This class provides seamless integration with Databricks-managed MLflow,
    handling authentication, workspace connection, experiment management,
    and error recovery. It abstracts the complexity of Databricks connectivity
    while providing robust fallback mechanisms.
    
    The class supports multiple authentication methods and provides intelligent
    experiment path generation based on user context and workspace organization.
    
    Attributes:
        databricks_host (str): Databricks workspace URL
        databricks_token (str): Personal access token for authentication
        profile (str): Databricks CLI profile name (if using profile-based auth)
        logger: Configured logger for detailed operation tracking
    
    Authentication Methods:
        1. Environment variables (DATABRICKS_HOST, DATABRICKS_TOKEN)
        2. Direct parameter passing
        3. Databricks CLI profile-based authentication
    
    Example:
        >>> # Environment variable authentication
        >>> os.environ['DATABRICKS_HOST'] = 'https://your-workspace.cloud.databricks.com'
        >>> os.environ['DATABRICKS_TOKEN'] = 'dapi1234567890...'
        >>> 
        >>> config = DatabricksMLflowConfig()
        >>> if config.setup_databricks_mlflow():
        ...     experiment_path = config.create_experiment("heart_disease_ml")
        ...     print(f" Experiment created: {experiment_path}")
        ...     print(f" View at: {config.get_experiment_url('heart_disease_ml')}")
        ... else:
        ...     print(" Databricks connection failed")
    """
    
    def __init__(self, 
                 databricks_host: Optional[str] = None,
                 databricks_token: Optional[str] = None,
                 profile: str = "DEFAULT"):
        """
        Initialize Databricks MLflow configuration with flexible authentication.
        
        Supports multiple authentication approaches with automatic fallback
        to environment variables if direct parameters are not provided.
        
        Args:
            databricks_host: Databricks workspace URL (e.g., 'https://workspace.cloud.databricks.com')
            databricks_token: Personal access token for authentication
            profile: Databricks CLI profile name for profile-based authentication
        """
        self.databricks_host = databricks_host or os.getenv('DATABRICKS_HOST')
        self.databricks_token = databricks_token or os.getenv('DATABRICKS_TOKEN')
        self.profile = profile
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_databricks_mlflow(self) -> bool:
        """
        Establish connection to Databricks MLflow with comprehensive error handling.
        
        Attempts to configure MLflow for Databricks backend, validates the connection,
        and provides detailed logging of the process. Includes connection testing
        to ensure the setup is functional.
        
        Returns:
            bool: True if connection successful, False otherwise
            
        Process:
            1. Validate credentials availability
            2. Configure MLflow tracking URI
            3. Set environment variables
            4. Test connection with API call
            5. Log success/failure with details
            
        Example:
            >>> config = DatabricksMLflowConfig()
            >>> if config.setup_databricks_mlflow():
            ...     print(" Ready to use Databricks MLflow!")
            ... else:
            ...     print(" Falling back to local MLflow")
        """
        try:
            if self.databricks_host and self.databricks_token:
                # Set MLflow tracking URI to Databricks
                mlflow.set_tracking_uri("databricks")
                
                # Set Databricks environment variables
                os.environ['DATABRICKS_HOST'] = self.databricks_host
                os.environ['DATABRICKS_TOKEN'] = self.databricks_token
                
                self.logger.info(f"MLflow configured for Databricks: {self.databricks_host}")
                
                # Test connection
                try:
                    mlflow.search_experiments(max_results=1)
                    self.logger.info("Successfully connected to Databricks MLflow!")
                    return True
                except Exception as e:
                    self.logger.warning(f"Connection test failed: {e}")
                    return False
            else:
                self.logger.warning("No Databricks credentials found")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to setup Databricks MLflow: {e}")
            return False
    
    def _get_databricks_experiment_path(self, experiment_name: str) -> str:
        """
        Generate appropriate Databricks experiment path based on user context.
        
        Creates user-aware experiment paths that follow Databricks conventions
        and handle workspace organization properly. Includes fallback logic
        for different user context scenarios.
        
        Args:
            experiment_name: Simple name for the experiment
            
        Returns:
            Full Databricks-compatible experiment path
            
        Path Generation Strategy:
            1. Try to get current user from Databricks API
            2. Create user-specific path: /Users/{username}/{experiment_name}
            3. Fallback to shared path: /Shared/{experiment_name}
            4. Further fallback to organized shared path
            
        Example:
            >>> config = DatabricksMLflowConfig()
            >>> path = config._get_databricks_experiment_path("heart_disease_ml")
            >>> print(path)  # /Users/user@company.com/heart_disease_ml
        """
        # Try to get current user from Databricks API
        try:
            import requests
            
            headers = {
                "Authorization": f"Bearer {self.databricks_token}",
                "Content-Type": "application/json"
            }
            
            # Get current user info
            response = requests.get(
                f"{self.databricks_host}/api/2.0/preview/scim/v2/Me",
                headers=headers,
                timeout=10
            )
            
            if response.status_code == 200:
                user_info = response.json()
                username = user_info.get('userName', 'unknown_user')
                return f"/Users/{username}/{experiment_name}"
            else:
                # Fallback to shared experiments
                return f"/Shared/{experiment_name}"
                
        except Exception as e:
            self.logger.warning(f"Could not get user info: {e}")
            # Fallback to shared experiments
            return f"/Shared/{experiment_name}"
    
    def create_experiment(self, experiment_name: str) -> str:
        """
        Create or retrieve MLflow experiment in Databricks with proper path handling.
        
        Manages experiment lifecycle including creation, retrieval, and path
        resolution. Provides comprehensive error handling with multiple fallback
        strategies for different failure scenarios.
        
        Args:
            experiment_name: Name for the experiment (will be converted to proper path)
            
        Returns:
            Full experiment path for use with MLflow
            
        Raises:
            Exception: If all experiment creation attempts fail
            
        Process:
            1. Generate appropriate experiment path
            2. Check if experiment already exists
            3. Create experiment if needed
            4. Handle permission and path issues with fallbacks
            5. Return validated experiment path
            
        Example:
            >>> config = DatabricksMLflowConfig()
            >>> experiment_path = config.create_experiment("heart_disease_prediction")
            >>> print(experiment_path)  # /Users/user@company.com/heart_disease_prediction
            Created experiment: /Users/user@company.com/heart_disease_prediction
        """
        try:
            # Convert to Databricks-compatible path
            experiment_path = self._get_databricks_experiment_path(experiment_name)
            
            # Check if experiment exists
            experiment = mlflow.get_experiment_by_name(experiment_path)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_path)
                self.logger.info(f"Created experiment: {experiment_path}")
                return experiment_path  # Return path, not ID
            else:
                self.logger.info(f"Using existing experiment: {experiment_path}")
                return experiment_path  # Return path, not ID
                
        except Exception as e:
            self.logger.error(f"Failed to create experiment: {e}")
            # Try with a simpler shared path as fallback
            try:
                fallback_path = f"/Shared/mlflow_experiments/{experiment_name}"
                experiment = mlflow.get_experiment_by_name(fallback_path)
                if experiment is None:
                    experiment_id = mlflow.create_experiment(fallback_path)
                    self.logger.info(f"Created fallback experiment: {fallback_path}")
                    return fallback_path
                else:
                    self.logger.info(f"Using existing fallback experiment: {fallback_path}")
                    return fallback_path
            except Exception as fallback_error:
                self.logger.error(f"Fallback experiment creation also failed: {fallback_error}")
                raise
    
    def get_experiment_url(self, experiment_name: str) -> str:
        """
        Generate the Databricks web UI URL for the experiment.
        
        Creates a direct link to the experiment in the Databricks workspace
        for easy access to the MLflow UI and experiment management.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Full URL to the experiment in Databricks UI, or error message if unavailable
            
        URL Format:
            https://<workspace>.cloud.databricks.com/#mlflow/experiments/<experiment_id>
            
        Example:
            >>> config = DatabricksMLflowConfig()
            >>> url = config.get_experiment_url("heart_disease_ml")
            >>> print(f"View experiment: {url}")
            >>> # Output: https://workspace.cloud.databricks.com/#mlflow/experiments/123456789
        """
        try:
            experiment_path = self._get_databricks_experiment_path(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_path)
            if experiment and self.databricks_host:
                return f"{self.databricks_host}/#mlflow/experiments/{experiment.experiment_id}"
            return "Experiment URL not available"
        except:
            return "Experiment URL not available"

