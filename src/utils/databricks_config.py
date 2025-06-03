"""Databricks MLflow configuration and utilities."""

import os
import mlflow
from typing import Optional
import logging

class DatabricksMLflowConfig:
    """Configuration for Databricks-managed MLflow."""
    
    def __init__(self, 
                 databricks_host: Optional[str] = None,
                 databricks_token: Optional[str] = None,
                 profile: str = "DEFAULT"):
        """
        Initialize Databricks MLflow configuration.
        
        Args:
            databricks_host: Databricks workspace URL
            databricks_token: Personal access token
            profile: Databricks CLI profile name
        """
        self.databricks_host = databricks_host or os.getenv('DATABRICKS_HOST')
        self.databricks_token = databricks_token or os.getenv('DATABRICKS_TOKEN')
        self.profile = profile
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_databricks_mlflow(self) -> bool:
        """Set up MLflow to use Databricks backend."""
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
        """Convert simple experiment name to Databricks-compatible path."""
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
        """Create or get experiment with proper Databricks path."""
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
        """Get Databricks experiment URL."""
        try:
            experiment_path = self._get_databricks_experiment_path(experiment_name)
            experiment = mlflow.get_experiment_by_name(experiment_path)
            if experiment and self.databricks_host:
                return f"{self.databricks_host}/#mlflow/experiments/{experiment.experiment_id}"
            return "Experiment URL not available"
        except:
            return "Experiment URL not available"

