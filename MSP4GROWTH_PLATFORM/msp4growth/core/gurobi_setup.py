"""
Gurobi environment setup module.
Handles initialization and configuration of the Gurobi environment.
"""
import gurobipy as gp
import logging
from typing import Optional
from ..config import get_gurobi_params

logger = logging.getLogger(__name__)

class GurobiEnvironment:
    """
    Singleton class to manage Gurobi environment
    """
    _instance = None
    _env = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(GurobiEnvironment, cls).__new__(cls)
            cls._instance._initialize_env()
        return cls._instance
    
    def _initialize_env(self):
        """Initialize the Gurobi environment with appropriate parameters"""
        try:
            logger.info("Initializing Gurobi environment")
            self._env = gp.Env(empty=True)
            
            # Set parameters from configuration
            params = get_gurobi_params()
            for param_name, param_value in params.items():
                if param_value:  # Only set if value is not empty
                    self._env.setParam(param_name, param_value)
            
            # Start the environment
            self._env.start()
            
            # Monkey-patch gp.Model to use our environment by default
            self._patch_model_class()
            
            logger.info("Gurobi environment initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gurobi environment: {str(e)}")
            raise
    
    def _patch_model_class(self):
        """
        Monkey-patch gp.Model to use our environment by default,
        but allow explicit overrides
        """
        _original_model = gp.Model
        
        def _patched_model(*args, **kwargs):
            if "env" not in kwargs:
                kwargs["env"] = self._env
            return _original_model(*args, **kwargs)
        
        gp.Model = _patched_model
        logger.debug("Monkey-patched gp.Model to use our environment by default")
    
    @property
    def env(self):
        """Get the Gurobi environment"""
        return self._env
    
    def __enter__(self):
        """Support for context manager usage"""
        return self._env
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources when exiting context"""
        # We don't dispose the environment here since it's meant to be reused
        pass


def initialize_gurobi() -> gp.Env:
    """
    Initialize the Gurobi environment and return it.
    This function ensures the environment is setup correctly.
    """
    gurobi_env = GurobiEnvironment()
    return gurobi_env.env
