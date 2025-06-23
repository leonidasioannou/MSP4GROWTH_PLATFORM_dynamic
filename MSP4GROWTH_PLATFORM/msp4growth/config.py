"""
Configuration module for MSP4GROWTH platform.
Centralizes configuration and handles sensitive information securely.
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import sys

# Base paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"
LOGS_DIR = BASE_DIR / "logs"

# Ensure required directories exist
RESULTS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# Configure logging
log_file = os.path.join(LOGS_DIR, "msp4growth.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Try to add file handler, but don't fail if it can't be created
try:
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)
    print(f"Logging to file: {log_file}")
except (PermissionError, OSError) as e:
    print(f"Warning: Could not create log file ({str(e)}). Logging to stdout only.")

logger = logging.getLogger(__name__)

# Thread configuration
THREAD_CONFIG = {
    "threads": int(os.environ.get("GUROBI_THREADS", 12)),
    "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", 12)),
    "mkl_num_threads": int(os.environ.get("MKL_NUM_THREADS", 12)),
    "openblas_num_threads": int(os.environ.get("OPENBLAS_NUM_THREADS", 12))
}

# Gurobi configuration
GUROBI_CONFIG = {
    "threads": THREAD_CONFIG["threads"],
    "wls_access_id": os.environ.get("GUROBI_WLS_ACCESS_ID", ""),
    "wls_secret": os.environ.get("GUROBI_WLS_SECRET", ""),
    "license_id": os.environ.get("GUROBI_LICENSE_ID", ""),
    "log_file": os.environ.get("GUROBI_LOG_FILE", "gurobi.log")
}

def get_gurobi_params() -> Dict[str, Any]:
    """
    Returns Gurobi parameters as a dictionary, suitable for setting via env.setParam()
    """
    return {
        "LogFile": GUROBI_CONFIG["log_file"],
        "WLSACCESSID": GUROBI_CONFIG["wls_access_id"],
        "WLSSECRET": GUROBI_CONFIG["wls_secret"],
        "LicenseID": GUROBI_CONFIG["license_id"],
        "Threads": GUROBI_CONFIG["threads"]
    }

def apply_thread_config() -> None:
    """
    Apply thread configuration to environment variables
    """
    os.environ["OMP_NUM_THREADS"] = str(THREAD_CONFIG["omp_num_threads"])
    os.environ["MKL_NUM_THREADS"] = str(THREAD_CONFIG["mkl_num_threads"])
    os.environ["OPENBLAS_NUM_THREADS"] = str(THREAD_CONFIG["openblas_num_threads"])
    logger.info(f"Thread configuration applied: {THREAD_CONFIG}")

def initialize_environment() -> None:
    """
    Initialize the application environment
    """
    # Apply thread configuration
    apply_thread_config()
    
    # Ensure results directory is available
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    logger.info("Environment initialized")