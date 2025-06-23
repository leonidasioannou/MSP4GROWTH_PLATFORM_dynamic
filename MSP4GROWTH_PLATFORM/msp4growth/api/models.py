"""
Pydantic models for the MSP4GROWTH API.
"""
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any

class ThreadConfig(BaseModel):
    """Thread configuration for computational resources"""
    threads: int = Field(12, description="Gurobi Threads parameter")
    omp_num_threads: int = Field(12, description="OMP_NUM_THREADS")
    mkl_num_threads: int = Field(12, description="MKL_NUM_THREADS")
    openblas_threads: int = Field(12, description="OPENBLAS_NUM_THREADS")

class DatasetThreshold(BaseModel):
    thresholds: List[float]
    weights: float
    enabled: Optional[bool] = True
    source: Optional[str] = "standard"
    fileId: Optional[str] = None

class MSPConfig(BaseModel):
    request_id: str
    input_polygon: List[List[float]]
    use_case: str
    model: Dict[str, Any]
    C_number: int
    N_size: List[int]
    datasets: Optional[Dict[str, Any]] = None
    thresholds: Dict[str, Any]
    calc_type: Optional[str] = "dist"
    threads: Optional[ThreadConfig] = None
    session_id: Optional[str] = None  # Add this line

class ErrorResponse(BaseModel):
    """Error response model"""
    detail: str

class AnalysisResponse(BaseModel):
    """Response model for successful analysis"""
    request_id: str
    results: Dict[str, str]