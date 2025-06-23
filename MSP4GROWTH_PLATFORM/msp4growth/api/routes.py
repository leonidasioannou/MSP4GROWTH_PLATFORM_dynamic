"""
API routes for the MSP4GROWTH platform.
"""
from fastapi import APIRouter, HTTPException, Depends
import logging
from typing import Dict

from .models import MSPConfig, AnalysisResponse
from ..core.data_loader import MSP4GROWTHAnalysis
from ..config import THREAD_CONFIG
from ..core.gurobi_setup import initialize_gurobi

# Create router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/run", response_model=AnalysisResponse, status_code=201)
async def run_analysis(config: MSPConfig):
    """
    Run MSP4GROWTH analysis with the provided configuration
    """
    # Initialize Gurobi environment
    try:
        env = initialize_gurobi()
    except Exception as e:
        logger.error(f"Failed to initialize Gurobi environment: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Gurobi initialization error: {str(e)}")
    
    # Use provided thread configuration or default
    thread_config = config.threads.dict() if config.threads else THREAD_CONFIG
    
    # Create analysis instance
    analyzer = MSP4GROWTHAnalysis(
        request_id=config.request_id,
        input_polygon=config.input_polygon,
        use_case=config.use_case,
        model=config.model,
        C_number=config.C_number,
        N_size=config.N_size,
        datasets=config.datasets,
        thresholds=config.thresholds,
        calc_type=config.calc_type,
        thread_config=thread_config
    )
    
    # Run analysis
    try:
        logger.info(f"Starting analysis for request ID: {config.request_id}")
        results = analyzer.run_analysis()
        logger.info(f"Analysis completed for request ID: {config.request_id}")
        
        # Convert Path objects to strings for JSON serialization
        results_dict = {name: str(path) for name, path in results.items()}
        
        return AnalysisResponse(
            request_id=config.request_id,
            results=results_dict
        )
        
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"}