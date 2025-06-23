"""
FastAPI application entry point for the MSP4GROWTH platform.
"""
from fastapi import FastAPI, Body, HTTPException, File, UploadFile, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import os
import sys
import json
import shutil
from shapely.geometry import Polygon
import uuid
from datetime import datetime, timedelta
from pathlib import Path
import traceback
# GeoPandas for handling GeoJSON files
import geopandas as gpd
# import box from shapely.geometry
from shapely.geometry import box
# Add session management
from starlette.middleware.sessions import SessionMiddleware

from msp4growth.config import RESULTS_DIR, LOGS_DIR, DATA_DIR
# fall back to /app for Docker, but use APP_ROOT locally
BASE_DIR = Path(os.getenv("APP_ROOT", "/app"))

# Create a temporary directory for file uploads
TEMP_DIR = Path(os.getenv("APP_ROOT", str("TEMP_DIR/"  "temp")))
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# Add the parent directory to sys.path to ensure imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import core modules, but handle it gracefully if they're not available yet
# try:
from msp4growth.core.data_loader import MSP4GROWTHAnalysis
CORE_MODULES_AVAILABLE = True
# except ImportError:
#     print("Warning: Core modules not available. API will run in mock mode.")
#     CORE_MODULES_AVAILABLE = False

# Request models
class ThreadConfig(BaseModel):
    threads: int = 12
    omp_num_threads: int = 12
    mkl_num_threads: int = 12
    openblas_threads: int = 12

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

# File metadata model
class FileMetadata(BaseModel):
    filename: str
    original_filename: str
    size: int
    upload_time: datetime
    file_path: str
    content_type: str

# This is a simple in-memory store for file metadata
# In a production app, you'd use Redis or a database
session_files = {}

# Session management
class SessionManager:
    def __init__(self, request: Request):
        self.request = request
    
    def get_session_id(self):
        """Get or create a session ID"""
        if "session_id" not in self.request.session:
            self.request.session["session_id"] = str(uuid.uuid4())
        return self.request.session["session_id"]

# Dependency for getting session manager
def get_session_manager(request: Request):
    return SessionManager(request)

# Create FastAPI application
app = FastAPI(
    title="MSP4GROWTH API",
    description="API for marine spatial planning optimization",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add SessionMiddleware
app.add_middleware(
    SessionMiddleware,
    secret_key="msp4growth-secret-key"  # Replace with a secure key in production
)

# File cleanup task
def cleanup_old_files():
    """Remove files older than 24 hours"""
    threshold = datetime.now() - timedelta(hours=24)
    
    for session_id, files in list(session_files.items()):
        for file_id, metadata in list(files.items()):
            if metadata.upload_time < threshold:
                try:
                    file_path = Path(metadata.file_path)
                    if file_path.exists():
                        file_path.unlink()
                    del session_files[session_id][file_id]
                except Exception as e:
                    logging.error(f"Error cleaning up file {file_id}: {e}")
        
        # Remove session if empty
        if not session_files[session_id]:
            del session_files[session_id]

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to MSP4GROWTH API",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/api/v1/health")
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    # ensure our local dirs exist
    for path in (RESULTS_DIR, LOGS_DIR, DATA_DIR, TEMP_DIR):
        path.mkdir(parents=True, exist_ok=True)
    
    print("MSP4GROWTH API started successfully!")

# The original endpoint that test_request.py is likely trying to access
@app.post("/run")
async def run_analysis_legacy(request: Dict = Body(...)):
    """
    Legacy endpoint for running the analysis
    """
    return await run_analysis_implementation(request)

# The new API versioned endpoint
@app.post("/api/v1/run")
async def run_analysis(request: MSPConfig):
    """
    Run MSP4GROWTH analysis with the provided configuration
    """
    return await run_analysis_implementation(request.dict())
async def load_server_file_content(file_id: str, session_id: str):
    """
    Load the content of a server file by ID and session
    """
    if session_id not in session_files or file_id not in session_files[session_id]:
        logging.error(f"File {file_id} not found in session {session_id}")
        return None
    
    metadata = session_files[session_id][file_id]
    try:
        # Load the file as GeoJSON
        gdf = gpd.read_file(metadata.file_path)
        return gdf
    except Exception as e:
        logging.error(f"Error loading file {file_id}: {e}")
        return None

async def run_analysis_implementation(config: Dict):
    """
    Implementation of the analysis logic, shared by both endpoints
    """
    print(f"Received analysis request: {config['request_id']}")
    
    # if not CORE_MODULES_AVAILABLE:
    #     # Return mock response in development mode
    #     mock_results = {
    #         "model_AUG": f"/app/results/{config['request_id']}/AUG_results.geojson",
    #         "model_WS": f"/app/results/{config['request_id']}/WS_results_0.5.geojson",
    #         "model_PF": f"/app/results/{config['request_id']}/PF_results.geojson"
    #     }
    #     return mock_results
    enabled_thresholds = {
        key: value for key, value in config["thresholds"].items() 
        if value.get("enabled", True) is True
    }
    config["thresholds"] = enabled_thresholds
    # Handle server files in thresholds
    # Where you process server files in thresholds
    for key, value in config["thresholds"].items():
        logging.info(f"Processing threshold {key}: {value.get('source')}")
        
        if value.get("source") == "server":
            file_id = value.get("fileId")
            logging.info(f"File ID: {file_id}")

            gdf = gpd.read_file(f"TEMP_DIR/temp/{file_id}.geojson")
                    
            # Add to datasets if not already there
            if not config.get("datasets"):
                config["datasets"] = {}
            
            # Convert GeoDataFrame to a format your analyzer can use
            logging.info(f"Adding file {file_id} to datasets with key {key}")
            config["datasets"][key] = {}
            config["datasets"][key]["geom"] = json.loads(gdf.to_json())
            
    # make sure the input polygon is EPSG:3857 if not conver it
    if config["input_polygon"] and len(config["input_polygon"]) > 0:
        # Make sure the polygon is closed (first and last points are the same)
        coords =   config["input_polygon"]
        logging.info(f"Input polygon coordinates: {coords}")
        if coords[0] != coords[-1]:
            # Add the first point at the end to close the polygon
            coords = coords + [coords[0]]
        
        # Create the polygon from the coordinates
        polygon = Polygon(coords)
        
        input_polygon_gdf = gpd.GeoDataFrame(
            geometry=[polygon],
            crs="EPSG:4326"  # Assuming input is in WGS84
        )
        
        # Reproject to EPSG:3857
        input_polygon_gdf = input_polygon_gdf.to_crs(epsg=3857)
        
        # Extract the coordinates of the reprojected polygon
        reprojected_polygon = input_polygon_gdf.geometry[0]
        reprojected_coords = list(reprojected_polygon.exterior.coords)
        
        # Convert to the desired format
        formatted_coords = [list(coord) for coord in reprojected_coords]
        
        # Update the config with the formatted coordinates
        config["input_polygon"] = formatted_coords
        logging.info(f"Reprojected input polygon coordinates: {formatted_coords}")
    # save the request to a file for debugging
    try:
        with open("./last_request.json", "w") as f:
            json.dump(config, f, indent=2)
    except Exception as e:
        logging.error(f"Error saving request to file: {e}")
    try:
        # Create analysis instance
        analyzer = MSP4GROWTHAnalysis(
        request_id=config["request_id"],
        input_polygon=config["input_polygon"],
        use_case=config["use_case"],
        model=config["model"],
        C_number=config["C_number"],
        N_size=config["N_size"],
        datasets=config.get("datasets"),
        thresholds=config["thresholds"],  # This now contains only enabled thresholds
        calc_type=config.get("calc_type", "dist")
    )
        # Filter to only include enabled datasets
        enabled_thresholds = {
            key: value for key, value in config["thresholds"].items()
            if value.get("enabled", True) is True
        }
        config["thresholds"] = enabled_thresholds
        # Run analysis
        result_paths = analyzer.run_analysis()
        print(f"Analysis completed. Results saved to: {result_paths}")
        
        # Read GeoJSON content and return it
        geojson_results = {}
        for name, path in result_paths.items():
            if os.path.exists(path):
                    # Load the GeoJSON file as a GeoDataFrame
                    gdf = gpd.read_file(path)
                    
                    # Transform to EPSG:4326 (WGS84 - standard lat/lon)
                    gdf = gdf.to_crs(epsg=4326)
                    
                    # Convert back to GeoJSON dict
                    geojson_results[name] = json.loads(gdf.to_json())
                
            else:
                geojson_results[name] = f"File not found: {str(path)}"
        
        return geojson_results
        
    except Exception as e:
        # print the stack trace for debugging
        # Get the full stack trace as a string
        stack_trace = traceback.format_exc()
        # Log the error message and stack trace
        logging.error(f"Error during analysis: {str(e)}\n{stack_trace}")
        
        raise HTTPException(status_code=500, detail=str(e))

# Save a request to file for debugging
@app.post("/debug/save-request")
async def save_request(request: Dict = Body(...)):
    """Save a request to a file for debugging"""
    try:
        with open("/app/logs/last_request.json", "w") as f:
            json.dump(request, f, indent=2)
        return {"status": "saved"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# =================================================================
# File Upload and Management Endpoints
# =================================================================

@app.post("/api/v1/files/upload")
async def upload_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    Upload a file for temporary storage during the user's session
    """
    try:
        # Get session ID
        session_id = session_manager.get_session_id()
        
        # Generate a unique filename to avoid collisions
        file_id = str(uuid.uuid4())
        file_extension = os.path.splitext(file.filename)[1]
        safe_filename = f"{file_id}{file_extension}"
        
        # Create file path
        file_path = TEMP_DIR / safe_filename
        
        # Save the file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Store metadata
        metadata = FileMetadata(
            filename=safe_filename,
            original_filename=file.filename,
            size=os.path.getsize(file_path),
            upload_time=datetime.now(),
            file_path=str(file_path),
            content_type=file.content_type or "application/octet-stream"
        )
        
        # Ensure the session exists in our dictionary
        if session_id not in session_files:
            session_files[session_id] = {}
            
        # Store file metadata in session
        session_files[session_id][file_id] = metadata
        
        # Schedule cleanup
        background_tasks.add_task(cleanup_old_files)
        
        # Return file info to client
        return {
            "file_id": file_id,
            "filename": file.filename,
            "size": metadata.size,
            "upload_time": metadata.upload_time,
        }
    
    except Exception as e:
        logging.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/files")
async def list_files(
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    List all files uploaded in the current session
    """
    session_id = session_manager.get_session_id()
    files = session_files.get(session_id, {})
    print(f"Listing files for session {session_id}: {files}")
    # Convert to list of simplified file info
    file_list = [
        {
            "file_id": file_id,
            "filename": metadata.original_filename,
            "size": metadata.size,
            "upload_time": metadata.upload_time
        }
        for file_id, metadata in files.items()
    ]
    
    return {"files": file_list}

@app.get("/api/v1/files/{file_id}")
async def get_file(
    file_id: str,
    bounds: Optional[str] = None,
    zoom: Optional[float] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Get a file by ID from the current session with optional bounds and zoom level for optimization
    
    - bounds: Optional comma-separated string in format "west,south,east,north"
    - zoom: Optional zoom level (higher means more detail)
    """
    session_id = session_manager.get_session_id()
    if session_id not in session_files or file_id not in session_files[session_id]:
        raise HTTPException(status_code=404, detail="File not found")
    
    metadata = session_files[session_id][file_id]
    print(f"Retrieving file {file_id} for session {session_id}")
    try:
        # Parse bounds if provided
        bounds_polygon = None
        if bounds:
            try:
                west, south, east, north = map(float, bounds.split(','))
                if west and south and east and north:
                    # Create a box polygon from the bounds
                    bounds_polygon = box(west, south, east, north)
            except (ValueError, TypeError) as e:
                logging.warning(f"Invalid bounds parameter: {bounds}. Error: {e}")
        
        # Convert the file to a GeoDataFrame
        gdf = gpd.read_file(metadata.file_path)
        gdf = gdf.to_crs(epsg=4326)  # Convert to WGS84 if needed
        gdf.set_crs(epsg=4326, inplace=True)  # Set CRS to WGS84
        
        # Apply bounds filtering if bounds were provided
        if bounds_polygon:
            # Filter features to only include those that intersect with the bounds
            gdf = gdf[gdf.intersects(bounds_polygon)]
        
        original_count = len(gdf)
        print(f"Original feature count: {original_count}, Zoom level: {zoom}")
        
        # For grid-like data, use systematic sampling instead of simplification
        if zoom is not None:
            try:
                # Get the centroids of all polygons to help with grid sampling
                gdf['centroid'] = gdf.geometry.centroid
                
                # Different sampling strategies based on zoom level
                if zoom >= 13:  # Very high zoom (closest) - full dataset
                    pass  # Use full dataset
                elif zoom >= 10:  # High zoom
                    if original_count > 2000:
                        # For grid data, sample systematically by taking every Nth cell
                        n = max(2, original_count // 2000)
                        gdf = gdf.iloc[::n].copy()
                elif zoom >= 8:  # Medium zoom
                    if original_count > 1000:
                        n = max(3, original_count // 1000)
                        gdf = gdf.iloc[::n].copy()
                elif zoom >= 7:  # Low zoom
                    if original_count > 500:
                        n = max(5, original_count // 500)
                        gdf = gdf.iloc[::n].copy()
                else:  # Very low zoom (farthest)
                    if original_count > 200:
                        n = max(10, original_count // 200)
                        gdf = gdf.iloc[::n].copy()
                    
                    # For extremely low zoom levels, consider aggregating cells with the same value
                    if zoom < 5 and "raster_value" in gdf.columns:
                        # Group by raster_value and dissolve geometries
                        gdf = gdf.dissolve(by="raster_value", aggfunc="first").reset_index()
                        
                        # Apply stronger simplification to the dissolved polygons
                        tolerance = 0.05
                        gdf.geometry = gdf.geometry.simplify(tolerance, preserve_topology=True)
                
                # Clean up
                if 'centroid' in gdf.columns:
                    gdf = gdf.drop(columns=['centroid'])
                
                print(f"Optimized feature count: {len(gdf)}, Reduction: {(original_count - len(gdf)) / original_count * 100:.1f}%")
                
            except Exception as e:
                logging.warning(f"Error applying zoom-based optimization: {e}")
                print(f"Optimization error: {e}")
        
        # Return the GeoDataFrame as a GeoJSON
        return Response(
            content=gdf.to_json(),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename={metadata.original_filename}"}
        )
    except Exception as e:
        logging.error(f"Error processing file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")
@app.delete("/api/v1/files/{file_id}")
async def delete_file(
    file_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Delete a file by ID from the current session
    """
    session_id = session_manager.get_session_id()
    if session_id not in session_files or file_id not in session_files[session_id]:
        raise HTTPException(status_code=404, detail="File not found")
    
    metadata = session_files[session_id][file_id]
    
    try:
        # Delete the file
        file_path = Path(metadata.file_path)
        if file_path.exists():
            file_path.unlink()
        
        # Remove from session
        del session_files[session_id][file_id]
        
        return {"message": "File deleted successfully"}
    
    except Exception as e:
        logging.error(f"Error deleting file {file_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/geojson/upload")
async def upload_geojson(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    session_manager: SessionManager = Depends(get_session_manager),
):
    """
    Upload a GeoJSON file and validate it
    """
    if not file.filename.endswith(('.geojson', '.json')):
        raise HTTPException(status_code=400, detail="File must be a GeoJSON file (.geojson or .json)")
    
    try:
        # Get the file content
        content = await file.read()
        
        # Basic validation - try to parse as JSON
        try:
            geojson_data = json.loads(content)
            
            # Validate GeoJSON structure
            if "type" not in geojson_data:
                raise HTTPException(status_code=400, detail="Invalid GeoJSON: missing 'type' property")
            
            # Check if it's a valid GeoJSON type
            valid_types = ['FeatureCollection', 'Feature', 'Point', 'LineString', 'Polygon', 
                          'MultiPoint', 'MultiLineString', 'MultiPolygon', 'GeometryCollection']
            
            if geojson_data["type"] not in valid_types:
                raise HTTPException(status_code=400, detail=f"Invalid GeoJSON type: {geojson_data['type']}")
            
            # Further validation for FeatureCollection
            if geojson_data["type"] == "FeatureCollection" and not isinstance(geojson_data.get("features"), list):
                raise HTTPException(status_code=400, detail="Invalid FeatureCollection: missing features array")
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format")
        
        # Reset file pointer for saving
        await file.seek(0)
        
        # Upload the file using the general upload endpoint
        upload_result = await upload_file(background_tasks, file, session_manager)
        
        # Count features
        feature_count = 0
        if geojson_data["type"] == "FeatureCollection":
            feature_count = len(geojson_data.get("features", []))
        elif geojson_data["type"] == "Feature":
            feature_count = 1
        elif geojson_data["type"] == "GeometryCollection":
            feature_count = len(geojson_data.get("geometries", []))
        else:
            feature_count = 1
        
        # Add GeoJSON-specific info
        upload_result["feature_count"] = feature_count
        upload_result["geojson_type"] = geojson_data["type"]
        
        return upload_result
        
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error processing GeoJSON file: {e}")
        raise HTTPException(status_code=500, detail=str(e))