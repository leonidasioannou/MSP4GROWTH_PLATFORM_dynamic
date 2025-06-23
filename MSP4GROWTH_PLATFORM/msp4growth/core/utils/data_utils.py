"""
Utility functions for geographic data processing.
"""
import os
import sys
import logging
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import billiard as mp
from pathlib import Path
from functools import partial
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
import json
logger = logging.getLogger(__name__)

def load_dataset(file_path: str, crs: str = 'epsg:3857') -> gpd.GeoDataFrame:
    """
    Load a GeoJSON or shapefile and reproject to the specified CRS.
    
    Args:
        file_path: Path to the GeoJSON or shapefile
        crs: Coordinate reference system to reproject to
        
    Returns:
        GeoDataFrame with the loaded data
        
    Raises:
        FileNotFoundError: If the file does not exist
        Exception: For other loading errors
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
        # Load and reproject
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(crs)
        logger.info(f"Loaded and reprojected '{file_path}' successfully.")
        return gdf
        
    except FileNotFoundError as e:
        logger.error(str(e))
        raise
    except Exception as e:
        logger.error(f"An error occurred while loading '{file_path}': {str(e)}")
        raise

def crop_habitat(gdf: gpd.GeoDataFrame, aoi: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Crop habitat data to area of interest.
    
    Args:
        gdf: GeoDataFrame with habitat data
        aoi: GeoDataFrame with area of interest
        
    Returns:
        Cropped GeoDataFrame
    """
    return gdf[gdf.within(aoi.unary_union)]

def load_gdf(
    gdf_path: str,
    aoi: Optional[Polygon] = None,
    output_folder: Optional[Union[str, Path]] = None
) -> Tuple[gpd.GeoDataFrame, Dict, np.ndarray, np.ndarray]:
    """
    Load a habitat mapping GeoDataFrame and prepare it for analysis.
    
    Args:
        gdf_path: Path to the GeoDataFrame file
        aoi: Area of interest for spatial filtering (EPSG:3857)
        output_folder: Folder to save intermediate results
        
    Returns:
        Tuple containing:
        - GeoDataFrame with habitat data
        - Dictionary mapping centroids to IDs
        - Distance matrix
        - Array of feature IDs
        
    Raises:
        Exception: If AOI has no data or other errors occur
    """
    if output_folder is None:
        output_folder = Path("Results")
    elif isinstance(output_folder, str):
        output_folder = Path(output_folder)
        
    # Ensure output folder exists
    output_folder.mkdir(exist_ok=True, parents=True)
    
    # Convert AOI to GeoDataFrame if provided
    if aoi is not None:
        aoi = gpd.GeoDataFrame(geometry=[aoi], crs='epsg:3857')
    
    # Load habitat data
    gdf = load_dataset(gdf_path)
    
    # Filter by AOI if provided
    if aoi is not None:
        try:
            gdf = crop_habitat(gdf, aoi).reset_index(drop=True)
            if gdf.empty:
                raise Exception("The area of interest you chose has no data applicable for this use case")
        except Exception as e:
            logger.error(str(e))
            raise
    
    logger.info(f"Loaded {len(gdf)} habitat features")
    
    # Extract centroids and create mapping
    centroids = gdf.geometry.centroid
    centroid_to_id = {idx: gdf.at[idx, 'fid'] for idx in gdf.index}
    centroid_points = list(centroids)
    
    # Define index set for cells of interest
    I_plus = gdf.index
    
    # Check for precomputed distance matrix
    dki_file = output_folder / 'dki.npz'
    pairs_file = output_folder / 'pairs.pkl'
    
    if dki_file.exists() and pairs_file.exists():
        # Load precomputed distance matrix
        data = np.load(dki_file, allow_pickle=True)
        dki_matrix = data['dki_matrix']
        fids = data['fids']
        
        # Filter if AOI is specified
        if aoi is not None:
            fid_set = set(gdf['fid'])
            indices = [i for i, fid in enumerate(fids) if fid in fid_set]
            dki_matrix = dki_matrix[np.ix_(indices, indices)]
            fids = fids[indices]
        
        # Load precomputed pairs
        with open(pairs_file, 'rb') as f:
            pairs = pickle.load(f)
            
        logger.info('Loaded precomputed distance matrix and pairs')
    else:
        # Compute all pairs
        pairs = [(k, i) for k in I_plus for i in I_plus if k != i]
        logger.info(f"Computing distances for {len(pairs)} pairs")
        
        # Prepare arguments for parallel distance calculation
        args = [(pair, gdf, centroid_points) for pair in pairs]
        
        # Calculate distances in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = list(pool.starmap(calculate_distance, args))
        
        # Build distance matrix
        fids = np.array(sorted(set(gdf['fid'])))
        n = len(fids)
        dki_matrix = np.zeros((n, n), dtype=float)
        
        # Create mapping from fid to index
        fid_to_idx = {fid: idx for idx, fid in enumerate(fids)}
        
        # Fill distance matrix
        for (fid1, fid2), value in results:
            if fid1 in fid_to_idx and fid2 in fid_to_idx:
                i = fid_to_idx[fid1]
                j = fid_to_idx[fid2]
                dki_matrix[i, j] = value
                # For symmetric distances
                dki_matrix[j, i] = value
        
        # Normalize distance matrix
        dki_matrix = normalize_dki_matrix(dki_matrix)
        
        # Save results for future use
        np.savez(dki_file, dki_matrix=dki_matrix, fids=fids)
        with open(pairs_file, 'wb') as f:
            pickle.dump(pairs, f)
            
        logger.info('Computed and saved distance matrix and pairs')
    
    return gdf, centroid_to_id, dki_matrix, fids

def calculate_distance(
    pair: Tuple[int, int],
    gdf: gpd.GeoDataFrame,
    centroid_points: List[Point]
) -> Tuple[Tuple[int, int], float]:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        pair: Tuple of (k, i) indices
        gdf: GeoDataFrame with geometries
        centroid_points: List of centroid points
        
    Returns:
        Tuple of ((fid1, fid2), distance)
    """
    k, i = pair
    return (gdf.at[k, 'fid'], gdf.at[i, 'fid']), round(centroid_points[k].distance(centroid_points[i]), 2)

def normalize_dki_matrix(
    dki_matrix: np.ndarray,
    lower_bound: float = 0,
    upper_bound: float = 100
) -> np.ndarray:
    """
    Normalize a distance matrix to a specified range.
    
    Args:
        dki_matrix: Distance matrix to normalize
        lower_bound: Lower bound for normalized values
        upper_bound: Upper bound for normalized values
        
    Returns:
        Normalized distance matrix
    """
    logger.info("Normalizing distance matrix")
    
    # Get min and max values
    min_val = dki_matrix.min()
    max_val = dki_matrix.max()
    
    # Skip normalization if min equals max (constant matrix)
    if min_val == max_val:
        return dki_matrix
    
    # Apply min-max normalization
    normalized_dki = (dki_matrix - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound
    
    logger.info("Distance matrix normalized")
    return normalized_dki

def categorize_values(
    vi: Dict[int, float],
    thresholds: List[float]
) -> Dict[int, int]:
    """
    Categorize values based on provided thresholds.
    
    Args:
        vi: Dictionary with float values
        thresholds: List of five thresholds in ascending or descending order
        
    Returns:
        Dictionary with categorized values (0-4)
    """
    categorized_vi = {}
    ascending = thresholds[0] < thresholds[-1]
    
    for key, value in vi.items():
        if ascending:
            if value < thresholds[0]:
                categorized_vi[key] = 0
            elif value < thresholds[1]:
                categorized_vi[key] = 1
            elif value < thresholds[2]:
                categorized_vi[key] = 2
            elif value < thresholds[3]:
                categorized_vi[key] = 3
            elif value < thresholds[4]:
                categorized_vi[key] = 4
            else:
                categorized_vi[key] = 0
        else:
            if value > thresholds[0]:
                categorized_vi[key] = 0
            elif value > thresholds[1]:
                categorized_vi[key] = 1
            elif value > thresholds[2]:
                categorized_vi[key] = 2
            elif value > thresholds[3]:
                categorized_vi[key] = 3
            elif value > thresholds[4]:
                categorized_vi[key] = 4
            else:
                categorized_vi[key] = 0
    
    return categorized_vi

def _compute_value_for_centroid(
    centroid_idx: int,
    p: Point,
    gs: List[Any],
    dataset: gpd.GeoDataFrame,
    centroid_to_id: Dict[int, int],
    calc_type: str
) -> Tuple[int, float]:
    """
    Find closest geometry to a point and return either distance or raster value.
    
    Args:
        centroid_idx: Index of the centroid
        p: Point geometry
        gs: List of geometries to check
        dataset: Dataset with geometries and values
        centroid_to_id: Mapping from centroid index to ID
        calc_type: Type of calculation ('point' or 'dist')
        
    Returns:
        Tuple of (centroid_id, value)
    """
    min_dist = float('inf')
    min_index = None
    
    for j, g in enumerate(gs):
        current_dist = p.distance(g)
        if current_dist < min_dist:
            min_dist = current_dist
            min_index = j
    
    if calc_type == 'point':
        value = dataset['raster_value'].iloc[min_index]
    else:
        value = min_dist
    
    return centroid_to_id[centroid_idx], value

def weight_calculation(
    gdf: gpd.GeoDataFrame,
    dataset: gpd.GeoDataFrame,
    centroid_to_id: Dict[int, int],
    dataset_name: str = '',
    subset: bool = False,
    thresholds: List[float] = None,
    type: str = 'point'
) -> Dict[int, int]:
    """
    Calculate weights based on distance or raster values and categorize into thresholds.
    
    Args:
        gdf: GeoDataFrame with geometries
        dataset: Dataset with values
        centroid_to_id: Mapping from centroid index to ID
        dataset_name: Name of the dataset for logging
        subset: Whether this is a subset of data
        thresholds: List of threshold values for categorization
        type: Type of calculation ('point' or 'dist')
        
    Returns:
        Dictionary of categorized weights
    """
    logger.info(f"Calculating weights for {dataset_name}")
    
    # Special case for WindSpeed and Bathymetry
    if dataset_name in ['WindSpeed', 'Bathymetry']:
        type = 'point'
    
    # Get centroids and geometries
    ps = gdf.geometry.centroid
    gs = dataset.geometry
    
    logger.info(f"Processing {len(ps)} centroids and {len(gs)} geometries")
    
    # Set up results directory and cache file
    results_folder = Path("Results")
    results_folder.mkdir(exist_ok=True)
    vi_file = results_folder / f"vi_{dataset_name}.json"
    
    # Check if cached values exist and are valid
    flag = False
    if vi_file.exists():
        try:
            with open(vi_file) as json_file:
                vi = json.load(json_file)
                logger.info(f"Loaded weights from cache: {vi_file}")
                
                # Check if cache is valid for current dataset
                if len(gdf) <= len(vi):
                    # Filter the vi dictionary to match gdf's fid values
                    vi = {
                        key: value 
                        for key, value in vi.items() 
                        if int(key) in gdf['fid'].values
                    }
                    flag = True
                else:
                    logger.warning("Cache has fewer entries than current dataset, recalculating")
        except Exception as e:
            logger.warning(f"Error loading cached weights: {str(e)}")
    
    # Calculate weights if not loaded from cache
    if not flag:
        # Prepare a function for parallel processing
        helper_func = partial(
            _compute_value_for_centroid,
            gs=gs,
            dataset=dataset,
            centroid_to_id=centroid_to_id,
            calc_type=type
        )
        
        # Prepare data for parallel processing
        data_for_pool = [(i, p) for i, p in enumerate(ps)]
        
        # Process in parallel
        with mp.Pool() as pool:
            results = pool.starmap(helper_func, data_for_pool)
        
        # Collect results
        vi = {centroid_id: value for centroid_id, value in results}
        
        # Ensure keys are integers
        vi = {int(key): value for key, value in vi.items()}
        
        # Save for future use
        with open(vi_file, 'w') as file:
            json.dump(vi, file, indent=4)
        logger.info(f"Saved weights to cache: {vi_file}")
    
    # Convert keys to int to be safe
    vi = {int(key): value for key, value in vi.items()}
    
    # Categorize according to thresholds
    updated_vi = categorize_values(vi, thresholds)
    
    logger.info("Weight calculation complete")
    return updated_vi

def assign_labels(df: pd.DataFrame) -> List[str]:
    """
    Generate cluster labels based on the grouping in the dataframe.
    
    Args:
        df: DataFrame with 'k1' and 'i1' columns
        
    Returns:
        List of labels ('C{n}' for centroids, 'N{n}' for neighbors)
    """
    logger.info("Assigning cluster labels")
    
    result = []
    grouped = df.groupby('k1')
    
    for idx, (key, sub_df) in enumerate(grouped, start=1):
        for _, row in sub_df.iterrows():
            # Check if row is a centroid (k1 == i1)
            if row['k1'] == row['i1']:
                result.append(f'C{idx}')
            else:
                result.append(f'N{idx}')
    
    logger.info("Labels assigned")
    return result