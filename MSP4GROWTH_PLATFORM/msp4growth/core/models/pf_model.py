"""
Particle Filter (PF) model implementation for MSP4GROWTH.
"""
import time
import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
from typing import Dict, List, Any, Optional
import time
logger = logging.getLogger(__name__)

# def run_particle_filter_model(
#     gdf: gpd.GeoDataFrame,
#     vi: Dict[int, float],
#     max_iterations: int = 50,
#     noise_std: float = 0.5,
#     results_dir: Path = None
# ) -> Path:
def run_particle_filter_model(gdf, vi, max_iterations: int = 50, noise_std: float = 0.5, results_dir: Path = None) -> Path:
    # Start timing the Particle Filtering optimization process
    str_time = time.time()

    print('Particle Filtering...')

    # Generate initial sample of particles
    particles = generate_filtered_sample_real(gdf, vi)

    # Run the particle filter optimization
    rem_particles = run_particle_filter_real(particles, gdf, vi)
    
    # Convert to GeoDataFrame explicitly
    rem_particles = gpd.GeoDataFrame(rem_particles, crs='EPSG:3857',geometry='geometry')
    # Replace geometry based on matching 'fid'
    rem_particles['geometry'] = rem_particles['fid'].map(gdf.set_index('fid')['geometry'])
    # Rename the 'value' column to 'interest_value'
    rem_particles['value'] = rem_particles['value'].astype(float)
    # Ensure the column name is 'interest_value'
    rem_particles.rename(columns={'value': 'interest value'}, inplace=True)
    
    results_folder = Path("Results")
    for json_fp in results_folder.glob("vi_*.json"):
        # load the simple { "fid": value, ... } dict
        with open(json_fp, 'r') as f:
            vi_dict = json.load(f)
        # ensure the keys match your rem_particles['fid'] dtype
        mapping = {int(fid): val for fid, val in vi_dict.items()}
        col_name = json_fp.stem.replace("vi_", "")
        rem_particles[col_name] = rem_particles['fid'].map(mapping)
       
    file_path = results_dir / "PF_results.geojson"
    # Save filtered results to GeoJSON files
    rem_particles.to_file(file_path, driver='GeoJSON')

    # Calculate and print total time taken for the particle filtering optimization
    end_time = time.time()
    time_taken = end_time - str_time
    minutes = time_taken // 60
    seconds = time_taken % 60
    print(f'Time taken for Particle Filtering: {minutes} minutes and {seconds} seconds')
    return file_path
# def _generate_filtered_sample(
#     dataset: gpd.GeoDataFrame,
#     vi: Dict[int, float]
# )
def generate_filtered_sample_real(dataset, vi):
    # Create a DataFrame to store the sample points
    valid_points = []
    num_points = len(dataset)
    while len(valid_points) < num_points:
        for idx, row in dataset.iterrows():
            polygon = row['geometry']
            poly_id = row['fid']
            
            # Generate random points within the polygon bounds
            min_x, min_y, max_x, max_y = polygon.bounds
            x_coords = np.random.uniform(min_x, max_x, 1)
            y_coords = np.random.uniform(min_y, max_y, 1)
            
            for x, y in zip(x_coords, y_coords):
                point = Point(x, y)
                if polygon.contains(point):
                    value = vi.get(poly_id)
                    valid_points.append({'geometry': point, 'fid': poly_id, 'value': value})
                    if len(valid_points) >= num_points:
                        break
            if len(valid_points) >= num_points:
                break
    
    # Convert to GeoDataFrame
    sample_df = pd.DataFrame(valid_points)
    
    return sample_df


def run_particle_filter_real(df, dataset, vi):
    iteration = 0
    rem_particles = df.copy()

    while True:
        rem_particles = rem_particles.reset_index(drop=True)
        if iteration > 0:
            rem_particles = add_noise_real(rem_particles)
            # Initialize a list to store values that will be appended
            values_to_append = []
            
            # Build spatial index
            dataset_sindex = dataset.sindex

            # Instead of looping through each row, use spatial indexing
            for idx, particle in rem_particles.iterrows():
                point = particle['geometry']
                value_found = False
                possible_matches_index = list(dataset_sindex.intersection(point.bounds))
                possible_matches = dataset.iloc[possible_matches_index]
                # Now you only iterate over the possible matches
                for _, row in possible_matches.iterrows():
                    polygon = row['geometry']
                    poly_id = row['fid']
                    if polygon.contains(point):
                        value = vi.get(poly_id)
                        values_to_append.append(value)
                        value_found = True
                        break
                if not value_found:
                        values_to_append.append(None)
                        
            # # Iterate over each point in rem_particles
            # for idx, particle in rem_particles.iterrows():
            #     point = particle['geometry']
            #     value_found = False

            #     # Check if the point is within any polygon in the dataset
            #     for _, row in dataset.iterrows():
            #         polygon = row['geometry']
            #         poly_id = row['OBJECTID']
                    
            #         if polygon.contains(point):
            #             # If the point is within the polygon, get the corresponding value from vi
            #             value = vi.get(poly_id)
            #             values_to_append.append(value)
            #             value_found = True
            #             break

            #     if not value_found:
            #         values_to_append.append(None)
            # # Update the 'value' column in rem_particles
            rem_particles['value'] = values_to_append
            # Filter out particles without a valid value (optional, depends on your use case)
            rem_particles = rem_particles.dropna(subset=['value'])

        # Normalize the final weight
        rem_particles['Probabilities'] = rem_particles['value'] / rem_particles['value'].sum() 
        # Re-sample points based on normalized weights
        resampled_cells, resampled_final_weights = re_sampling_real(rem_particles[['geometry', 'fid', 'value']], rem_particles['Probabilities'], rem_particles['value'])
        # Create the new DataFrame with resampled values
        rem_particles = resampled_cells.copy()
        rem_particles['value'] = resampled_final_weights
        # Convergence check 
        print(f"Iteration {iteration} completed.")
        print(f"Standard deviation: {rem_particles['value'].std()}")
        print(rem_particles['fid'].nunique())
        iteration += 1
        if iteration >= 50:  # Set a maximum number of iterations
            print("Max iterations reached, stopping for performance.")
            break
    # Keep unique rem_paprticles
    rem_particles = rem_particles.drop_duplicates(subset=['fid']).reset_index(drop=True)
    return rem_particles

    """
    Run the particle filter algorithm.
    
    Args:
        df: DataFrame with initial particles
        dataset: GeoDataFrame with polygons
        vi: Dictionary mapping IDs to importance values
        max_iterations: Maximum number of iterations
        noise_std: Standard deviation for position noise
        
    Returns:
        DataFrame with filtered particles
    """
    iteration = 0
    rem_particles = df.copy()
    
    logger.info(f"Starting particle filter with max {max_iterations} iterations")
    
    # Create spatial index for faster lookups
    dataset_sindex = dataset.sindex
    
    while True:
        rem_particles = rem_particles.reset_index(drop=True)
        
        if iteration > 0:
            # Add noise to particle positions
            rem_particles = _add_noise(rem_particles, noise_std)
            
            # Initialize list for new values
            values_to_append = []
            
            # Check each particle
            for idx, particle in rem_particles.iterrows():
                point = particle['geometry']
                value_found = False
                
                # Use spatial index for efficient intersection testing
                possible_matches_index = list(dataset_sindex.intersection(point.bounds))
                possible_matches = dataset.iloc[possible_matches_index]
                
                # Check if point is within any polygon
                for _, row in possible_matches.iterrows():
                    polygon = row['geometry']
                    poly_id = row['fid']
                    
                    if polygon.contains(point):
                        value = vi.get(poly_id)
                        values_to_append.append(value)
                        value_found = True
                        break
                
                if not value_found:
                    values_to_append.append(None)
            
            # Update values and remove particles without valid values
            rem_particles['value'] = values_to_append
            rem_particles = rem_particles.dropna(subset=['value'])
        
        # Calculate probabilities for resampling
        rem_particles['Probabilities'] = rem_particles['value'] / rem_particles['value'].sum()
        
        # Resample particles
        resampled_cells, resampled_final_weights = _re_sampling(
            rem_particles[['geometry', 'fid', 'value']],
            rem_particles['Probabilities'],
            rem_particles['value']
        )
        
        # Update particle set
        rem_particles = resampled_cells.copy()
        rem_particles['value'] = resampled_final_weights
        
        # Log progress
        logger.info(f"Iteration {iteration} completed")
        logger.info(f"Standard deviation: {rem_particles['value'].std()}")
        logger.info(f"Unique polygons: {rem_particles['fid'].nunique()}")
        
        # Check termination conditions
        iteration += 1
        if iteration >= max_iterations:
            logger.info("Maximum iterations reached, stopping")
            break
    
    # Keep only unique particles by fid
    rem_particles = rem_particles.drop_duplicates(subset=['fid']).reset_index(drop=True)
    
    return rem_particles

def re_sampling_real(cells, probabilities, final_weights):
    # Sample with replacement according to the given probabilities
    indices = np.random.choice(len(cells), size=len(cells), replace=True, p=probabilities)
    resampled_cells = cells.iloc[indices].reset_index(drop=True)
    resampled_final_weights = final_weights.iloc[indices].reset_index(drop=True)
    return resampled_cells, resampled_final_weights
def add_noise_real(df, noise_std=0.5, seed=42):
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Extract x and y coordinates
    x_coords = df['geometry'].apply(lambda geom: geom.x)
    y_coords = df['geometry'].apply(lambda geom: geom.y)
    
    # Generate random noise
    noise_x = np.random.normal(loc=0, scale=noise_std, size=len(df))
    noise_y = np.random.normal(loc=0, scale=noise_std, size=len(df))
    
    # Add noise to the coordinates
    x_coords_noisy = x_coords + noise_x
    y_coords_noisy = y_coords + noise_y
    
    # Create new geometry with noisy coordinates
    df['geometry'] = [Point(x, y) for x, y in zip(x_coords_noisy, y_coords_noisy)]
    
    return df