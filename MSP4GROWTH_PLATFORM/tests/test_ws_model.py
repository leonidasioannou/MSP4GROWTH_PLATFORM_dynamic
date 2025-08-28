#!/usr/bin/env python3
"""
Test script to diagnose issues with the Weighted Sum model implementation
"""
import os
import sys
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Polygon

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# Try different import approaches
try:
    # First try direct import
    from msp4growth.core.models.ws_model import run_weighted_sum_model, WS
    from msp4growth.core.utils.geo_utils import assign_labels
except ImportError:
    try:
        # Try alternative import path
        from msp4growth.core.models.ws_model import run_weighted_sum_model, WS
        
        # Define a simple assign_labels function for testing
        def assign_labels(df):
            """Simple version of assign_labels for testing"""
            result = []
            grouped = df.groupby('k1')
            for idx, (key, sub_df) in enumerate(grouped, start=1):
                for _, row in sub_df.iterrows():
                    if row['k1'] == row['i1']:
                        result.append(f'C{idx}')
                    else:
                        result.append(f'N{idx}')
            return result
    except ImportError:
        print("Cannot import required modules. Check your project structure.")
        print("Current sys.path:", sys.path)
        print("Looking for files in:", os.listdir(project_root))
        if os.path.exists(os.path.join(project_root, "msp4growth")):
            print("msp4growth directory exists. Contents:", 
                  os.listdir(os.path.join(project_root, "msp4growth")))
        sys.exit(1)

def create_test_data():
    """Create test data for the WS model"""
    print("Creating test data...")
    
    # Create a simple GeoDataFrame with 4 polygons
    polygons = [
        Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        Polygon([(1, 0), (2, 0), (2, 1), (1, 1)]),
        Polygon([(0, 1), (1, 1), (1, 2), (0, 2)]),
        Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
    ]
    
    gdf = gpd.GeoDataFrame({
        'fid': [1, 2, 3, 4],
        'geometry': polygons
    })
    
    # Create a simple importance value dictionary
    vi = {1: 0.5, 2: 0.7, 3: 0.2, 4: 0.9}
    
    # Create a distance matrix
    dki_matrix = np.array([
        [0.0, 1.0, 1.0, 1.4],
        [1.0, 0.0, 1.4, 1.0],
        [1.0, 1.4, 0.0, 1.0],
        [1.4, 1.0, 1.0, 0.0]
    ])
    
    # Convert matrix to dictionary format expected by WS
    dki_dict = {}
    for i in range(4):
        for j in range(4):
            if i != j:  # Exclude diagonal elements
                dki_dict[(i+1, j+1)] = dki_matrix[i, j]
    
    return gdf, vi, dki_dict, dki_matrix

def test_ws_function():
    """Test the WS function directly"""
    gdf, vi, dki_dict, _ = create_test_data()
    
    # Create a test results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    (results_dir / "Results").mkdir(exist_ok=True)
    
    print("\nTesting WS function...")
    try:
        # Print the types and sizes to debug
        print(f"Type of I_plus: {type(vi.keys())}")
        print(f"Number of elements in I_plus: {len(vi.keys())}")
        print(f"Type of vi: {type(vi)}")
        print(f"Number of elements in vi: {len(vi)}")
        print(f"Type of dki_dict: {type(dki_dict)}")
        print(f"Number of elements in dki_dict: {len(dki_dict)}")
        
        # Call the WS function
        filter_data = WS(
            I_plus=vi.keys(),
            vi=vi,
            dki=dki_dict,
            lambda_value=0.5,
            c=2,
            u=2,
            l=1,
            results_dir=results_dir
        )
        
        print(f"WS function returned object of type: {type(filter_data)}")
        print(f"Filter data shape: {filter_data.shape}")
        print(f"Filter data columns: {filter_data.columns}")
        print("WS function test PASSED")
        return filter_data
    except Exception as e:
        print(f"WS function test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def test_run_weighted_sum_model():
    """Test the full run_weighted_sum_model function"""
    gdf, vi, dki_dict, dki_matrix = create_test_data()
    
    # Create a test results directory
    results_dir = Path("test_results")
    results_dir.mkdir(exist_ok=True)
    
    print("\nTesting run_weighted_sum_model function...")
    try:
        # Print input data types for debugging
        print(f"Type of gdf: {type(gdf)}")
        print(f"Type of vi: {type(vi)}")
        print(f"Type of dki_matrix: {type(dki_matrix)}")
        print(f"Shape of dki_matrix: {dki_matrix.shape}")
        
        # Call run_weighted_sum_model with dict
        file_path = run_weighted_sum_model(
            gdf=gdf,
            vi=vi,
            normalized_dki=dki_dict,  # Use the dictionary format
            lambda_value=0.5,
            c=2,
            u=2,
            l=1,
            results_dir=results_dir
        )
        
        print(f"run_weighted_sum_model returned: {file_path}")
        print("run_weighted_sum_model test PASSED")
        return file_path
    except Exception as e:
        print(f"run_weighted_sum_model test FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("===== WS Model Test Script =====")
    
    # First test the WS function directly
    filter_data = test_ws_function()
    
    # Then test the full run_weighted_sum_model function
    if filter_data is not None:
        file_path = test_run_weighted_sum_model()
        if file_path is not None and os.path.exists(file_path):
            print(f"\nFinal output file exists at: {file_path}")
            print("All tests PASSED")
        else:
            print("\nFinal output file was not created.")
            print("Tests FAILED")
    else:
        print("\nSkipping run_weighted_sum_model test due to WS function failure")