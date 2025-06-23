"""
Weighted Sum (WS) model implementation for MSP4GROWTH.
"""
import time
import pandas as pd
import logging
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Iterable, Tuple, Union
import geopandas as gpd
import numpy as np
from tqdm import tqdm
from ..utils.geo_utils import assign_labels
import os
from gurobipy import Model, GRB, quicksum
import json
logger = logging.getLogger(__name__)

def run_weighted_sum_model(gdf, vi, dki_matrix, fids,
                          lambda_value=0.5,
                          c=3,
                          u=10,
                          l=3,
                          results_dir=None):
    # Begin timing for the Weighted Sum optimization
    str_time = time.time()

    print('Weighted Sum...')
    
    print(f'Loading numpy file for lambda={lambda_value}...')
    # Execute the Weighted Sum optimization function `WS` with current lambda value
    model = WS(I_plus=vi.keys(), vi=vi, dki_matrix=dki_matrix, fids=fids, 
               lambda_value=lambda_value, c=c, u=u, l=l, results_dir=results_dir)
    
    # Check if the model is empty (no solutions found)
    if model.empty:
        print("No solutions found for the weighted sum model.")
        # Create an empty GeoJSON file
        empty_gdf = gpd.GeoDataFrame(columns=gdf.columns)
        file_path = results_dir / f"WS_results_{lambda_value}.geojson"
        empty_gdf.to_file(file_path, driver='GeoJSON')
        
        # Calculate and print total time taken
        end_time = time.time()
        time_taken = end_time - str_time
        minutes = time_taken // 60
        seconds = time_taken % 60
        print(f'Time taken for Weighted Sum: {minutes} minutes and {seconds} seconds')
        return file_path
    
    # Process results if not empty
    model = model.sort_values(by='k1')
    model['Description'] = assign_labels(model)
    model['interest value'] = model['i1'].map(vi)  

    # Extract unique fid values for filtering purposes
    model_object_ids = model['i1'].unique()

    # Merge descriptions back into the GeoDataFrame to prepare for filtering
    gdf_with_desc = gdf.merge(model[['i1', 'Description', 'interest value']], left_on='fid', right_on='i1', how='left')

    # Filter the GeoDataFrame based on OBJECTIDs from the weighted sum results
    filtered_gdf = gdf_with_desc[gdf_with_desc['fid'].isin(model_object_ids)]

    results_folder = Path("Results")
    for json_fp in results_folder.glob("vi_*.json"):
        # load the simple { "fid": value, ... } dict
        with open(json_fp, 'r') as f:
            vi_dict = json.load(f)
        # ensure the keys match your filtered_gdf['fid'] dtype
        mapping = {int(fid): val for fid, val in vi_dict.items()}
        col_name = json_fp.stem.replace("vi_", "")
        filtered_gdf[col_name] = filtered_gdf['fid'].map(mapping)
       
    
    file_path = results_dir / f"WS_results_{lambda_value}.geojson"
    # Save filtered results to GeoJSON files, named according to the lambda value
    filtered_gdf.to_file(file_path, driver='GeoJSON')
   
    # Calculate and print total time taken for the weighted sum optimization
    end_time = time.time()
    time_taken = end_time - str_time
    minutes = time_taken // 60
    seconds = time_taken % 60
    print(f'Time taken for Weighted Sum: {minutes} minutes and {seconds} seconds')
    return file_path

def WS(I_plus=None, vi=None, dki_matrix=None, fids=None, lambda_value=0.5, c=3, u=10, l=3, results_dir="."):
    """
    Solves a Weighted Sum (WS) optimization problem using Gurobi.
    """
    # Pre-flight setup
    results_dir = Path(results_dir)
    (results_dir / "Results").mkdir(parents=True, exist_ok=True)
    
    vi_df = pd.DataFrame(list(vi.items()), columns=['Cell_ID', 'Value'])
    vi_df.to_csv(f'{results_dir}/Results/vi_WS.csv', index=False)
    
    # Debug info
    print(f"Number of candidate locations: {len(list(I_plus))}")
    print(f"vi values sample: {list(vi.items())[:5]}")
    print(f"dki_matrix shape: {dki_matrix.shape}")
    
    # Create a mapping for fids to matrix indices
    fid_to_idx = {fid: idx for idx, fid in enumerate(sorted(I_plus))}
    print(fid_to_idx)
    
    print('Creating variables...')
    # Create the model and variables - VERY IMPORTANT: Note the "_" in variable names
    model = Model("Maximize_Weighted_Objective")
    
    xki = {}
    for k in tqdm(I_plus):
        for i in I_plus:
            # ⭐ CRITICAL: Using the exact same variable naming format as your original code
            xki[(k, i)] = model.addVar(vtype=GRB.BINARY, name=f"xki_({k},_{i})")
    
    model.update()
    
    print('Creating model...')
    
    print('Adding objective function...')
    # ⭐ Adding a constant term to ensure at least some centers open
    benefit_term = quicksum(vi[i] * xki[(k, i)] for i in tqdm(I_plus) for k in I_plus)
    distance_term = quicksum(
        dki_matrix[fid_to_idx[k], fid_to_idx[i]] * xki[(k, i)]
        for k in tqdm(I_plus) for i in I_plus if k != i
    )
    
    # ⭐ Try increasing lambda to favor benefits (e.g., 0.7)
    # Also add a small constant to ensure opening centers is favorable
    incentive = 0.1 * quicksum(xki[(k, k)] for k in I_plus)  
    model.setObjective(
        lambda_value * benefit_term - (1 - lambda_value) * distance_term + incentive,
        GRB.MAXIMIZE
    )
    
    print('Adding constraints...')
    # Total centers constraint
    model.addConstr(quicksum(xki[(k, i)] for (k, i) in xki if k == i) <= c)
    
    # ⭐ Force at least one center to open
    model.addConstr(quicksum(xki[(k, k)] for k in I_plus) >= 1, "At_least_one_center")
    
    # Per-center assignment limits
    for k in tqdm(I_plus):
        sum_served = quicksum(xki[(k, i)] for i in I_plus if k != i)
        model.addConstr(sum_served <= u * xki[(k, k)], f"Upper_limit_{k}")
        model.addConstr(sum_served >= l * xki[(k, k)], f"Lower_limit_{k}")
    
    # Every cell assigned to at most one center
    for i in tqdm(I_plus):
        model.addConstr(
            quicksum(xki[(k, i)] for k in I_plus) <= 1,
            f"Cell_{i}_assigned_to_one_center"
        )
    
    print('Solving model...')
    model.setParam('Threads', 12)
    model.setParam('OutputFlag', 1)
    model.setParam('TimeLimit', 600)
    
    model.optimize()
    
    print(f"Status: {model.status}")
    print(f"Objective value: {model.objVal if model.status == GRB.OPTIMAL else 'N/A'}")
    
    print("Solved")
    
    # Extract solution using the EXACT SAME pattern as your original code
    data = []
    pattern = re.compile(r'\((\d+),_(\d+)\)')  # Note the underscore here
    
    for v in model.getVars():
        var_name = v.VarName
        if var_name.startswith('xki_'):
            name_without_prefix = var_name[4:]
            match = pattern.search(name_without_prefix)
            if match:
                k1, i1 = match.groups()
                z = v.X
                data.append({"k1": float(k1), "i1": float(i1), "z": z})
    
    df = pd.DataFrame(data)
    df.to_csv(f'{results_dir}/Results/model_results.csv', index=False)
    
    filter_data = df[df['z'] == 1.0]
    filter_data.to_csv(f'{results_dir}/Results/filter_model_results.csv', index=False)
    
    # Print solution summary
    if not filter_data.empty:
        print(f"Found {len(filter_data)} assignments with z=1")
        centers = filter_data[filter_data['k1'] == filter_data['i1']]['k1'].unique()
        print(f"Number of centers opened: {len(centers)}")
    else:
        print("No feasible solution found or all variables are 0")
    
    return filter_data