"""
Augmecon model implementation for MSP4GROWTH.
"""
import os
from dotenv import load_dotenv
load_dotenv()

# Set the local license file path
license_file_path = "/home/leonidas/Programming/MSP4GROWTH_PLATFORM/gurobi.lic"
os.environ["GRB_LICENSE_FILE"] = license_file_path

# Setup the Gurobi environment
import gurobipy as gp
env = gp.Env(empty=True)

# Set environment parameter to use the license file
env.setParam('LogFile', 'gurobi.log')

# Instead of using WLS parameters, just start the environment with the license file
env.start()

import time
import logging
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, Tuple
import json
from pyomo.core.base import (
    Binary,
    ConcreteModel,
    Constraint,
    ObjectiveList,
    ConstraintList,
    Param,
    Set,
    Var,
    maximize,
    minimize
)
from pyaugmecon import PyAugmecon

from ..utils.geo_utils import assign_labels

logger = logging.getLogger(__name__)


# def run_augmecon_model(
#     gdf: gpd.GeoDataFrame,
#     vi: Dict[int, float],
#     dki_matrix: np.ndarray,
#     fids: np.ndarray,
#     c: int = 3,
#     u: int = 10,
#     l: int = 3,
#     results_dir: Path = None
# )
def run_augmecon_model(gdf, vi, dki_matrix, fids,
                       
                       c: int = 3,
                       u: int = 10,
                       l: int = 3,
                       results_dir: Path = None):
    # Start timing the Augmecon process.
    str_time = time.time()
    os.makedirs(results_dir, exist_ok=True)
    # Run the multi-objective optimization using the loaded `vi` and `normalized_dki` with the `two_objective_model` function
    _, df = two_objective_model(vi.keys(), vi, dki_matrix, fids, c=c, u=u, l=l, results_dir=results_dir)
    # Organize and annotate results with descriptions and interest values
    df = df.sort_values(by='k1')
    df['Description'] = assign_labels(df)
    df['interest value'] = df['i1'].map(vi)

    # Identify unique fid values from results for filtering
    model_AUG_object_ids = df['i1'].unique()

    # Merge descriptions into the GeoDataFrame for further filtering
    gdf_with_desc_aug = gdf.merge(df[['i1', 'Description', 'interest value']], left_on='fid', right_on='i1', how='left')

    # Filter the GeoDataFrame for rows matching Augmecon model OBJECTIDs
    filtered_AUG_gdf = gdf_with_desc_aug[gdf_with_desc_aug['fid'].isin(model_AUG_object_ids)]
    
    results_folder = Path("Results")
    for json_fp in results_folder.glob("vi_*.json"):
        # load the simple { "fid": value, ... } dict
        with open(json_fp, 'r') as f:
            vi_dict = json.load(f)
        # ensure the keys match your filtered_AUG_gdf['fid'] dtype
        mapping = {int(fid): val for fid, val in vi_dict.items()}
        col_name = json_fp.stem.replace("vi_", "")
        filtered_AUG_gdf[col_name] = filtered_AUG_gdf['fid'].map(mapping)
    
    file_path = results_dir / "AUG_results.geojson"
    os.makedirs(results_dir, exist_ok=True)
    # Save filtered results to GeoJSON files for further use or visualization
    print(f'Saving results to {file_path}')
    filtered_AUG_gdf.to_file(file_path, driver='GeoJSON')

    # Calculate time taken for the Augmecon optimization step
    end_time = time.time()
    time_taken = end_time - str_time
    minutes = time_taken // 60
    seconds = time_taken % 60
    print(f'Time taken for Augmencon2: {minutes} minutes and {seconds} seconds')
    return file_path

def two_objective_model(I_plus=None, vi=None, dki_matrix=None, fids=None, c=3, u=10, l=3, results_dir=None):
    """
    Constructs and solves a two-objective model with PyAugmecon.
    
    - I_plus: Set of locations for modeling.
    - vi: Importance weight dictionary.
    - dki: Distance cost dictionary.
    - c, u, l: Parameters defining constraint limits.
    """
    vi_df = pd.DataFrame(list(vi.items()), columns=['Cell_ID', 'Value'])
    os.makedirs(f'{results_dir}/Results', exist_ok=True)
    vi_df.to_csv(f'{results_dir}/Results/vi_AUG.csv', index=False)

    model = ConcreteModel()
    model.I_plus = Set(ordered=True, initialize=I_plus)
    model.c = Param(initialize=c)  
    model.u = Param(initialize=u) 
    model.l = Param(initialize=l)  

    print('Creating variables...')
    model.xki = Var([(k, i) for k in tqdm(model.I_plus) for i in model.I_plus], within=Binary, initialize=0) # Initialize to 0
    
    # Constraints
    print('Creating constraints...')
    def diagonal_constraint(model):
        return sum(model.xki[(k, k)] for k in model.I_plus) <= model.c
    model.centroid = Constraint(rule=diagonal_constraint)
    print('Centroid constraint created')

    def upper_limit_constraint(model, k):
        return sum(model.xki[k, i] for i in model.I_plus if k != i) <= model.u * model.xki[k, k]
    model.upper_limit = ConstraintList()
    for k in model.I_plus:
        model.upper_limit.add(upper_limit_constraint(model, k))
    print('Upper limit constraint created')

    def lower_limit_constraint(model, k):
        return sum(model.xki[k, i] for i in model.I_plus if k != i) >= model.l * model.xki[k, k]
    model.lower_limit = ConstraintList()
    for k in model.I_plus:
        model.lower_limit.add(lower_limit_constraint(model, k))
    print('Lower limit constraint created')

    def assignment_constraint(model, i):
        return sum(model.xki[k, i] for k in model.I_plus) <= 1
    model.assignment_constraints = ConstraintList()
    for i in model.I_plus:
        model.assignment_constraints.add(assignment_constraint(model, i))
    print('Assignment constraint created')

    print('Creating objectives...')
    def objective1_rule(mod):
        return sum(vi[i] * mod.xki[(k, i)] for i in tqdm(mod.I_plus) for k in mod.I_plus)
    # def objective2_rule(mod):
    #     return sum(dki[(k, i)] * mod.xki[(k, i)] for k in tqdm(mod.I_plus) for i in mod.I_plus if k != i)
    model.fid_to_idx = {fid: idx for idx, fid in enumerate(sorted(model.I_plus))}

    def objective2_rule(mod):
        return sum(dki_matrix[mod.fid_to_idx[k], mod.fid_to_idx[i]] * mod.xki[(k, i)] 
               for k in tqdm(mod.I_plus) for i in mod.I_plus if k != i)

    model.obj_list = ObjectiveList()
    model.obj_list.add(expr=objective1_rule(model), sense=maximize)
    model.obj_list.add(expr=objective2_rule(model), sense=minimize)

    # By default deactivate all the objective functions
    for o in range(len(model.obj_list)):
        model.obj_list[o + 1].deactivate()

    opts = {
        'name': 'MultiObjective_Optimization',
        'grid_points': 2  
    }

    print('Creating model...')
    augmecon = PyAugmecon(model, opts)
    print('Solving model...')
    augmecon.solve()
    sols = augmecon.get_pareto_solutions()  
    decision_vars = augmecon.get_decision_variables(sols[0]) 
    print('Model solved')

    # Processing solution data into DataFrame for final results
    df_xki = pd.DataFrame(decision_vars['xki'], columns=['Value'])
    df_xki['Variable'] = 'xki'
    df_xki.reset_index(inplace=True)
    df_xki.rename(columns={'index': 'Indices'}, inplace=True)

    df_slack = pd.DataFrame(decision_vars['Slack'], columns=['Value'])
    df_slack['Variable'] = 'Slack'
    df_slack.reset_index(inplace=True)
    df_slack.rename(columns={'index': 'Indices'}, inplace=True)

    df_combined = pd.concat([df_xki, df_slack], ignore_index=True)
    df_combined = df_combined.iloc[:, :3]
    df_combined.columns = ['k1', 'i1', 'z']
    df_combined = df_combined[:-1]
    df_combined = df_combined[df_combined['z'] == 1.0]
    df_combined.to_csv(f'{results_dir}/Results/model_results_aug.csv', index=False)
    
    return augmecon, df_combined
