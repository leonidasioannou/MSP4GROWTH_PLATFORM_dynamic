import numpy as np
import pandas as pd
from pulp import *
import re
from tqdm import tqdm
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
import os
import json
import geopandas as gpd
import pickle
import billiard as mp
import geopandas as gpd
import sys
from shapely.geometry import Point, Polygon, LineString
from functools import partial


def load_dataset(file_path, crs='epsg:3857'):
    """Attempts to load a GeoJSON or shapefile, reproject to the specified CRS, and handle errors if the file is missing."""
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")
        
        # Load the file and reproject
        gdf = gpd.read_file(file_path)
        gdf = gdf.to_crs(crs)
        print(f"Loaded and reprojected '{file_path}' successfully.")
        return gdf

    except FileNotFoundError as e:
        print(e)
        sys.exit(1)  # Stop the program if the file is missing
    except Exception as e:
        print(f"An error occurred while loading '{file_path}': {e}")
        sys.exit(1)  # Stop the program for any other loading error

def calculate_dist(p, gs):
        """Calculate distances from a point `p` to all points in the geometry set `gs`."""
        return [p.distance(point) for point in gs]

def crop_habitat(gdf, aoi):
    return gdf[gdf.within(aoi.unary_union)]

def load_gdf(gdf_path, aoi=None, output_folder=None):
    """
    Load a habitat mapping GeoDataFrame from a file path and reproject it to EPSG:3857.
    
    - gdf_path: Path to the GeoDataFrame file.
    - aoi: Area of interest for spatial filtering with CRS EPSG:3857 .
    Returns:
    - gdf: GeoDataFrame loaded and reprojected to EPSG:3857.
    """

    aoi = gpd.GeoDataFrame(geometry=[aoi], crs='epsg:3857')

    gdf = load_dataset(gdf_path)

    if aoi is not None:
        try:
            gdf = crop_habitat(gdf, aoi).reset_index(drop=True)
            if gdf.empty:
                # Raise an exception with your custom error message
                raise Exception("The area of interest you chose has no data applicable for this use case")
        except Exception as e:
            print(e)  # Prints the custom error message
            sys.exit(1)  # Stop the program if the area of interest is empty
    print(len(gdf))
    centroids = gdf.geometry.centroid

    # Create mappings from centroid points to OBJECTIDs for identification
    centroid_to_id = {idx: gdf.at[idx, 'fid'] for idx in gdf.index}
    centroid_points = list(centroids)

    # Define index set `I_plus` for cells of interest in the analysis
    I_plus = gdf.index
    """
    Checking if precomputed `dki` (distances) and `pairs` are available in pickle files. 
    If not, calculate distances in parallel using multiprocessing.
    """

    dki_file = os.path.join(output_folder, 'dki.npz')
    pairs_file = f'{output_folder}/pairs.pkl'

    if os.path.exists(dki_file) and os.path.exists(pairs_file):
        # Load the distance matrix and the corresponding fids array from the NPZ file.
        data = np.load(dki_file, allow_pickle=True)
        dki_matrix = data['dki_matrix']
        fids = data['fids']
        
        # If an area of interest is specified, filter the matrix accordingly.
        if aoi is not None:
            # Create a set of valid fids from the filtered GeoDataFrame.
            fid_set = set(gdf['fid'])
            # Get the indices in fids that belong to the fid_set.
            indices = [i for i, fid in enumerate(fids) if fid in fid_set]
            # Filter the matrix (both rows and columns) and the fids array.
            dki_matrix = dki_matrix[np.ix_(indices, indices)]
            fids = fids[indices]
            
        with open(pairs_file, 'rb') as f:
            pairs = pickle.load(f)
        print('dki and pairs have been loaded from numpy and pickle files.')
        
    else:
        # Compute all pairs as before.
        pairs = [(k, i) for k in tqdm(I_plus) for i in I_plus if k != i]
        print("Number of pairs:", len(pairs))
        
        # Prepare arguments for multiprocessing distance calculations    
        args = [(pair, gdf, centroid_points) for pair in tqdm(pairs)]
        print("Number of arguments:", len(args))
        
        with mp.Pool(mp.cpu_count()) as pool:
            # Assume calculate_distance returns ((fid1, fid2), distance)
            results = list(tqdm(pool.starmap(calculate_distance, args), total=len(pairs)))
        
        # Build a sorted array of unique fids from the GeoDataFrame.
        fids = np.array(sorted(set(gdf['fid'])))
        n = len(fids)
        # Initialize the distance matrix with zeros.
        dki_matrix = np.zeros((n, n), dtype=float)
        
        # Create a mapping from each fid to its index in the fids array.
        fid_to_idx = {fid: idx for idx, fid in enumerate(fids)}
        
        # Fill the distance matrix using the computed distances.
        for (fid1, fid2), value in results:
            if fid1 in fid_to_idx and fid2 in fid_to_idx:
                i = fid_to_idx[fid1]
                j = fid_to_idx[fid2]
                dki_matrix[i, j] = value
                # Optionally, if your distances are symmetric, also fill the symmetric cell.
                dki_matrix[j, i] = value
        print(dki_matrix)
        dki_matrix = normalize_dki_matrix(dki_matrix)
        print(dki_matrix)
        # Save the distance matrix and the fids array in a single NPZ file.
        np.savez(dki_file, dki_matrix=dki_matrix, fids=fids)
        
        with open(pairs_file, 'wb') as f:
            pickle.dump(pairs, f)
        print('dki and pairs have been saved to numpy and pickle files.')
        
    return gdf, centroid_to_id, dki_matrix, fids

def categorize_values(vi, thresholds):
    """
    Categorize the values of a dictionary based on provided thresholds.

    Parameters:
    - vi: Dictionary with float values.
    - thresholds: List of five integers in ascending or descending order.

    Returns:
    - A dictionary with the same keys as vi, but with updated integer values based on thresholds.
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


# def weight_calculation(gdf, dataset, centroid_to_id, dataset_name='', subset=False, thresholds=None, type='point'):
#     """
#     Calculate weight based on distance or raster value and categorize it into thresholds.
    
#     - type: Determines whether weights are calculated based on `point` values or distances.
#     """
#     print(dataset_name)
#     if dataset_name == 'WindSpeed' or dataset_name == 'Bathymetry':
#         type = 'point'
        
#     print("CALCULATING WEIGHTS")
    
#     ps = gdf.geometry.centroid
#     print(len(ps))
#     gs = dataset.geometry
#     print(len(gs))
    
#      # Set up results folder and the path for the raw weight dictionary (vi)
#     results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
#     os.makedirs(results_folder, exist_ok=True)
#     vi_file = os.path.join(results_folder, f"vi_{dataset_name}.json")
    
#     # If raw weights are already saved, just load them and perform categorization.
#     if os.path.exists(vi_file):
#         with open(vi_file) as json_file:
#             # Load the JSON file into a Python dictionary   
#             vi = json.load(json_file)
#             print("Loaded vi from file:", vi_file)
#             if len(gdf) != len(vi):
#                 # Filter the vi the fid column of the gdf to match the keys of vi 
#                 vi = {key: value for key, value in vi.items() if int(key) in gdf['fid'].values}
#     else:
#         vi = {}
#         # Loop through each centroid; tqdm adds a progress bar if you're into that
#         for i, p in enumerate(tqdm(ps, desc="Processing centroids")):
#             min_dist = float('inf')
#             min_index = None
            
#             # Loop through each geometry in the dataset to get the minimum distance
#             for j, g in enumerate(gs):
#                 # Use Shapely's distance method to calculate the distance between two geometries
#                 current_dist = p.distance(g)
#                 if current_dist < min_dist:
#                     min_dist = current_dist
#                     min_index = j
            
#             # Depending on the type, retrieve either the raster value or the distance
#             if type == 'point':
#                 # Assumes 'raster_value' column exists in dataset; get value of the closest geometry.
#                 value = dataset['raster_value'].iloc[min_index]
#             else:
#                 value = min_dist
            
#             # Save the result in the dictionary using the cell id from centroid_to_id.
#             vi[centroid_to_id[i]] = value

#         vi = {int(key): value for key, value in vi.items()}
#        # Save the computed vi for future reuse
#         with open(vi_file, 'w') as file:
#             json.dump(vi, file, indent=4)
#         print("Saved vi to file:", vi_file)

#     vi = {int(key): value for key, value in vi.items()}
#     updated_vi = categorize_values(vi, thresholds)
#     print("WEIGHTS CALCULATED")
#     return updated_vi

def _compute_value_for_centroid(centroid_idx, p, gs, dataset, centroid_to_id, calc_type):
    """
    Helper function that finds the closest geometry to point 'p', returning
    either the minimum distance or the associated raster value.
    """
    min_dist = float('inf')
    min_index = None
    
    for j, g in enumerate(gs):
        current_dist = p.distance(g)
        if current_dist < min_dist:
            min_dist = current_dist
            min_index = j
    
    # Depending on the type, retrieve either the raster value or the distance
    if calc_type == 'point':
        value = dataset['raster_value'].iloc[min_index]
    else:
        value = min_dist

    return centroid_to_id[centroid_idx], value


def weight_calculation(
    gdf, 
    dataset, 
    centroid_to_id, 
    dataset_name='', 
    subset=False, 
    thresholds=None, 
    type='point'
):
    """
    Calculate weight based on distance or raster value and categorize it into thresholds.
    
    - type: Determines whether weights are calculated based on `point` values or distances.
    """
    print(dataset_name)
    if dataset_name in ['WindSpeed', 'Bathymetry']:
        type = 'point'
        
    print("CALCULATING WEIGHTS")
    ps = gdf.geometry.centroid
    print(len(ps))
    gs = dataset.geometry
    print(len(gs))
    
    # Results path
    results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
    os.makedirs(results_folder, exist_ok=True)
    vi_file = os.path.join(results_folder, f"vi_{dataset_name}.json")
    flag = False
    # If raw weights are already saved, just load them
    if os.path.exists(vi_file):
        with open(vi_file) as json_file:
            vi = json.load(json_file)
            print("Loaded vi from file:", vi_file)
            if len(gdf) <= len(vi):
                # Filter the vi dictionary based on gdf's fid values
                vi = {
                    key: value 
                    for key, value in vi.items() 
                    if int(key) in gdf['fid'].values
                }
                flag = True
            else:
                print("Warning: The loaded vi has fewer entries than the gdf. Recalculating.")

    if flag == False:
        # Initialize an empty dictionary to store the computed values
        helper_func = partial(
            _compute_value_for_centroid, 
            gs=gs, 
            dataset=dataset, 
            centroid_to_id=centroid_to_id, 
            calc_type=type
        )
        # Prepare data for parallel processing
        data_for_pool = [(i, p) for i, p in enumerate(ps)]
        # Use multiprocessing to compute values in parallel
        with mp.Pool() as pool:
            results = pool.starmap(helper_func, data_for_pool)
        
        # Collect results into a dictionary (centroid_to_id -> distance/raster_val)
        vi = {}
        for centroid_id, value in results:
            vi[centroid_id] = value

        vi = {int(key): value for key, value in vi.items()}
        
        # Save the computed vi for future reuse
        with open(vi_file, 'w') as file:
            json.dump(vi, file, indent=4)
        print("Saved vi to file:", vi_file)

    # Convert keys to int just to be safe
    vi = {int(key): value for key, value in vi.items()}
    
    # Then categorize according to your thresholds logic
    updated_vi = categorize_values(vi, thresholds)
    
    print("WEIGHTS CALCULATED")
    return updated_vi

# Function to calculate Euclidean distance between points
def calculate_distance(pair, gdf, centroid_points):
    """Calculate distance between two points identified by pair indices `k` and `i`."""
    k, i = pair
    return (gdf.at[k, 'fid'], gdf.at[i, 'fid']), round(centroid_points[k].distance(centroid_points[i]), 2)

# Min-max normalization function
def min_max_normalize(value, min_value, max_value, new_min, new_max):
    """Scale `value` from [min_value, max_value] range to [new_min, new_max] range."""
    return ((value - min_value) / (max_value - min_value)) * (new_max - new_min) + new_min

# Normalize dki values to a scale from 1 to 5
def normalize_dki(dki):
    """Normalize dictionary of dki values to a 1-100 scale using min-max normalization."""
    print("NORMALIZING DKIS")
    min_dki = min(dki.values())
    max_dki = max(dki.values())
    normalized_dki = {}
    for key, value in dki.items():
        normalized_dki[key] = min_max_normalize(value, min_dki, max_dki, 0, 100)
    print("NORMALIZING DONE")
    return normalized_dki

def normalize_dki_matrix(dki_matrix, lower_bound=0, upper_bound=100):
    """
    Normalize a numpy array of dki values to a specified scale (default: 1-100)
    using min-max normalization.
    """
    print("NORMALIZING DKIS")
    min_val = dki_matrix.min()
    max_val = dki_matrix.max()
    # Apply min-max normalization:
    normalized_dki = (dki_matrix - min_val) / (max_val - min_val) * (upper_bound - lower_bound) + lower_bound
    print("NORMALIZING DONE")
    return normalized_dki

# Function to assign labels to results 
def assign_labels(df):
    """Generate cluster labels based on the grouped DataFrame structure."""
    print("ASSIGNING LABELS")
    result = []
    grouped = df.groupby('k1')
    for idx, (key, sub_df) in enumerate(grouped, start=1):  # Start idx from 1
        for _, row in sub_df.iterrows():
            # Checking if k1 is equal to i1
            if row['k1'] == row['i1']:
                result.append(f'C{idx}')
            else:
                result.append(f'N{idx}')
    print("LABELS ASSIGNED")
    return result

def WS(I_plus=None, vi=None, dki_matrix=None, fids=None, lambda_value=0.5, c=3, u=10, l=3):
    """
    Solves a Weighted Sum (WS) optimization problem using linear programming.
 
    - I_plus: Set of locations for modeling.
    - vi: Importance weight dictionary.
    - dki: Distance cost dictionary.
    - lambda_value: Balancing parameter for weighted sum.
    - c, u, l: Parameters defining constraint limits.
    """
    vi_df = pd.DataFrame(list(vi.items()), columns=['Cell_ID', 'Value'])
    vi_df.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Results/vi_WS.csv', index=False)
   
    print('Creating variables...')
    xki = LpVariable.dicts("xki", [(k, i) for k in tqdm(I_plus) for i in I_plus], cat='Binary')
   
 
    print('Creating model...')
    model = LpProblem("Maximize_Weighted_Objective", LpMaximize)
   
    fid_to_idx = {fid: idx for idx, fid in enumerate(sorted(I_plus))}
    print(fid_to_idx)
    print('Adding objective function...')
    model += (lambda_value * lpSum(vi[i] * xki[(k, i)] for i in tqdm(I_plus) for k in I_plus) - \
            (1 - lambda_value) * lpSum(dki_matrix[fid_to_idx[k], fid_to_idx[i]] * xki[(k, i)] \
               for k in tqdm(I_plus) for i in I_plus if k != i)), "Weighted_Sum"
 
    print('Adding contstrains...')
    model += lpSum(xki[(k, i)] for (k, i) in xki if k == i) <= c
 
    for k in tqdm(I_plus):
        sum = lpSum(xki[(k, i)] for i in I_plus if k != i)
        model += sum <= u * xki[(k, k)], f"Upper_limit_{k}"
        model += sum >= l * xki[(k, k)], f"Lower_limit_{k}"
 
    for i in tqdm(I_plus):
        model += lpSum(xki[(k, i)] for k in I_plus) <= 1, f"Cell_{i}_assigned_to_one_center"
   
    print('Solving model...')
    model.solve(GUROBI_CMD(threads=12))
   
    print("Status:", LpStatus[model.status])
    print("Solved")
 
    # Collect variable values into DataFrame for results
    data = []
    pattern = re.compile(r'\((\d+),_(\d+)\)')
 
    for v in model.variables():
        complex_name = v.name.split("_", 1)[1]
        match = pattern.search(complex_name)
        if match:
            k1, i1 = match.groups()  
            z = v.varValue
            data.append({"k1": float(k1), "i1": float(i1), "z": z})
 
    df = pd.DataFrame(data)
    df.to_csv(f'{ os.path.dirname(os.path.abspath(__file__))}/Results/model_results.csv', index=False)
    filter_data = df[df['z'] == 1.0]
    filter_data.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Results/filter_model_results.csv', index=False)
   
    return filter_data

def two_objective_model(I_plus=None, vi=None, dki_matrix=None, fids=None, c=3, u=10, l=3):
    """
    Constructs and solves a two-objective model with PyAugmecon.
    
    - I_plus: Set of locations for modeling.
    - vi: Importance weight dictionary.
    - dki: Distance cost dictionary.
    - c, u, l: Parameters defining constraint limits.
    """
    vi_df = pd.DataFrame(list(vi.items()), columns=['Cell_ID', 'Value'])
    vi_df.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Results/vi_AUG.csv', index=False)

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
    df_combined.to_csv(f'{os.path.dirname(os.path.abspath(__file__))}/Results/model_results_aug.csv', index=False)
    
    return augmecon, df_combined

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

def re_sampling_real(cells, probabilities, final_weights):
    # Sample with replacement according to the given probabilities
    indices = np.random.choice(len(cells), size=len(cells), replace=True, p=probabilities)
    resampled_cells = cells.iloc[indices].reset_index(drop=True)
    resampled_final_weights = final_weights.iloc[indices].reset_index(drop=True)
    return resampled_cells, resampled_final_weights

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
