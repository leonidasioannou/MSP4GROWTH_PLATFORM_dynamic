import geopandas as gpd
from tqdm import tqdm
from MSP4GROWTH_Functions_final import *
import pickle
import os
import time
import json
from shapely.geometry import shape
from pathlib import Path

class MSP4GROWTH_load_data:

    def __init__(self, request_id, input_polygon, use_case, model, C_number, N_size, datasets, thresholds, calc_type="dist"):
        self.request_id = request_id
        self.input_polygon = input_polygon
        self.use_case = use_case
        self.model = model
        self.datasets = datasets
        self.thresholds = thresholds
        self.calc_type = calc_type
        self.C_number = C_number
        self.N_size = N_size
        self.results_abs_path = Path(os.path.dirname(os.path.abspath(__file__)))/"Results"/ str(self.request_id)

    def model_choice(self, gdf, vi, dki_matrix, fids):
        resulting_geojsons: dict[Path] = {}
        if not self.model:
            raise Exception("Invalid model choice. Please choose from 'Augment', 'WS', or 'PF'.")
        if "model_AUG" in self.model.keys():
            try:
                print("STARTING AUG")
                aug_results: Path = self.model_AUG(gdf, vi, dki_matrix, fids)
                resulting_geojsons["model_AUG"] = aug_results
            except Exception as e:
                print(f"Failed to produce results for AUG model because: {str(e)}")
        if "model_WS" in self.model.keys():
            # try:
            print("STARTING WS")
            ws_results: Path = self.model_WS(gdf, vi, dki_matrix, fids, lambda_value=self.model["model_WS"] if self.model["model_WS"] else 0.5)
            resulting_geojsons["model_WS"] = ws_results
            # except Exception as e:
                # print(f"Failed to produce results for WS model: {str(e)}")
        if "model_PF" in self.model.keys():
            try:
                print("STARTING PF")
                pf_results: Path = self.model_PF(gdf, vi)
                resulting_geojsons["model_PF"] = pf_results
            except Exception as e:
                print(f"Failed to produce results for PF model: {str(e)}")
        return resulting_geojsons

    def run_analysis(self):
        # Start timing the entire preprocessing step
        str_time = time.time()

        aoi = Polygon(self.input_polygon)                                                 

        subset = True if aoi else False
        file_path = self.results_abs_path 

        # Ensure the directory exists
        os.makedirs(file_path, exist_ok=True)

        # Load datasets with error handling
        # Habitat data
        gdf, centroid_to_id, dki_matrix, fids = load_gdf(f'{ os.path.dirname(os.path.abspath(__file__))}/Data/Habitat_map_clean.geojson', 
                                            aoi=aoi,
                                            output_folder=file_path)
        print(len(gdf), 'Habitat data loaded successfully.')
        
        datasets = {}
        weights_set = {}
        weights = []
        for threshold_type in self.thresholds.keys():
            if threshold_type not in ['WindSpeed', 'Bathymetry', 'Coastline', 'Ports']:
                if self.datasets.get(threshold_type):
                    features = self.datasets[threshold_type]["geom"]["features"]
                    geometry = [shape(feature["geometry"]) for feature in features]
                    properties = [feature["properties"] for feature in features]
                    new_gdf = gpd.GeoDataFrame(properties, geometry=geometry)
                    new_gdf.set_crs(epsg=3857, inplace=True)
                    datasets.update({threshold_type: new_gdf})
            else:
                datasets.update({threshold_type: load_dataset(f'{ os.path.dirname(os.path.abspath(__file__))}/Data/{threshold_type}.geojson')})
            
            try:
                thresholds = self.thresholds[threshold_type]["thresholds"]
                if not isinstance(thresholds, list) or len(thresholds) != 5:
                    raise ValueError(f"'thresholds' for '{threshold_type}' must be a list of exactly 5 elements. Found: {thresholds}")
            except KeyError:
                raise KeyError(f"Missing 'thresholds' key for '{threshold_type}'. Check input data.")

            try:
                weight = self.thresholds[threshold_type]["weights"]
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"'weights' for '{threshold_type}' must be a numeric value. Found: {type(weight).__name__}")
            except KeyError:
                raise KeyError(f"Missing 'weights' key for '{threshold_type}'. Check input data.")
            
            weights_set.update({
                threshold_type: weight_calculation(
                    gdf, 
                    datasets[threshold_type], 
                    centroid_to_id, 
                    dataset_name=threshold_type,
                    subset=subset,
                    thresholds=thresholds,
                    type=self.calc_type
                )
            })

            weights.append(weight)

        print("All datasets loaded successfully.")

        '''THEY COME FROM THE USER (weights for each inputed parameter)'''
        # weights = [0.3, 0.3, 0.2, 0.2] # Weights for each parameter 
        # vi_2 = weight_calculation(gdf, wind_speed, centroid_to_id, dataset_name='wind',subset=subset,thresholds=[2, 3, 4, 4.5, 5],type='point')
        
        # **Check that weights sum to 1**
        total_weight = sum(weights)
        if  total_weight != 1:  # Allowing for floating-point precision issues
            raise ValueError(f"Total weights sum to {total_weight}, but they must sum to 1. Adjust the weights.") 

        vi = {k: sum([w * v[k] for w, v in zip(weights, list(weights_set.values()))]) for k in tqdm(list(weights_set.values())[0].keys())}
        print('vi calculated successfully.')

        # Save the combined vi weights to a pickle file for reuse
        with open(f'{file_path}/vi.pkl', 'wb') as f:
            pickle.dump(vi, f)

        # # Normalize the pairwise distance data (dki) for compatibility in analysis
        # normalized_dki = normalize_dki(dki)

        # # Save the normalized dki values to a pickle file for reuse
        # with open(f'{file_path}/normalized_dki.pkl', 'wb') as f:
        #     pickle.dump(normalized_dki, f)

        # Report total preprocessing time in minutes and seconds
        end_time = time.time()
        time_taken = end_time - str_time
        minutes = time_taken // 60
        seconds = time_taken % 60
        print(f'Time taken for preprocessing: {minutes} minutes and {seconds} seconds')

        return self.model_choice(gdf, vi, dki_matrix, fids)
    
    def model_AUG(self, gdf, vi, dki_matrix, fids):
        # Start timing the Augmecon process.
        str_time = time.time()

        # Run the multi-objective optimization using the loaded `vi` and `normalized_dki` with the `two_objective_model` function
        _, df = two_objective_model(vi.keys(), vi, dki_matrix, fids, c=self.C_number, u=self.N_size[1], l=self.N_size[0])

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

        file_path = self.results_abs_path / "AUG_results.geojson"
        # Save filtered results to GeoJSON files for further use or visualization
        filtered_AUG_gdf.to_file(file_path, driver='GeoJSON')

        # Calculate time taken for the Augmecon optimization step
        end_time = time.time()
        time_taken = end_time - str_time
        minutes = time_taken // 60
        seconds = time_taken % 60
        print(f'Time taken for Augmencon2: {minutes} minutes and {seconds} seconds')
        return file_path

    def model_WS(self, gdf, vi, dki_matrix, fids , lambda_value = 0.5):
        # Begin timing for the Weighted Sum optimization
        str_time = time.time()

        print('Weighted Sum...')
        
        print(f'Loading numpy file for lambda={lambda_value}...')

        # Execute the Weighted Sum optimization function `WS` with current lambda value
        model = WS(vi.keys(), vi, dki_matrix,fids, lambda_value=lambda_value)

        # Sort results for easier analysis and map labels/interest values to the results
        model = model.sort_values(by='k1')
        model['Description'] = assign_labels(model)
        model['interest value'] = model['i1'].map(vi)  

        # Extract unique fid values for filtering purposes
        model_object_ids = model['i1'].unique()

        # Merge descriptions back into the GeoDataFrame to prepare for filtering
        gdf_with_desc = gdf.merge(model[['i1', 'Description', 'interest value']], left_on='fid', right_on='i1', how='left')

        # Filter the GeoDataFrame based on OBJECTIDs from the weighted sum results
        filtered_gdf = gdf_with_desc[gdf_with_desc['fid'].isin(model_object_ids)]
        
        file_path = self.results_abs_path / f"WS_results_{lambda_value}.geojson"
        # Save filtered results to GeoJSON files, named according to the lambda value
        filtered_gdf.to_file(file_path, driver='GeoJSON')

        # Calculate and print total time taken for the weighted sum optimization
        end_time = time.time()
        time_taken = end_time - str_time
        minutes = time_taken // 60
        seconds = time_taken % 60
        print(f'Time taken for Weighted Sum: {minutes} minutes and {seconds} seconds')
        return file_path

    def model_PF(self, gdf, vi):
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
        
        file_path = self.results_abs_path / "PF_results.geojson"
        # Save filtered results to GeoJSON files
        rem_particles.to_file(file_path, driver='GeoJSON')

        # Calculate and print total time taken for the particle filtering optimization
        end_time = time.time()
        time_taken = end_time - str_time
        minutes = time_taken // 60
        seconds = time_taken % 60
        print(f'Time taken for Particle Filtering: {minutes} minutes and {seconds} seconds')
        return file_path

if __name__ == '__main__':
    # replace abs path with a more dynamic one
    script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
    test_input_path = script_dir / "test_input" / "test_input.json"
    with open(test_input_path, 'r') as file:
        parsed_input = json.load(file)
    a = MSP4GROWTH_load_data(request_id=parsed_input["request_id"],
                            input_polygon=parsed_input["input_polygon"] ,
                            use_case=parsed_input["use_case"] ,
                            model=parsed_input["model"] ,
                            C_number=parsed_input["C_number"],
                            N_size=parsed_input["N_size"] ,
                            datasets=parsed_input.get("datasets") ,
                            thresholds= parsed_input["thresholds"], 
                            calc_type = parsed_input.get("calc_type"))

    geojsons = a.run_analysis()
    print(geojsons)