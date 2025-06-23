"""
Core data loader and analysis module for MSP4GROWTH.
"""
import os
import time
import logging
import pickle
import geopandas as gpd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from shapely.geometry import Polygon, shape
import datetime
from shapely.ops import transform
from functools import partial
from ..config import RESULTS_DIR, DATA_DIR
from .models.aug_model import run_augmecon_model
from .models.ws_model import run_weighted_sum_model
from .models.pf_model import run_particle_filter_model
from .utils.geo_utils import load_gdf, load_dataset, weight_calculation, normalize_dki_matrix
import numpy as np
from shapely.geometry import Polygon
from .report_generator import MSP4GROWTHReportGenerator
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MSP4GROWTHAnalysis:
    """
    Main class for MSP4GROWTH data loading and analysis.
    Handles loading data, preprocessing, and running optimization models.
    """
    
    def __init__(
        self,
        request_id: str,
        input_polygon: List[List[float]],
        use_case: str,
        model: Dict[str, Any],
        C_number: int,
        N_size: List[int],
        datasets: Optional[Dict[str, Any]] = None,
        thresholds: Dict[str, Any] = None,
        calc_type: str = "dist",
        thread_config: Optional[Dict[str, int]] = None,
        generate_report: bool = True
    ):
        """
        Initialize the analysis with configuration parameters.
        
        Args:
            request_id: Unique identifier for this analysis request
            input_polygon: List of coordinates defining the area of interest
            use_case: Type of use case (e.g., "aquaculture")
            model: Dictionary of models to run with their parameters
            C_number: C parameter for optimization models
            N_size: List containing lower and upper N size constraints [min, max]
            datasets: Optional dictionary of additional datasets
            thresholds: Dictionary of thresholds for each dataset
            calc_type: Type of calculation ("dist" or "point")
            thread_config: Thread configuration parameters
            generate_report: Whether to generate PDF report automatically
        """
        self.request_id = request_id
        self.input_polygon = input_polygon
        self.use_case = use_case
        self.model = model
        self.datasets = datasets or {}
        self.thresholds = thresholds or {}
        self.calc_type = calc_type
        self.C_number = C_number
        self.N_size = N_size
        
        # Set up results directory
        self.results_dir = RESULTS_DIR / str(self.request_id)
        self.results_dir.mkdir(exist_ok=True, parents=True)

        # Add report generation flag
        self.generate_report = generate_report
    
        # Apply thread configuration if provided
        if thread_config:
            for key, value in thread_config.items():
                if key == "threads":
                    continue  # Already set in Gurobi env
                env_var = f"{key.upper()}"
                os.environ[env_var] = str(value)
                
        logger.info(f"Initialized MSP4GROWTHAnalysis for request {request_id}")
    
    def run_analysis(self) -> Dict[str, Path]:
        """
        Run the full analysis pipeline:
        1. Load and preprocess data
        2. Calculate weights and distances
        3. Run selected optimization models
        4. Generate PDF report

        Returns:
            Dictionary of model names to result file paths and report path
        """
        start_time = time.time()
        
        # # Create polygon from coordinates
        # aoi_4326 = Polygon(self.input_polygon)
        # logger.info(f"Area of Interest (AOI) created with {len(aoi_4326.exterior.coords)} coordinates")
        # #         # Set up the transformation
        # wgs84 = pyproj.CRS('EPSG:4326')
        # web_mercator = pyproj.CRS('EPSG:3857')
        # project = pyproj.Transformer.from_crs(wgs84, web_mercator, always_xy=True).transform
        
        # # # Transform the polygon
        # aoi = transform(project, aoi_4326)
        aoi = Polygon(self.input_polygon)
        # Load habitat data
        gdf, centroid_to_id, dki_matrix, fids = self._load_habitat_data(aoi)
        logger.info(f"Loaded {len(gdf)} habitat data records successfully")
        
        # Load and process all datasets
        weights_set, weights = self._process_datasets(gdf, centroid_to_id, aoi)
        
        # Calculate combined weights
        vi = self._calculate_combined_weights(weights_set, weights)
        
        # Save preprocessed data
        self._save_preprocessed_data(vi)
        
        # Run optimization models
        result_paths = self._run_optimization_models(gdf, vi, dki_matrix, fids)
        # Generate report if requested
        report_path = None
        if self.generate_report and result_paths:
            try:
                logger.info("Generating PDF report...")
                report_generator = MSP4GROWTHReportGenerator(self)
                report_path = report_generator.generate_report(result_paths)
                
                # Clean up temporary files
                report_generator.cleanup()
                
                logger.info(f"PDF report generated successfully: {report_path}")
            except Exception as e:
                logger.error(f"Failed to generate PDF report: {str(e)}")
                # Continue execution even if report generation fails
        # Log completion time
        end_time = time.time()
        minutes = (end_time - start_time) // 60
        seconds = (end_time - start_time) % 60
        logger.info(f"Analysis completed in {minutes} minutes and {seconds:.2f} seconds")
        
        return {'model_results': result_paths, 'report_path': report_path}
    
    def _load_habitat_data(self, aoi: Polygon) -> Tuple[gpd.GeoDataFrame, Dict, np.ndarray, np.ndarray]:
        """
        Load habitat mapping data and preprocess for analysis.
        
        Args:
            aoi: Area of interest polygon
            
        Returns:
            Tuple containing:
            - GeoDataFrame of habitat data
            - Dictionary mapping centroids to IDs
            - Distance matrix
            - Array of feature IDs
        """
        try:
            habitat_path = DATA_DIR / "Habitat_map_clean.geojson"
            gdf, centroid_to_id, dki_matrix, fids = load_gdf(
                str(habitat_path),
                aoi=aoi,
                output_folder=self.results_dir
            )
            return gdf, centroid_to_id, dki_matrix, fids
        except Exception as e:
            logger.error(f"Error loading habitat data: {str(e)}")
            raise
    
    def _process_datasets(
        self, 
        gdf: gpd.GeoDataFrame, 
        centroid_to_id: Dict, 
        aoi: Polygon
    ) -> Tuple[Dict, List[float]]:
        """
        Process all datasets and calculate weights.
        
        Args:
            gdf: Habitat GeoDataFrame
            centroid_to_id: Mapping from centroids to IDs
            aoi: Area of interest
            
        Returns:
            Tuple containing:
            - Dictionary of dataset weights
            - List of weight values
        """
        datasets = {}
        weights_set = {}
        weights = []
        
        # Process each threshold type
        for threshold_type in self.thresholds.keys():
            # Load dataset based on whether it's a standard or custom dataset
            if threshold_type not in ['WindSpeed', 'Bathymetry', 'Coastline', 'Ports']:
                if threshold_type in self.datasets:
                    # Load custom dataset from provided data
                    logger.info(f"Loading custom dataset for {threshold_type}")
                    features = self.datasets[threshold_type]["geom"]["features"]
                    geometry = [shape(feature["geometry"]) for feature in features]
                    properties = [feature["properties"] for feature in features]
                    dataset_gdf = gpd.GeoDataFrame(properties, geometry=geometry)
                    dataset_gdf.set_crs(epsg=3857, inplace=True)
                    datasets[threshold_type] = dataset_gdf
                else:
                    logger.warning(f"No dataset provided for {threshold_type}")
                    continue
            else:
                # Load standard dataset from file
                dataset_path = DATA_DIR / f"{threshold_type}.geojson"
                datasets[threshold_type] = load_dataset(str(dataset_path))
            
            # Validate thresholds and weights
            try:
                thresholds = self.thresholds[threshold_type]["thresholds"]
                if not isinstance(thresholds, list) or len(thresholds) != 5:
                    raise ValueError(f"'thresholds' for '{threshold_type}' must be a list of exactly 5 elements")
            except KeyError:
                raise KeyError(f"Missing 'thresholds' key for '{threshold_type}'")

            try:
                weight = self.thresholds[threshold_type]["weights"]
                if not isinstance(weight, (int, float)):
                    raise ValueError(f"'weights' for '{threshold_type}' must be a numeric value")
            except KeyError:
                raise KeyError(f"Missing 'weights' key for '{threshold_type}'")
            
            # Calculate weights for this dataset
            weights_set[threshold_type] = weight_calculation(
                gdf, 
                datasets[threshold_type], 
                centroid_to_id, 
                dataset_name=threshold_type,
                subset=bool(aoi),
                thresholds=thresholds,
                type=self.calc_type
            )
            
            weights.append(weight)
        
        # Validate that weights sum to 1
        total_weight = sum(weights)
        if abs(total_weight - 1) > 1e-9:  # Allow for small floating-point errors
            raise ValueError(f"Total weights sum to {total_weight}, but they must sum to 1")
            
        return weights_set, weights
    
    def _calculate_combined_weights(self, weights_set: Dict, weights: List[float]) -> Dict:
        """
        Calculate combined weights from individual dataset weights.
        
        Args:
            weights_set: Dictionary of dataset weights
            weights: List of weight values
            
        Returns:
            Dictionary of combined weights
        """
        # Calculate weighted average of all weight sets
        sample_keys = list(list(weights_set.values())[0].keys())
        vi = {}
        
        for k in sample_keys:
            vi[k] = sum(w * weight_set.get(k, 0) for w, weight_set in zip(weights, weights_set.values()))
            
        logger.info("Combined weights calculated successfully")
        return vi
    
    def _save_preprocessed_data(self, vi: Dict):
        """
        Save preprocessed data for later use or analysis.
        
        Args:
            vi: Combined weights dictionary
        """
        # Save vi to pickle file
        vi_path = self.results_dir / "vi.pkl"
        with open(vi_path, 'wb') as f:
            pickle.dump(vi, f)
            
        logger.info(f"Preprocessed data saved to {self.results_dir}")
    
    def _run_optimization_models(
        self, 
        gdf: gpd.GeoDataFrame, 
        vi: Dict, 
        dki_matrix: np.ndarray,
        fids: np.ndarray
    ) -> Dict[str, Path]:
        """
        Run requested optimization models.
        
        Args:
            gdf: Habitat GeoDataFrame
            vi: Combined weights dictionary
            dki_matrix: Distance matrix
            fids: Feature IDs array
            
        Returns:
            Dictionary mapping model names to result file paths
        """
        resulting_geojsons = {}
        
        if not self.model:
            raise ValueError("Invalid model choice. Please choose from 'Augment', 'WS', or 'PF'.")
            
        # Run AUG model if requested
        if "model_AUG" in self.model and self.model['model_AUG'] > 0:
            try:
                logger.info("Starting AUG model")
                aug_results = run_augmecon_model(
                    gdf, vi, dki_matrix, fids, 
                    c=self.C_number, 
                    u=self.N_size[1], 
                    l=self.N_size[0],
                    results_dir=self.results_dir
                )
                resulting_geojsons["model_AUG"] = aug_results
                logger.info(f"AUG model completed, results saved to {aug_results}")
            except Exception as e:
                logger.error(f"Failed to produce results for AUG model: {str(e)}")
                
        # Run WS model if requested
        if "model_WS" in self.model and self.model['model_WS'] > 0:
            # try:
            logger.info("Starting WS model")
            lambda_value = self.model['model_WS']
            ws_results = run_weighted_sum_model(
                            gdf, vi, dki_matrix,
                            fids,
                            lambda_value=lambda_value,
                            c=self.C_number, 
                            u=self.N_size[1], 
                            l=self.N_size[0],
                            results_dir=self.results_dir
                        )
            print(f"WS model returned: {ws_results}")
            resulting_geojsons["model_WS"] = ws_results
            logger.info(f"WS model completed, results saved to {ws_results}")
            # except Exception as e:
            #     logger.error(f"Failed to produce results for WS model: {str(e)}")
                
        # Run PF model if requested
        if "model_PF" in self.model and self.model['model_PF'] > 0:
            try:
                logger.info("Starting PF model")
                pf_results = run_particle_filter_model(
                    gdf, vi, 
                    results_dir=self.results_dir
                )
                resulting_geojsons["model_PF"] = pf_results
                logger.info(f"PF model completed, results saved to {pf_results}")
            except Exception as e:
                logger.error(f"Failed to produce results for PF model: {str(e)}")
                
        return resulting_geojsons
    
    # Add a separate method for generating reports from existing results
    def generate_report_from_results(self, result_paths: Dict[str, Path]) -> Optional[Path]:
        """
        Generate a PDF report from existing analysis results.
        
        Args:
            result_paths: Dictionary of model names to result file paths
            
        Returns:
            Path to generated PDF report or None if generation failed
        """
        try:
            logger.info("Generating PDF report from existing results...")
            report_generator = MSP4GROWTHReportGenerator(self)
            report_path = report_generator.generate_report(result_paths)
            
            # Clean up temporary files
            report_generator.cleanup()
            
            logger.info(f"PDF report generated successfully: {report_path}")
            return report_path
            
        except Exception as e:
            logger.error(f"Failed to generate PDF report: {str(e)}")
            return None
