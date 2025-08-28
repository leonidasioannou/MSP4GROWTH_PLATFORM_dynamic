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
from datetime import datetime as dt
from pathlib import Path
from functools import partial
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from scipy.spatial.distance import cdist
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

def find_closest_stations(gdf, stations_df, n_closest=2):
    """
    Find the n closest meteorological stations for each grid cell
    
    Parameters:
    gdf: GeoDataFrame with grid cells (EPSG:3857)
    stations_df: DataFrame with station data (must have 'Latitude', 'Longitude', 'Station' columns)
    n_closest: number of closest stations to find (default: 2)
    
    Returns:
    Dictionary with cell_id as key and list of closest stations as value
    """
    
    # Get unique stations with their coordinates
    unique_stations = stations_df[['Station', 'Latitude', 'Longitude']].drop_duplicates('Station')
    
    # Create points from station coordinates (WGS84 lat/lon)
    station_points = [Point(row['Longitude'], row['Latitude']) for _, row in unique_stations.iterrows()]
    station_gdf = gpd.GeoDataFrame(unique_stations, geometry=station_points, crs='EPSG:4326')
    
    # Reproject stations to match grid CRS (EPSG:3857)
    station_gdf = station_gdf.to_crs(gdf.crs)
    
    # Get centroids of grid cells
    cell_centroids = gdf.geometry.centroid
    
    # Convert to arrays for distance calculation
    cell_coords = np.array([[geom.x, geom.y] for geom in cell_centroids])
    station_coords = np.array([[geom.x, geom.y] for geom in station_gdf.geometry])
    
    # Calculate distances between all cells and stations
    distances = cdist(cell_coords, station_coords)
    
    # Find closest stations for each cell
    cell_station_mapping = {}
    
    for i, cell_id in enumerate(gdf['fid']):  # Using 'fid' as the cell identifier
        # Get indices of n closest stations
        closest_indices = np.argsort(distances[i])[:n_closest]
        
        # Get station names and distances
        closest_stations = []
        for idx in closest_indices:
            station_info = {
                'station_name': unique_stations.iloc[idx]['Station'],
                'distance_km': distances[i][idx] / 1000,  # Convert to km (distance in meters for EPSG:3857)
                'coordinates': {
                    'lat': unique_stations.iloc[idx]['Latitude'],
                    'lon': unique_stations.iloc[idx]['Longitude']
                }
            }
            closest_stations.append(station_info)
        
        cell_station_mapping[str(cell_id)] = closest_stations
    
    return cell_station_mapping

def calculate_station_averages(stations_df, meteorological_columns):
    """
    Calculate average values for each meteorological variable per station
    
    Parameters:
    stations_df: DataFrame with station meteorological data
    meteorological_columns: list of column names containing meteorological variables
    
    Returns:
    Dictionary with station averages
    """
    
    # Group by station and calculate means
    station_averages = {}
    
    for station_name in stations_df['Station'].unique():
        station_data = stations_df[stations_df['Station'] == station_name]
        
        averages = {}
        for col in meteorological_columns:
            if col in station_data.columns:
                # Calculate mean, excluding NaN values
                avg_value = station_data[col].mean()
                averages[col] = float(avg_value) if not pd.isna(avg_value) else None
        
        station_averages[station_name] = {
            'coordinates': {
                'lat': station_data['Latitude'].iloc[0],
                'lon': station_data['Longitude'].iloc[0]
            },
            'averages': averages,
            'data_points': len(station_data)
        }
    
    return station_averages

def get_season(month):
    """
    Classify months into seasons (Northern Hemisphere)
    """
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def get_time_of_day(hour):
    """
    Classify hours into time periods
    """
    if 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    elif 18 <= hour < 24:
        return 'Evening'
    else:
        return 'Night'

def calculate_wind_statistics(wind_speed, wind_direction):
    """
    Calculate comprehensive wind statistics
    """
    wind_stats = {}
    
    # Basic wind speed statistics
    wind_stats['speed'] = {
        'mean': round(wind_speed.mean(), 2),
        'median': round(wind_speed.median(), 2),
        'std': round(wind_speed.std(), 2),
        'min': round(wind_speed.min(), 2),
        'max': round(wind_speed.max(), 2),
        'percentile_25': round(wind_speed.quantile(0.25), 2),
        'percentile_75': round(wind_speed.quantile(0.75), 2)
    }
    
    # Wind speed categories (Beaufort scale simplified)
    wind_categories = pd.cut(wind_speed, 
                           bins=[0, 0.3, 1.6, 3.4, 5.5, 8.0, 10.8, float('inf')],
                           labels=['Calm', 'Light Air', 'Light Breeze', 'Gentle Breeze', 
                                  'Moderate Breeze', 'Fresh Breeze', 'Strong Breeze+'])
    wind_stats['categories'] = wind_categories.value_counts().to_dict()
    
    # Predominant wind directions
    if not wind_direction.isna().all():
        # Convert degrees to cardinal directions
        def deg_to_cardinal(deg):
            directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                         'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
            idx = round(deg / 22.5) % 16
            return directions[idx]
        
        cardinal_directions = wind_direction.apply(deg_to_cardinal)
        wind_stats['predominant_directions'] = cardinal_directions.value_counts().head(5).to_dict()
        wind_stats['mean_direction'] = round(wind_direction.mean(), 1)
    
    return wind_stats

def comprehensive_station_analysis(stations_df):
    """
    Perform comprehensive analysis for each meteorological station
    """

    # Create a copy to avoid modifying the original dataframe
    stations_df = stations_df.copy()

    # Prepare datetime column
    stations_df['datetime'] = pd.to_datetime(stations_df['Timestamp'])
    stations_df['year'] = stations_df['datetime'].dt.year
    stations_df['month'] = stations_df['datetime'].dt.month
    stations_df['day'] = stations_df['datetime'].dt.day
    stations_df['hour'] = stations_df['datetime'].dt.hour
    stations_df['season'] = stations_df['month'].apply(get_season)
    stations_df['time_of_day'] = stations_df['hour'].apply(get_time_of_day)
    stations_df['day_of_year'] = stations_df['datetime'].dt.dayofyear
    
    # Identify meteorological columns
    excluded_cols = ['Station', 'Latitude', 'Longitude', 'Timestamp', 'datetime', 
                     'year', 'month', 'day', 'hour', 'season', 'time_of_day', 'day_of_year']
    meteorological_columns = [col for col in stations_df.columns if col not in excluded_cols]
    
    comprehensive_analysis = {}
    
    for station_name in stations_df['Station'].unique():
        print(f"Analyzing station: {station_name}")
        station_data = stations_df[stations_df['Station'] == station_name].copy()
        
        analysis = {
            'station_info': {
                'name': station_name,
                'coordinates': {
                    'lat': station_data['Latitude'].iloc[0],
                    'lon': station_data['Longitude'].iloc[0]
                },
                'data_period': {
                    'start_date': station_data['datetime'].min().strftime('%Y-%m-%d %H:%M:%S'),
                    'end_date': station_data['datetime'].max().strftime('%Y-%m-%d %H:%M:%S'),
                    'total_records': len(station_data),
                    'years_covered': sorted([int(x) for x in station_data['year'].unique().tolist()])
                }
            }
        }
        
        # === OVERALL STATISTICS ===
        overall_stats = {}
        for col in meteorological_columns:
            if col in station_data.columns and not station_data[col].isna().all():
                stats = {
                    'mean': round(station_data[col].mean(), 2),
                    'median': round(station_data[col].median(), 2),
                    'std': round(station_data[col].std(), 2),
                    'min': round(station_data[col].min(), 2),
                    'max': round(station_data[col].max(), 2),
                    'percentile_25': round(station_data[col].quantile(0.25), 2),
                    'percentile_75': round(station_data[col].quantile(0.75), 2),
                    'missing_values': int(station_data[col].isna().sum()),
                    'missing_percentage': round(station_data[col].isna().mean() * 100, 1)
                }
                overall_stats[col] = stats
        
        analysis['overall_statistics'] = overall_stats
        
        # === SEASONAL ANALYSIS ===
        seasonal_analysis = {}
        for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
            season_data = station_data[station_data['season'] == season]
            if len(season_data) > 0:
                seasonal_stats = {}
                for col in meteorological_columns:
                    if col in season_data.columns and not season_data[col].isna().all():
                        seasonal_stats[col] = {
                            'mean': round(season_data[col].mean(), 2),
                            'min': round(season_data[col].min(), 2),
                            'max': round(season_data[col].max(), 2),
                            'records': len(season_data)
                        }
                seasonal_analysis[season] = seasonal_stats
        
        analysis['seasonal_analysis'] = seasonal_analysis
        
        # === MONTHLY PATTERNS ===
        monthly_patterns = {}
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        for month in range(1, 13):
            month_data = station_data[station_data['month'] == month]
            if len(month_data) > 0:
                month_name = month_names[month - 1]
                monthly_stats = {}
                for col in meteorological_columns:
                    if col in month_data.columns and not month_data[col].isna().all():
                        monthly_stats[col] = round(month_data[col].mean(), 2)
                monthly_patterns[month_name] = monthly_stats
        
        analysis['monthly_patterns'] = monthly_patterns
        
        # === DAILY PATTERNS (by hour) ===
        daily_patterns = {}
        for hour in range(24):
            hour_data = station_data[station_data['hour'] == hour]
            if len(hour_data) > 0:
                hourly_stats = {}
                for col in meteorological_columns:
                    if col in hour_data.columns and not hour_data[col].isna().all():
                        hourly_stats[col] = round(hour_data[col].mean(), 2)
                daily_patterns[f"{hour:02d}:00"] = hourly_stats
        
        analysis['daily_patterns'] = daily_patterns
        
        # === TIME OF DAY ANALYSIS ===
        time_of_day_analysis = {}
        for time_period in ['Morning', 'Afternoon', 'Evening', 'Night']:
            period_data = station_data[station_data['time_of_day'] == time_period]
            if len(period_data) > 0:
                period_stats = {}
                for col in meteorological_columns:
                    if col in period_data.columns and not period_data[col].isna().all():
                        period_stats[col] = {
                            'mean': round(period_data[col].mean(), 2),
                            'min': round(period_data[col].min(), 2),
                            'max': round(period_data[col].max(), 2)
                        }
                time_of_day_analysis[time_period] = period_stats
        
        analysis['time_of_day_analysis'] = time_of_day_analysis
        
        # === WIND ANALYSIS ===
        if 'Wind Speed (m/s)' in station_data.columns and 'Wind Direction (Deg.)' in station_data.columns:
            wind_stats = calculate_wind_statistics(
                station_data['Wind Speed (m/s)'].dropna(),
                station_data['Wind Direction (Deg.)'].dropna()
            )
            analysis['wind_analysis'] = wind_stats
        
        # === PRECIPITATION ANALYSIS ===
        if 'Rain (mm)' in station_data.columns:
            rain_data = station_data['Rain (mm)'].dropna()
            if len(rain_data) > 0:
                precipitation_stats = {
                    'total_rainfall': round(rain_data.sum(), 2),
                    'rainy_hours': int((rain_data > 0).sum()),
                    'rainy_hours_percentage': round((rain_data > 0).mean() * 100, 1),
                    'average_rainfall_when_raining': round(rain_data[rain_data > 0].mean(), 2) if (rain_data > 0).any() else 0,
                    'max_hourly_rainfall': round(rain_data.max(), 2),
                    'seasonal_rainfall': {}
                }
                
                # Seasonal rainfall totals
                for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                    season_rain = station_data[station_data['season'] == season]['Rain (mm)'].dropna()
                    if len(season_rain) > 0:
                        precipitation_stats['seasonal_rainfall'][season] = round(season_rain.sum(), 2)
                
                analysis['precipitation_analysis'] = precipitation_stats
        
        # === TEMPERATURE ANALYSIS ===
        temp_cols = ['Temperature (°C)', 'Max Temperature (°C)', 'Min Temperature (°C)']
        available_temp_cols = [col for col in temp_cols if col in station_data.columns]
        
        if available_temp_cols:
            temperature_analysis = {}
            
            for col in available_temp_cols:
                temp_data = station_data[col].dropna()
                if len(temp_data) > 0:
                    temperature_analysis[col.replace(' (°C)', '')] = {
                        'annual_mean': round(temp_data.mean(), 2),
                        'absolute_min': round(temp_data.min(), 2),
                        'absolute_max': round(temp_data.max(), 2),
                        'seasonal_means': {}
                    }
                    
                    # Seasonal temperature means
                    for season in ['Winter', 'Spring', 'Summer', 'Autumn']:
                        season_temp = station_data[station_data['season'] == season][col].dropna()
                        if len(season_temp) > 0:
                            temperature_analysis[col.replace(' (°C)', '')]['seasonal_means'][season] = round(season_temp.mean(), 2)
            
            analysis['temperature_analysis'] = temperature_analysis
        
        # === EXTREME WEATHER EVENTS ===
        extreme_events = {}
        
        # Temperature extremes (if available)
        if 'Temperature (°C)' in station_data.columns:
            temp_data = station_data['Temperature (°C)'].dropna()
            if len(temp_data) > 0:
                temp_p95 = temp_data.quantile(0.95)
                temp_p5 = temp_data.quantile(0.05)
                
                extreme_events['temperature'] = {
                    'hot_days_above_95th_percentile': int((temp_data > temp_p95).sum()),
                    'cold_days_below_5th_percentile': int((temp_data < temp_p5).sum()),
                    'threshold_hot': round(temp_p95, 1),
                    'threshold_cold': round(temp_p5, 1)
                }
        
        # Wind extremes
        if 'Wind Speed (m/s)' in station_data.columns:
            wind_data = station_data['Wind Speed (m/s)'].dropna()
            if len(wind_data) > 0:
                wind_p95 = wind_data.quantile(0.95)
                extreme_events['wind'] = {
                    'high_wind_events_above_95th_percentile': int((wind_data > wind_p95).sum()),
                    'threshold_high_wind': round(wind_p95, 2)
                }
        
        # Precipitation extremes
        if 'Rain (mm)' in station_data.columns:
            rain_data = station_data['Rain (mm)'].dropna()
            if len(rain_data) > 0 and (rain_data > 0).any():
                rain_p95 = rain_data[rain_data > 0].quantile(0.95)
                extreme_events['precipitation'] = {
                    'heavy_rain_events_above_95th_percentile': int((rain_data > rain_p95).sum()),
                    'threshold_heavy_rain': round(rain_p95, 2)
                }
        
        analysis['extreme_weather_events'] = extreme_events
        
        # === DATA QUALITY ASSESSMENT ===
        data_quality = {
            'completeness': {},
            'temporal_coverage': {
                'hours_per_day': round(len(station_data) / len(station_data['day'].unique()), 1),
                'days_covered': len(station_data.groupby(['year', 'month', 'day'])),
                'months_covered': len(station_data.groupby(['year', 'month']))
            }
        }
        
        for col in meteorological_columns:
            if col in station_data.columns:
                completeness = (1 - station_data[col].isna().mean()) * 100
                data_quality['completeness'][col] = round(completeness, 1)
        
        analysis['data_quality'] = data_quality
        
        comprehensive_analysis[station_name] = analysis
    
    return comprehensive_analysis

def create_station_comparison_report(comprehensive_analysis):# Create a copy to avoid modifying the original dataframe
    """
    Create a comparative analysis report across all stations
    """
    comparison_report = {
        'summary': {
            'total_stations': len(comprehensive_analysis),
            'station_names': list(comprehensive_analysis.keys())
        },
        'comparative_statistics': {}
    }
    
    # Compare key metrics across stations
    metrics_to_compare = [
        'Temperature (°C)', 'Humidity (%)', 'Wind Speed (m/s)', 
        'Rain (mm)', 'Wind Direction (Deg.)'
    ]
    
    for metric in metrics_to_compare:
        station_values = {}
        for station_name, analysis in comprehensive_analysis.items():
            if metric in analysis.get('overall_statistics', {}):
                station_values[station_name] = analysis['overall_statistics'][metric]['mean']
        
        if station_values:
            comparison_report['comparative_statistics'][metric] = {
                'highest_station': max(station_values, key=station_values.get),
                'highest_value': max(station_values.values()),
                'lowest_station': min(station_values, key=station_values.get),
                'lowest_value': min(station_values.values()),
                'all_stations': station_values
            }
    
    return comparison_report

def generate_comprehensive_analysis(gdf, csv_path, output_dir=Path("Results")):
    """
    Main function to generate comprehensive meteorological analysis
    """
    print("=== COMPREHENSIVE METEOROLOGICAL ANALYSIS ===")
    print("Loading data...")
    
    # Load data
    stations_df = pd.read_csv(csv_path)
    
    print(f"Grid cells: {len(gdf)}")
    print(f"Station records: {len(stations_df)}")
    print(f"Unique stations: {stations_df['Station'].nunique()}")
    print(f"Date range: {stations_df['Timestamp'].min()} to {stations_df['Timestamp'].max()}")
    
    # Identify meteorological columns (exclude location and identifier columns)
    excluded_cols = ['Station', 'Latitude', 'Longitude', 'Timestamp']
    meteorological_columns = [col for col in stations_df.columns if col not in excluded_cols]
    print(f"Meteorological variables: {meteorological_columns}")
    
    # Set up results directory and cache file
    results_folder = Path("Results")
    results_folder.mkdir(exist_ok=True)
    cell_station_file = results_folder / "cell_station_mapping.json"
    comprehensive_station_analysis_file = results_folder / "comprehensive_station_analysis.json"
    station_averages_file = results_folder / "station_averages.json"
    create_station_comparison_file = results_folder / "station_comparison_report.json"
    station_summary_file = results_folder / "station_summary_report.json"

    # 1. Grid-Station Mapping
    print("\n1. Creating grid-station mapping...")
    
    # Check if cached values exist and are valid
    flag = False
    if cell_station_file.exists():
        try:
            with open(cell_station_file) as json_file:
                cell_station_mapping = json.load(json_file)
                logger.info(f"Loaded weights from cache: {cell_station_file}")
                
                # Check if cache is valid for current dataset
                if len(gdf) <= len(cell_station_mapping):
                    # Filter the cell_station_mapping dictionary to match gdf's fid values
                    cell_station_mapping = {
                        key: value
                        for key, value in cell_station_mapping.items()
                        if int(key) in gdf['fid'].values
                    }
                    flag = True
                else:
                    logger.warning("Cache has fewer entries than current dataset, recalculating")
        except Exception as e:
            logger.warning(f"Error loading cached weights: {str(e)}")
    
    # Calculate weights if not loaded from cache
    if not flag:
        cell_station_mapping = find_closest_stations(gdf, stations_df)
        # Save cell-station mapping
        with open(f"{output_dir}/cell_station_mapping.json", 'w') as f:
            json.dump(cell_station_mapping, f, indent=2)

    # 2. Comprehensive Station Analysis
    print("\n2. Performing comprehensive station analysis...")
    if comprehensive_station_analysis_file.exists():
        with open(comprehensive_station_analysis_file) as f:
            comprehensive_analysis = json.load(f)
    else:
        comprehensive_analysis = comprehensive_station_analysis(stations_df)
        # Save comprehensive analysis
        with open(f"{output_dir}/comprehensive_station_analysis.json", 'w') as f:
            json.dump(comprehensive_analysis, f, indent=2)
    
    # 3. Station Comparison Report
    print("\n3. Creating station comparison report...")
    if create_station_comparison_file.exists():
        with open(create_station_comparison_file) as f:
            comparison_report = json.load(f)
    else:
        comparison_report = create_station_comparison_report(comprehensive_analysis)
        # Save comparison report
        with open(f"{output_dir}/station_comparison_report.json", 'w') as f:
            json.dump(comparison_report, f, indent=2)

    # 4. Calculate station averages
    print("\n4. Calculating station averages...")
    if station_averages_file.exists():
        with open(station_averages_file) as f:
            station_averages = json.load(f)
    else:
        station_averages = calculate_station_averages(stations_df, meteorological_columns)
        # Save station averages
        with open(f"{output_dir}/station_averages.json", 'w') as f:
            json.dump(station_averages, f, indent=2)

    
    if station_summary_file.exists():
        with open(station_summary_file) as f:
            summary_stats = json.load(f)
    else:
        # Create summary report
        summary_stats = {}
        for station_name, analysis in comprehensive_analysis.items():
            summary_stats[station_name] = {
                'coordinates': analysis['station_info']['coordinates'],
                'data_records': analysis['station_info']['data_period']['total_records'],
                'temperature_mean': analysis['overall_statistics'].get('Temperature (°C)', {}).get('mean', 'N/A'),
                'humidity_mean': analysis['overall_statistics'].get('Humidity (%)', {}).get('mean', 'N/A'),
                'wind_speed_mean': analysis['overall_statistics'].get('Wind Speed (m/s)', {}).get('mean', 'N/A'),
                'total_rainfall': analysis.get('precipitation_analysis', {}).get('total_rainfall', 'N/A')
            }
        
        with open(f"{output_dir}/station_summary.json", 'w') as f:
            json.dump(summary_stats, f, indent=2)
    
    print("\n=== ANALYSIS COMPLETE ===")
    print(f"Files:")
    print(f"- {output_dir}/cell_station_mapping.json")
    print(f"- {output_dir}/station_averages.json")
    print(f"- {output_dir}/comprehensive_station_analysis.json")  
    print(f"- {output_dir}/station_comparison_report.json")
    print(f"- {output_dir}/station_summary.json")

    return cell_station_mapping, comprehensive_analysis, comparison_report

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