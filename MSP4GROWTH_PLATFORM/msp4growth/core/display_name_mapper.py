"""
Display name mapping system for MSP4GROWTH reports.
Maps technical code words to user-friendly display names.
"""

class DisplayNameMapper:
    """
    Handles mapping of technical code words to display-friendly names.
    """
    
    def __init__(self):
        # Define the mapping dictionary
        self.display_names = {
            # Column names
            'interest value': 'Suitability Score',
            'suitability': 'Suitability Score',
            'fid': 'Feature ID',
            'area_km2': 'Area (km²)',
            'Area': 'Area (km²)',
            'objective_value': 'Objective Value',
            'geometry': 'Geometry',
            
            # Dataset names
            'WindSpeed': 'Wind Speed',
            'Bathymetry': 'Water Depth',
            'Coastline': 'Distance to Coast',
            'Ports': 'Distance to Ports',
            'shipping_routes': 'Shipping Routes',
            'protected_areas': 'Protected Areas',
            'Aquaculture': 'Aquaculture Sites',
            
            # Model names
            'model_AUG': 'MOIP - Augmented ε-Constraint',
            'model_WS': 'MOIP - Weighted Sum',
            'model_PF': 'Particle Filter',
            'AUG': 'MOIP - Augmented ε-Constraint',
            'WS': 'MOIP - Weighted Sum',
            'PF': 'Particle Filter',
            'PF Model': 'Particle Filter Model',
            'AUG Model': 'MOIP - Augmented ε-Constraint Model',
            'WS Model': 'MOIP - Weighted Sum Model',
            
            # Use cases
            'aquaculture': 'Development of Aquaculture Facilities',
            'offshore_wind': 'Offshore Wind Energy',
            'marine_protected': 'Marine Protected Areas',
            'shipping': 'Shipping and Navigation',
            'fisheries': 'Fisheries Management',
            'tourism': 'Marine Tourism',
            'oil_gas': 'Oil & Gas Exploration',
            'renewable_energy': 'Renewable Energy',
            
            # Technical terms
            'calc_type': 'Calculation Method',
            'dist': 'Distance-based',
            'point': 'Point-based',
            'thresholds': 'Threshold Values',
            'weights': 'Weight Coefficients',
            'C_number': 'Number of Optimal Sites (C)',
            'N_size': 'Size of Optimal Sites (N)',
            'request_id': 'Analysis ID',
            
            # Units and measures
            'meters': 'meters',
            'kilometers': 'kilometers',
            'km': 'km',
            'km2': 'km²',
            'degrees': '°',
            'percent': '%',
            'knots': 'knots',
            'db': 'dB',
            'mg_l': 'mg/L',
            'm_s': 'm/s',
            
            # Status and quality indicators
            'high': 'High',
            'medium': 'Medium',
            'low': 'Low',
            'very_high': 'Very High',
            'very_low': 'Very Low',
            'excellent': 'Excellent',
            'good': 'Good',
            'fair': 'Fair',
            'poor': 'Poor',
            'suitable': 'Suitable',
            'unsuitable': 'Unsuitable',
            'restricted': 'Restricted',
            'prohibited': 'Prohibited'
        }
    
    def get_display_name(self, code_name: str) -> str:
        """
        Get the display name for a code word.
        
        Args:
            code_name: The technical code name
            
        Returns:
            User-friendly display name
        """
        return self.display_names.get(code_name, self._format_fallback(code_name))
    
    def _format_fallback(self, code_name: str) -> str:
        """
        Format unknown code names into readable text.
        
        Args:
            code_name: The code name to format
            
        Returns:
            Formatted display name
        """
        # Handle None or empty strings
        if not code_name:
            return "N/A"
        
        # Convert to string if not already
        code_name = str(code_name)
        
        # Replace underscores and hyphens with spaces
        formatted = code_name.replace('_', ' ').replace('-', ' ')
        
        # Capitalize each word
        formatted = ' '.join(word.capitalize() for word in formatted.split())
        
        return formatted
    
    def format_column_names(self, df_columns: list) -> dict:
        """
        Create a mapping for DataFrame column renaming.
        
        Args:
            df_columns: List of DataFrame column names
            
        Returns:
            Dictionary for pandas rename operation
        """
        return {col: self.get_display_name(col) for col in df_columns}
    
    def add_custom_mapping(self, mappings: dict):
        """
        Add custom mappings to the display names.
        
        Args:
            mappings: Dictionary of code_name: display_name pairs
        """
        self.display_names.update(mappings)