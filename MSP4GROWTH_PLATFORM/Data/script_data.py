import os
import re
import pandas as pd

# Path to your folder containing the CSVs
folder_path = './Meteo_Station_data'

# All CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# Define a reference column structure (from one of the good files)
reference_file = os.path.join(folder_path, 'Achna_Meteo_data.csv')  # Replace with any known-good file
reference_df = pd.read_csv(reference_file)
standard_columns = reference_df.columns.tolist()

# List to hold all DataFrames
dataframes = []

# Define fixed coordinates for stations if needed
fixed_coords = {
    'Amathus': {'Latitude': 34.695, 'Longitude': 33.134}
    # Add more stations here if needed
}
def convert_mixed_timestamps(series):
    # Regex pattern for dd/mm/yyyy HH:MM:SS
    pattern = re.compile(r'^\d{2}/\d{2}/\d{4} \d{2}:\d{2}:\d{2}$')
    
    # Split into two masks
    match_ddmm = series.astype(str).str.match(pattern)
    
    # Create empty result series
    result = pd.Series(pd.NaT, index=series.index)
    
    # Convert dd/mm/yyyy formatted rows
    result[match_ddmm] = pd.to_datetime(series[match_ddmm], format='%d/%m/%Y %H:%M:%S', errors='coerce')
    
    # Convert the rest with automatic parsing
    result[~match_ddmm] = pd.to_datetime(series[~match_ddmm], errors='coerce')
    
    return result

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    station_name = file.replace('_Meteo_data.csv', '').replace('_Meto_data.csv', '')

    try:
        df = pd.read_csv(file_path)
        
        #drop "Rec. Light Work Load (%)","Rec. Medium Work Load (%)","Rec. Heavy Work Load (%)" columns if they exist
        df.drop(columns=['Rec. Light Work Load (%)', 'Rec. Medium Work Load (%)', 'Rec. Heavy Work Load (%)', 'Altitude','Node'], errors='ignore', inplace=True)

        # if column wind speed is in knots convert it to m/s
        if 'Wind Speed (Knots)' in df.columns:
            df['Wind Speed (m/s)'] = df['Wind Speed (Knots)'] * 0.514444
            df.drop(columns=['Wind Speed (Knots)'], inplace=True)
        
        df['Station'] = station_name

        # Inject fixed coordinates if available
        if station_name in fixed_coords:
            df['Latitude'] = fixed_coords[station_name]['Latitude']
            df['Longitude'] = fixed_coords[station_name]['Longitude']
        
        dataframes.append(df)

    except Exception as e:
        print(f"Error processing {file}: {e}")

# Merge all and drop empty columns
merged_df = pd.concat(dataframes, ignore_index=True)

# set type of columns
for col in merged_df.columns:
    if 'Timestamp' in col:
        merged_df[col] = convert_mixed_timestamps(merged_df[col])
        #merged_df[col] = pd.to_datetime(merged_df[col], format= '%d/%m/%Y %H:%M:%S', errors='coerce')
    elif 'Station' in col:
        merged_df[col] = merged_df[col].astype(str)
    else:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

# Optional: Save to CSV
output_file = os.path.join(folder_path, 'Merged_data.csv')
merged_df.to_csv(output_file, index=False)

print(f"Merged {len(dataframes)} files. Output saved to '{output_file}'")
