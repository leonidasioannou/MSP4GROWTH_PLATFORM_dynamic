#!/bin/bash
# Script to set up symbolic links to make data accessible to the application

# Define paths
PYTHON_PACKAGE_DIR="/home/developer/.local/lib/python3.10/site-packages"
WORKSPACE_DIR="/home/developer/workspace"
DATA_DIR="${WORKSPACE_DIR}/data"
PACKAGE_DATA_DIR="${PYTHON_PACKAGE_DIR}/data"

# Create data directory in workspace if it doesn't exist
mkdir -p ${DATA_DIR}

# Create sample data files if they don't exist
for FILE in "Habitat_map_clean.geojson" "Bathymetry.geojson" "Coastline.geojson" "Ports.geojson" "WindSpeed.geojson"; do
    if [ ! -f "${DATA_DIR}/${FILE}" ]; then
        echo "Creating sample ${FILE} in ${DATA_DIR}"
        # Copy the sample file content
        cat > "${DATA_DIR}/${FILE}" << 'EOF'
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "fid": 1,
        "name": "Area 1",
        "raster_value": -50
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [3670000, 4110000],
            [3675000, 4110000],
            [3675000, 4115000],
            [3670000, 4115000],
            [3670000, 4110000]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "fid": 2,
        "name": "Area 2",
        "raster_value": -60
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [3675000, 4110000],
            [3680000, 4110000],
            [3680000, 4115000],
            [3675000, 4115000],
            [3675000, 4110000]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "fid": 3,
        "name": "Area 3",
        "raster_value": 3.5
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [3670000, 4115000],
            [3675000, 4115000],
            [3675000, 4120000],
            [3670000, 4120000],
            [3670000, 4115000]
          ]
        ]
      }
    },
    {
      "type": "Feature",
      "properties": {
        "fid": 4,
        "name": "Area 4",
        "raster_value": 4.0
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [3675000, 4115000],
            [3680000, 4115000],
            [3680000, 4120000],
            [3675000, 4120000],
            [3675000, 4115000]
          ]
        ]
      }
    }
  ]
}
EOF
    fi
done

# Create data directory in Python package directory if it doesn't exist
if [ ! -d "${PACKAGE_DATA_DIR}" ]; then
    echo "Creating data directory in Python package: ${PACKAGE_DATA_DIR}"
    mkdir -p ${PACKAGE_DATA_DIR}
fi

# Create symbolic links to make data files accessible from package directory
echo "Creating symbolic links from ${DATA_DIR} to ${PACKAGE_DATA_DIR}"
for FILE in "${DATA_DIR}"/*; do
    FILENAME=$(basename "${FILE}")
    if [ ! -e "${PACKAGE_DATA_DIR}/${FILENAME}" ]; then
        echo "Creating symbolic link for ${FILENAME}"
        ln -sf "${FILE}" "${PACKAGE_DATA_DIR}/${FILENAME}"
    fi
done

echo "Symbolic links setup complete!"
