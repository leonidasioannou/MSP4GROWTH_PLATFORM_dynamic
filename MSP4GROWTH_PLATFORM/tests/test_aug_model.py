import pytest
import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path

from msp4growth.core.models.aug_model import run_augmecon_model

@pytest.fixture
def simple_gdf():
    # Create a simple GeoDataFrame with 4 points
    df = pd.DataFrame({
        'fid': [1, 2, 3, 4],
        'geometry': [
            # Points in a square
            gpd.points_from_xy([0], [0])[0],
            gpd.points_from_xy([1], [0])[0],
            gpd.points_from_xy([0], [1])[0],
            gpd.points_from_xy([1], [1])[0]
        ]
    })
    return gpd.GeoDataFrame(df, geometry='geometry')

@pytest.fixture
def simple_vi():
    # Importance values for each fid
    return {1: 0.5, 2: 1.0, 3: 0.2, 4: 0.8}

@pytest.fixture
def simple_dki():
    # 4x4 distance matrix
    coords = [(0,0),(1,0),(0,1),(1,1)]
    mat = np.zeros((4,4))
    for i, (x1, y1) in enumerate(coords):
        for j, (x2, y2) in enumerate(coords):
            mat[i,j] = np.hypot(x2-x1, y2-y1)
    return mat

@pytest.fixture
def fids_array():
    return np.array([1,2,3,4])

def test_run_augmecon_creates_file(tmp_path, simple_gdf, simple_vi, simple_dki, fids_array):
    # Prepare a temporary results directory
    results_dir = tmp_path / "results"
    # Run model
    out_path = run_augmecon_model(
        gdf=simple_gdf,
        vi=simple_vi,
        dki_matrix=simple_dki,
        fids=fids_array,
        c=2,
        u=2,
        l=1,
        results_dir=results_dir
    )
    # Assert file was created
    assert out_path.exists(), f"Expected output file at {out_path}"
    assert out_path.suffix == ".geojson"

    # Load and check GeoJSON
    out_gdf = gpd.read_file(out_path)
    # Check that 'Description' and 'interest value' columns exist
    assert 'Description' in out_gdf.columns
    assert 'interest value' in out_gdf.columns
    # Ensure geometries count <= number of features
    assert len(out_gdf) <= len(simple_gdf)

@pytest.mark.parametrize("c,u,l", [
    (1, 1, 1),
    (2, 2, 1),
])
def test_parameters_affect_selection(tmp_path, simple_gdf, simple_vi, simple_dki, fids_array, c, u, l):
    results_dir = tmp_path / "res2"
    out_path = run_augmecon_model(
        gdf=simple_gdf,
        vi=simple_vi,
        dki_matrix=simple_dki,
        fids=fids_array,
        c=c,
        u=u,
        l=l,
        results_dir=results_dir
    )
    out_gdf = gpd.read_file(out_path)
    # c is max centroids: number of unique centers in output <= c
    unique_centroids = out_gdf['Description'].nunique()
    assert unique_centroids <= c
if __name__ == "__main__":
    import pytest
    pytest.main([__file__])
