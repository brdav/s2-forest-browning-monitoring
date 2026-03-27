"""Transpose Swisstopo NDVI/NDSI dataset from (T, N) to (N, T) layout using Dask."""

import dask.array as da
from dask.distributed import Client, LocalCluster

from forest_browning.config import (
    DASK_LOCAL_DIRECTORY,
    SPATIAL_DATASET_ZARR,
    TEMPORAL_DATASET_ZARR,
)


def transpose_zarr(source_zarr, target_zarr, component="ndvi"):
    """Transpose a Zarr dataset from (T, N) to (N, T) layout."""
    src = da.from_zarr(source_zarr, component=component)
    T, N = src.shape

    # Transpose and rechunk
    dst = src.T.rechunk(chunks=(4000, T))

    dst.to_zarr(target_zarr, component=component, overwrite=True, compute=True)


if __name__ == "__main__":
    with LocalCluster(
        n_workers=8,
        threads_per_worker=1,
        memory_limit="10GB",
        local_directory=DASK_LOCAL_DIRECTORY,
    ) as cluster:
        with Client(cluster) as client:
            print(f"Dashboard available at: {client.dashboard_link}")

            transpose_zarr(SPATIAL_DATASET_ZARR, TEMPORAL_DATASET_ZARR, "ndvi")
            transpose_zarr(SPATIAL_DATASET_ZARR, TEMPORAL_DATASET_ZARR, "ndsi")
