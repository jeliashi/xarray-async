"""
This MVP is designed to showcase how to read consolidated
zarr stores with xarray on a single thread.
This is not meant to be ready for use and should be used at your risk
The point of this is to use the fsspec filesystems in a single thread and not a seperate one.

Another important note here is that this doesn't utilize dask but rather the event loop used
during python runtime. The usefulness here is for when there is a service that needs to open
many different zarr stores concurrently without needing a threadpool. This would be more useful
for ASGI applications.
"""
import asyncio

import s3fs
import xarray as xr

from src.fsspec.mapping.mapper import AsyncFSMap
from src.xarray.backends.zarr import AsyncZarrBackendEntrypint

entry_point = AsyncZarrBackendEntrypint()


async def get_ds(path: str, fs) -> xr.Dataset:
    mapper = AsyncFSMap(path, fs)
    return await entry_point.open_dataset(mapper)


async def main():
    fs = s3fs.S3FileSystem(asynchronous=True)
    ds = await get_ds("s3://sensible-prod-weather/noaa/gefs/zarr/20230725/18/2_m_above_ground/TMP/", fs)
    return await ds._sel(lat=39.5, lon=-104.99, method="nearest")

if __name__ == "__main__":
    ds = asyncio.run(main())
    print(ds)