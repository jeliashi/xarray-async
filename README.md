# MVP for Async Zarr with Xarray
## Why?
Opening zarr stores using fsspec is already very optimized because fsspec mappers utilizes a background thread that does all of the calls asynchronously. This is then wrapped in a synchronous function.

There remains a challenge however in trying to open multiple stores within an event loop. A user would need to spin up a threadpool and open/retrieve data in multiple threads. A general usecase for this would be an ASGI application (like FastAPI) that needs to serve requests to multiple concurrent virtual users.

Therefore, this solution offers a way to access stores with 0 context switching between threads and creates opening and reading zarr stores completely non-blocking when waiting on IO.

## What was changed?
- Introduced a new fsspec mapper that has async methods using the fsspec.asyn.AsyncFileSystem
- changed zarr reading to be done with async functions in a very brittle monkeypatch way
- added a backend entrypoint for xarray which reads the zarr store using async directly
- added async _isel and _sel methods to xarray.Dataset to allow for non-blocking data filtering.

## Notes:
The purpose of this project is to spur discussion about how to allow datasets to be accessed with a minimal no context switches in an async framework
There is already async functionality in the fsspec project by introducing the `getitems` method to mapper objects but this project takes it a step further and exposes an async API through the entire chain of fsspec-zarr-xarray.

This intentionally does not use `dask` because I'm not comfortable enough with dask to extend the needed `dask.array` components to make this project single-threaded.  
