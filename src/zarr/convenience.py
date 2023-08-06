import sys

from zarr.convenience import StoreLike, open
from zarr.storage import normalize_store_arg

from .core import Array
from .storage import ConsolidatedMetadataStore

sys.modules["zarr.core"].Array = Array
sys.modules["zarr.hierarchy"].Array = Array


async def open_consolidated(
    store: StoreLike, metadata_key=".zmetadata", mode="r+", **kwargs
):
    zarr_version = kwargs.get("zarr_version")
    store = normalize_store_arg(
        store,
        storage_options=kwargs.get("storage_options"),
        mode=mode,
        zarr_version=zarr_version,
    )
    if mode not in {"r", "r+"}:
        raise ValueError(
            "invalid mode, expected either 'r' or 'r+'; found {!r}".format(mode)
        )

    path = kwargs.pop("path", None)
    if store._store_version == 2:
        ConsolidatedStoreClass = ConsolidatedMetadataStore
    else:
        raise NotImplementedError()
        # assert_zarr_v3_api_available()
        # ConsolidatedStoreClass = ConsolidatedMetadataStoreV3
        # # default is to store within 'consolidated' group on v3
        # if not metadata_key.startswith("meta/root/"):
        #     metadata_key = "meta/root/consolidated/" + metadata_key

    # setup metadata store
    meta_store = ConsolidatedStoreClass(store, metadata_key=metadata_key)
    await meta_store.init_coro

    # pass through
    chunk_store = kwargs.pop("chunk_store", None) or store
    return open(
        store=meta_store, chunk_store=chunk_store, mode=mode, path=path, **kwargs
    )
