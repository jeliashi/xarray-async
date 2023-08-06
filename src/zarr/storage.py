from zarr.errors import MetadataError
from zarr.storage import ConsolidatedMetadataStore as zCMS
from zarr.storage import KVStore, Store, StoreLike
from zarr.util import json_loads


class ConsolidatedMetadataStore(zCMS):
    def __init__(self, store: StoreLike, metadata_key=".zmetadata"):
        self.init_coro = self.ainit(store, metadata_key)

    async def ainit(self, store: StoreLike, metadata_key=".zmetadata"):
        self.store = Store._ensure_store(store)

        # retrieve consolidated metadata
        meta = json_loads(await self.store[metadata_key])

        # check format of consolidated metadata
        consolidated_format = meta.get("zarr_consolidated_format", None)
        if consolidated_format != 1:
            raise MetadataError(
                "unsupported zarr consolidated metadata format: %s"
                % consolidated_format
            )

        # decode metadata
        self.meta_store: Store = KVStore(meta["metadata"])
