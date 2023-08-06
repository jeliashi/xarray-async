import asyncio
import contextlib
import os
import sys

import numpy as np
from xarray import conventions
from xarray.backends.common import _decode_variable_name, _normalize_path
from xarray.backends.store import StoreBackendEntrypoint
from xarray.backends.zarr import (
    DIMENSION_KEY,
    ZarrArrayWrapper,
    ZarrBackendEntrypoint,
    ZarrStore,
    _get_zarr_dims_and_attrs,
)
from xarray.core import indexing
from xarray.core.utils import FrozenDict

from ..conventions import decode_cf_variable
from ..core.variable import Variable
from ..dataset import Dataset
from ...zarr.convenience import open_consolidated

sys.modules["xarray.conventions"].decode_cf_variable = decode_cf_variable


class AsyncArrayWrapper(ZarrArrayWrapper):
    async def __array__(self, dtype=None):
        key = indexing.BasicIndexer((slice(None),) * self.ndim)
        return np.asarray(await self[key], dtype=dtype)

    async def __getitem__(self, key):
        array = self.get_array()
        if isinstance(key, indexing.BasicIndexer):
            return await array[key.tuple]
        elif isinstance(key, indexing.VectorizedIndexer):
            return await array.vindex[
                indexing._arrayize_vectorized_indexer(key, self.shape).tuple
            ]
        else:
            assert isinstance(key, indexing.OuterIndexer)
            return await array.oindex[key.tuple]


class AsyncStore(ZarrStore):
    @classmethod
    async def open_group(
        cls,
        store,
        mode="r",
        synchronizer=None,
        group=None,
        consolidated=False,
        consolidate_on_close=False,
        chunk_store=None,
        storage_options=None,
        append_dim=None,
        write_region=None,
        safe_chunks=True,
        stacklevel=2,
    ):
        if isinstance(store, os.PathLike):
            raise NotImplementedError("cannot do local storage zarr")

        open_kwargs = dict(
            mode=mode,
            synchronizer=synchronizer,
            path=group,
        )
        open_kwargs["storage_options"] = storage_options

        if chunk_store:
            open_kwargs["chunk_store"] = chunk_store
            if consolidated is None:
                consolidated = False

        if consolidated is None:
            try:
                zarr_group = await open_consolidated(store, **open_kwargs)
            except KeyError:
                raise NotImplementedError("not ready for non-consolidated stores")
        elif consolidated:
            # TODO: an option to pass the metadata_key keyword
            zarr_group = await open_consolidated(store, **open_kwargs)
        else:
            raise NotImplementedError("not ready for non-consolidated stores")
        return cls(
            zarr_group,
            mode,
            consolidate_on_close,
            append_dim,
            write_region,
            safe_chunks,
        )

    async def load(self):
        variables = FrozenDict(
            (_decode_variable_name(k), v)
            for k, v in (await self.get_variables()).items()
        )
        attributes = FrozenDict(self.get_attrs())
        return variables, attributes

    async def get_variables(self):
        return FrozenDict(
            await asyncio.gather(
                *[
                    self.open_store_variable_with_key(k, v, k)
                    for k, v in self.zarr_group.arrays()
                ]
            )
        )

    async def open_store_variable_with_key(self, name, zarr_array, label):
        return label, await self.open_store_variable(name, zarr_array)

    async def open_store_variable(self, name, zarr_array):
        data = AsyncArrayWrapper(name, self)
        try_nczarr = self._mode == "r"
        dimensions, attributes = _get_zarr_dims_and_attrs(
            zarr_array, DIMENSION_KEY, try_nczarr
        )
        attributes = dict(attributes)
        encoding = {
            "chunks": zarr_array.chunks,
            "preferred_chunks": dict(zip(dimensions, zarr_array.chunks)),
            "compressor": zarr_array.compressor,
            "filters": zarr_array.filters,
        }
        # _FillValue needs to be in attributes, not encoding, so it will get
        # picked up by decode_cf
        if getattr(zarr_array, "fill_value") is not None:
            attributes["_FillValue"] = zarr_array.fill_value

        variable = Variable(dimensions, data, attributes, encoding)
        await variable.maybe_preload()
        return variable


class AsyncStoreBackendEntrypoint(StoreBackendEntrypoint):
    async def open_dataset(
        self,
        store,
        *,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
    ):
        vars, attrs = await store.load()
        encoding = store.get_encoding()

        vars, attrs, coord_names = conventions.decode_cf_variables(
            vars,
            attrs,
            mask_and_scale=mask_and_scale,
            decode_times=decode_times,
            concat_characters=concat_characters,
            decode_coords=decode_coords,
            drop_variables=drop_variables,
            use_cftime=use_cftime,
            decode_timedelta=decode_timedelta,
        )

        ds = Dataset(vars, attrs=attrs)
        ds = ds.set_coords(coord_names.intersection(vars))
        ds.set_close(store.close)
        ds.encoding = encoding

        return ds


class AsyncZarrBackendEntrypint(ZarrBackendEntrypoint):
    async def open_dataset(
        self,
        filename_or_obj,
        mask_and_scale=True,
        decode_times=True,
        concat_characters=True,
        decode_coords=True,
        drop_variables=None,
        use_cftime=None,
        decode_timedelta=None,
        group=None,
        mode="r",
        synchronizer=None,
        consolidated=None,
        chunk_store=None,
        storage_options=None,
        stacklevel=3,
    ):
        filename_or_obj = _normalize_path(filename_or_obj)
        store = await AsyncStore.open_group(
            filename_or_obj,
            group=group,
            mode=mode,
            synchronizer=synchronizer,
            consolidated=consolidated,
            consolidate_on_close=False,
            chunk_store=chunk_store,
            storage_options=storage_options,
            stacklevel=stacklevel + 1,
        )

        store_entrypoint = AsyncStoreBackendEntrypoint()
        with close_on_error(store):
            ds = await store_entrypoint.open_dataset(
                store,
                mask_and_scale=mask_and_scale,
                decode_times=decode_times,
                concat_characters=concat_characters,
                decode_coords=decode_coords,
                drop_variables=drop_variables,
                use_cftime=use_cftime,
                decode_timedelta=decode_timedelta,
            )
        return ds


@contextlib.contextmanager
def close_on_error(f):
    """Context manager to ensure that a file opened by xarray is closed if an
    exception is raised before the user sees the file object.
    """
    try:
        yield
    except Exception:
        f.close()
        raise
