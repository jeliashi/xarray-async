import asyncio
import sys

import numpy as np
from numcodecs.compat import ensure_ndarray_like
from zarr.core import Array as ZA
from zarr.errors import err_too_many_indices
from zarr.indexing import (
    BasicIndexer,
    CoordinateIndexer,
    MaskIndexer,
    OrthogonalIndexer,
    check_fields,
    ensure_tuple,
    is_pure_fancy_indexing,
    pop_fields,
)
from zarr.util import check_array_shape

from .indexing import OIndex, VIndex

sys.modules["zarr.core"].OIndex = OIndex
sys.modules["zarr.core"].VIndex = VIndex


class Array(ZA):
    async def __array__(self, *args):
        a = await self[...]
        if args:
            a = a.astype(args[0])
        return a

    async def islice(self, start=None, end=None):
        if len(self.shape) == 0:
            # Same error as numpy
            raise TypeError("iteration over a 0-d array")
        if start is None:
            start = 0
        if end is None or end > self.shape[0]:
            end = self.shape[0]

        if not isinstance(start, int) or start < 0:
            raise ValueError("start must be a nonnegative integer")

        if not isinstance(end, int) or end < 0:
            raise ValueError("end must be a nonnegative integer")

        # Avoid repeatedly decompressing chunks by iterating over the chunks
        # in the first dimension.
        chunk_size = self.chunks[0]
        chunk = None
        for j in range(start, end):
            if j % chunk_size == 0:
                chunk = await self[j : j + chunk_size]
            # init chunk if we start offset of chunk borders
            elif chunk is None:
                chunk_start = j - j % chunk_size
                chunk_end = chunk_start + chunk_size
                chunk = await self[chunk_start:chunk_end]
            yield await chunk[j % chunk_size]

    async def __iter__(self):
        return await self.islice()

    async def __aiter__(self):
        return await self.islice()

    async def __getitem__(self, selection):
        fields, pure_selection = pop_fields(selection)
        if is_pure_fancy_indexing(pure_selection, self.ndim):
            result = await self.vindex[selection]
        else:
            result = await self.get_basic_selection(pure_selection, fields=fields)
        return result

    async def get_basic_selection(self, selection=Ellipsis, out=None, fields=None):
        if not self._cache_metadata:
            self._load_metadata()
        check_fields(fields, self._dtype)
        if self._shape == ():
            return await self._get_basic_selection_zd(
                selection=selection, out=out, fields=fields
            )
        else:
            return await self._get_basic_selection_nd(
                selection=selection, out=out, fields=fields
            )

    async def _get_basic_selection_zd(self, selection, out=None, fields=None):
        selection = ensure_tuple(selection)
        if selection not in ((), (Ellipsis,)):
            err_too_many_indices(selection, ())

        try:
            # obtain encoded data for chunk
            ckey = self._chunk_key((0,))
            cdata = await self.chunk_store[ckey]

        except KeyError:
            # chunk not initialized
            chunk = np.zeros_like(self._meta_array, shape=(), dtype=self._dtype)
            if self._fill_value is not None:
                chunk.fill(self._fill_value)

        else:
            chunk = self._decode_chunk(cdata)

        # handle fields
        if fields:
            chunk = chunk[fields]

        # handle selection of the scalar value via empty tuple
        if out is None:
            out = chunk[selection]
        else:
            out[selection] = chunk[selection]

        return out

    async def _get_basic_selection_nd(self, selection, out=None, fields=None):
        indexer = BasicIndexer(selection, self)

        return await self._get_selection(indexer=indexer, out=out, fields=fields)

    async def _get_selection(self, indexer, out=None, fields=None):
        out_dtype = check_fields(fields, self._dtype)
        out_shape = indexer.shape
        if out is None:
            out = np.empty_like(
                self._meta_array, shape=out_shape, dtype=out_dtype, order=self._order
            )
        else:
            check_array_shape("out", out, out_shape)
            # sequentially get one key at a time from storage
        await asyncio.gather(
            *[
                self._chunk_getitem(
                    chunk_coords,
                    chunk_selection,
                    out,
                    out_selection,
                    drop_axes=indexer.drop_axes,
                    fields=fields,
                )
                for chunk_coords, chunk_selection, out_selection in indexer
            ]
        )

        if out.shape:
            return out
        else:
            return out[()]

    async def _chunk_getitem(
        self,
        chunk_coords,
        chunk_selection,
        out,
        out_selection,
        drop_axes=None,
        fields=None,
    ):
        out_is_ndarray = True
        try:
            out = ensure_ndarray_like(out)
        except TypeError:
            out_is_ndarray = False

        assert len(chunk_coords) == len(self._cdata_shape)
        ckey = self._chunk_key(chunk_coords)
        try:
            # obtain compressed data for chunk
            cdata = await self.chunk_store[ckey]
        except KeyError:
            # chunk not initialized
            if self._fill_value is not None:
                if fields:
                    fill_value = self._fill_value[fields]
                else:
                    fill_value = self._fill_value
                out[out_selection] = fill_value

        else:
            self._process_chunk(
                out,
                cdata,
                chunk_selection,
                drop_axes,
                out_is_ndarray,
                fields,
                out_selection,
            )

    async def get_orthogonal_selection(self, selection, out=None, fields=None):
        if not self._cache_metadata:
            self._load_metadata()
        check_fields(fields, self._dtype)
        indexer = OrthogonalIndexer(selection, self)
        return await self._get_selection(indexer=indexer, out=out, fields=fields)

    async def set_orthogonal_selection(self, selection, value, fields=None):
        raise NotImplementedError()

    async def get_coordinate_selection(self, selection, out=None, fields=None):
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # setup indexer
        indexer = CoordinateIndexer(selection, self)

        # handle output - need to flatten
        if out is not None:
            out = out.reshape(-1)

        out = await self._get_selection(indexer=indexer, out=out, fields=fields)

        # restore shape
        out = out.reshape(indexer.sel_shape)

        return out

    async def get_mask_selection(self, selection, out=None, fields=None):
        if not self._cache_metadata:
            self._load_metadata()

        # check args
        check_fields(fields, self._dtype)

        # setup indexer
        indexer = MaskIndexer(selection, self)

        return await self._get_selection(indexer=indexer, out=out, fields=fields)
