from asyncio import iscoroutinefunction
from typing import Any

import numpy as np
from xarray.core import variable


class Variable(variable.Variable):
    async def maybe_preload(self):
        if self.size < 1e5 and iscoroutinefunction(getattr(self._data, "__array__")):
            self._data = await self._data.__array__()

    @property
    def data(self) -> Any:
        return self._data

    @data.setter
    def data(self, data):
        data = variable.as_compatible_data(data)
        if data.shape != self.shape:
            raise ValueError(
                f"replacement data must match the Variable's shape. "
                f"replacement data has shape {data.shape}; Variable has shape {self.shape}"
            )
        self._data = data

    async def _isel(
        self,
        indexers: variable.Mapping[Any, Any] = None,
        missing_dims="raise",
        **indexers_kwargs: Any,
    ):
        indexers = variable.either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        indexers = variable.drop_dims_from_indexers(indexers, self.dims, missing_dims)

        key = tuple(indexers.get(dim, slice(None)) for dim in self.dims)
        return await self.__agetitem__(key)

    async def __agetitem__(self, key):
        """Return a new Variable object whose contents are consistent with
        getting the provided key from the underlying data.

        NB. __getitem__ and __setitem__ implement xarray-style indexing,
        where if keys are unlabeled arrays, we index the array orthogonally
        with them. If keys are labeled array (such as Variables), they are
        broadcasted with our usual scheme and then the array is indexed with
        the broadcasted key, like numpy's fancy indexing.

        If you really want to do indexing like `x[x > 0]`, manipulate the numpy
        array `x.values` directly.
        """
        dims, indexer, new_order = self._broadcast_indexes(key)
        data = await self._data[indexer]
        if new_order:
            data = np.moveaxis(data, range(len(new_order)), new_order)
        return self._finalize_indexing_result(dims, data)
