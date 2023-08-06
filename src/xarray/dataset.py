import asyncio
from typing import Any, Hashable, Iterable, Mapping

from xarray import Dataset as XDs
from xarray.core.indexes import isel_indexes
from xarray.core.indexing import is_fancy_indexer, map_index_queries
from xarray.core.utils import drop_dims_from_indexers, either_dict_or_kwargs


class Dataset(XDs):
    async def _isel(
        self,
        indexers: Mapping[Any, Any] | None = None,
        drop: bool = False,
        missing_dims="raise",
        **indexers_kwargs: Any,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "isel")
        if any(is_fancy_indexer(idx) for idx in indexers.values()):
            return self._isel_fancy(indexers, drop=drop, missing_dims=missing_dims)

        # Much faster algorithm for when all indexers are ints, slices, one-dimensional
        # lists, or zero or one-dimensional np.ndarray's
        indexers = drop_dims_from_indexers(indexers, self.dims, missing_dims)

        variables = {}
        dims: dict[Hashable, int] = {}
        coord_names = self._coord_names.copy()

        indexes, index_variables = isel_indexes(self.xindexes, indexers)

        async def place_var(name, var):
            if name in index_variables:
                var = index_variables[name]
            else:
                var_indexers = {k: v for k, v in indexers.items() if k in var.dims}
                if var_indexers:
                    if hasattr(var, "_isel"):
                        var = await var._isel(var_indexers)
                    else:
                        var = var.isel(var_indexers)
                    if drop and var.ndim == 0 and name in coord_names:
                        coord_names.remove(name)
                        return
            variables[name] = var
            dims.update(zip(var.dims, var.shape))

        await asyncio.gather(
            *[place_var(name, var) for name, var in self._variables.items()]
        )
        # preserve variable order
        # if name in index_variables:
        #     var = index_variables[name]
        # else:
        #     var_indexers = {k: v for k, v in indexers.items() if k in var.dims}
        #     if var_indexers:
        #         var = var.isel(var_indexers)
        #         if drop and var.ndim == 0 and name in coord_names:
        #             coord_names.remove(name)
        #             continue
        # variables[name] = var
        # dims.update(zip(var.dims, var.shape))

        return self._construct_direct(
            variables=variables,
            coord_names=coord_names,
            dims=dims,
            attrs=self._attrs,
            indexes=indexes,
            encoding=self._encoding,
            close=self._close,
        )

    async def _sel(
        self,
        indexers: Mapping[Any, Any] = None,
        method: str = None,
        tolerance: int | float | Iterable[int | float] | None = None,
        drop: bool = False,
        **indexers_kwargs: Any,
    ):
        indexers = either_dict_or_kwargs(indexers, indexers_kwargs, "sel")
        query_results = map_index_queries(
            self, indexers=indexers, method=method, tolerance=tolerance
        )

        if drop:
            no_scalar_variables = {}
            for k, v in query_results.variables.items():
                if v.dims:
                    no_scalar_variables[k] = v
                else:
                    if k in self._coord_names:
                        query_results.drop_coords.append(k)
            query_results.variables = no_scalar_variables

        result = await self._isel(indexers=query_results.dim_indexers, drop=drop)
        return result._overwrite_indexes(*query_results.as_tuple()[1:])
