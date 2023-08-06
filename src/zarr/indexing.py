from zarr.errors import VindexInvalidSelectionError
from zarr.indexing import OIndex as ZO
from zarr.indexing import VIndex as ZV
from zarr.indexing import (
    ensure_tuple,
    is_coordinate_selection,
    is_mask_selection,
    pop_fields,
    replace_lists,
)


class OIndex(ZO):
    async def __getitem__(self, selection):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return await self.array.get_orthogonal_selection(selection, fields=fields)

    async def __setitem__(self, selection, value):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        return await self.array.set_orthogonal_selection(
            selection, value, fields=fields
        )


class VIndex(ZV):
    async def __getitem__(self, selection):
        fields, selection = pop_fields(selection)
        selection = ensure_tuple(selection)
        selection = replace_lists(selection)
        if is_coordinate_selection(selection, self.array):
            return await self.array.get_coordinate_selection(selection, fields=fields)
        elif is_mask_selection(selection, self.array):
            return await self.array.get_mask_selection(selection, fields=fields)
        else:
            raise VindexInvalidSelectionError(selection)
