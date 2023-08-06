from fsspec.asyn import AsyncFileSystem
from fsspec.mapping import FSMap, maybe_convert


class AsyncFSMap(FSMap):
    def __init__(self, root, fs, check=False, create=False, missing_exceptions=None):
        assert isinstance(fs, AsyncFileSystem)
        super().__init__(root, fs, check, create, missing_exceptions)

    async def clear(self):
        try:
            await self.fs._rm(self.root, True)
            await self.fs._mkdir(self.root)
        except Exception:
            pass

    async def getitems(self, keys, on_error="raise"):
        keys2 = [self._key_to_str(k) for k in keys]
        oe = on_error if on_error == "raise" else "return"
        try:
            out = await self.fs._cat(keys2, on_error=oe)
            if isinstance(out, bytes):
                out = {keys2[0]: out}
        except self.missing_exceptions as e:
            raise KeyError from e
        out = {
            k: (KeyError() if isinstance(v, self.missing_exceptions) else v)
            for k, v in out.items()
        }
        return {
            key: out[k2]
            for key, k2 in zip(keys, keys2)
            if on_error == "return" or not isinstance(out[k2], BaseException)
        }

    async def setitems(self, values_dict):
        values = {self._key_to_str(k): maybe_convert(v) for k, v in values_dict.items()}
        await self.fs._pipe(values)

    async def delitems(self, keys):
        """Remove multiple keys from the store"""
        await self.fs._rm([self._key_to_str(k) for k in keys])

    async def __getitem__(self, key, default=None):
        """Retrieve data"""
        k = self._key_to_str(key)
        try:
            result = await self.fs._cat(k)
        except self.missing_exceptions:
            if default is not None:
                return default
            raise KeyError(key)
        return result

    async def pop(self, key, default=None):
        result = await self.__getitem__(key, default)
        try:
            del self[key]
        except KeyError:
            pass
        return result

    async def __setitem__(self, key, value):
        """Store value in key"""
        key = self._key_to_str(key)
        await self.fs._makedirs(self.fs._parent(key), exist_ok=True)
        await self.fs._pipe_file(key, maybe_convert(value))

    async def __iter__(self):
        return (self._str_to_key(x) for x in await self.fs._find(self.root))

    async def __len__(self):
        return len(self.fs._find(self.root))

    async def __delitem__(self, key):
        """Remove key"""
        try:
            await self.fs._rm(self._key_to_str(key))
        except:  # noqa: E722
            raise KeyError

    async def __contains__(self, key):
        """Does key exist in mapping?"""
        path = self._key_to_str(key)
        return await self.fs._exists(path) and await self.fs._isfile(path)
