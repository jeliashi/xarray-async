from xarray import conventions

from .core.variable import Variable


def decode_cf_variable(
    name,
    var,
    concat_characters=True,
    mask_and_scale=True,
    decode_times=True,
    decode_endianness=True,
    stack_char_dim=True,
    use_cftime=None,
    decode_timedelta=None,
):
    # Ensure datetime-like Variables are passed through unmodified (GH 6453)
    if conventions._contains_datetime_like_objects(var):
        return var

    original_dtype = var.dtype

    if decode_timedelta is None:
        decode_timedelta = decode_times

    if concat_characters:
        if stack_char_dim:
            var = conventions.strings.CharacterArrayCoder().decode(var, name=name)
        var = conventions.strings.EncodedStringCoder().decode(var)

    if mask_and_scale:
        for coder in [
            conventions.variables.UnsignedIntegerCoder(),
            conventions.variables.CFMaskCoder(),
            conventions.variables.CFScaleOffsetCoder(),
        ]:
            var = coder.decode(var, name=name)

    if decode_timedelta:
        var = conventions.times.CFTimedeltaCoder().decode(var, name=name)
    if decode_times:
        var = conventions.times.CFDatetimeCoder(use_cftime=use_cftime).decode(
            var, name=name
        )

    dimensions, data, attributes, encoding = conventions.variables.unpack_for_decoding(
        var
    )
    # TODO(shoyer): convert everything below to use coders

    if decode_endianness and not data.dtype.isnative:
        # do this last, so it's only done if we didn't already unmask/scale
        data = conventions.NativeEndiannessArray(data)
        original_dtype = data.dtype

    encoding.setdefault("dtype", original_dtype)

    if "dtype" in attributes and attributes["dtype"] == "bool":
        del attributes["dtype"]
        data = conventions.BoolTypeArray(data)

    return Variable(dimensions, data, attributes, encoding=encoding)
