from __future__ import annotations

import numpy as np

import pandas as pd
from pandas._libs import lib
from pandas.core.dtypes.dtypes import (
    ExtensionDtype,
    register_extension_dtype,
)
from pandas.core.arrays import ExtensionArray
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.cast import find_common_type
from pandas.core.indexers import unpack_tuple_and_ellipses
from pandas._typing import DtypeObj, SortKind

import modin.pandas as mpd
from pandas.core.arraylike import OpsMixin


class ModinDtype(ExtensionDtype):
    # _wrapped_dtype can be any numpy or ExtensionDtype *except* for another
    #  ModinDtype.
    _wrapped_dtype: DtypeObj

    def __init__(self, dtype):
        dtype = pandas_dtype(dtype)
        if isinstance(dtype, ModinDtype):
            dtype = dtype._wrapped_dtype

        self._wrapped_dtype = dtype

    @classmethod
    def construct_array_type(cls):
        return ModinExtensionArray

    def __eq__(self, other) -> bool:
        if isinstance(other, ModinDtype):
            return other._wrapped_dtype == self._wrapped_dtype
        if isinstance(other, str):
            try:
                other = self.construct_from_string(other)
            except TypeError:
                return False
            return self.__eq__(other)
        return NotImplemented

    def __hash__(self) -> int:
        tup = (type(self), self._wrapped_dtype)
        return hash(tup)

    @property
    def name(self) -> str:
        dtype = self._wrapped_dtype
        return f"ModinDtype[{dtype}]"

    @property
    def type(self):
        return self._wrapped_dtype.type

    @property
    def kind(self) -> str:
        return self._wrapped_dtype.kind

    @property
    def _is_numeric(self) -> bool:
        dtype = self._wrapped_dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind in "iufcb"
        return dtype._is_numeric

    @property
    def _is_boolean(self) -> bool:
        dtype = self._wrapped_dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind == "b"
        return dtype._is_boolean

    @property
    def _can_hold_na(self) -> bool:
        dtype = self._wrapped_dtype
        if isinstance(dtype, np.dtype):
            return dtype.kind not in "iubUS"
        return dtype._can_hold_na

    def _get_common_dtype(self, dtypes: list[DtypeObj]) -> DtypeObj | None:
        dtypes = [x._wrapped_dtype if isinstance(x, ModinDtype) else x for x in dtypes]

        common = find_common_type(dtypes)
        return ModinDtype(common)

    @classmethod
    def construct_from_string(cls, string: str) -> ModinDtype:
        if string.startswith("ModinDtype[") and string.endswith("]"):
            string = string[11:-1]
        else:
            raise TypeError(f"Cannot interpet {string} as ModinDtype")

        dtype = pandas_dtype(string)
        return cls(dtype)


register_extension_dtype(ModinDtype)


def wrap_modin_series_op(opname):
    def meth(self, *args, **kwargs):
        res = getattr(self._modin_ser, opname)(*args, **kwargs)
        if isinstance(res, mpd.Series):
            return type(self)(res)
        return res

    meth.__name__ = opname
    return meth


class ModinExtensionArray(OpsMixin, ExtensionArray):
    _modin_ser: mpd.Series

    def __init__(self, ser: mpd.Series):
        # Make sure ser is unwrapped; we don't want to recurse
        assert not isinstance(ser.dtype, ModinDtype)
        assert isinstance(ser, mpd.Series), type(ser)
        self._modin_ser = ser

    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        if isinstance(dtype, ModinDtype):
            dtype = dtype._wrapped_dtype

        assert not isinstance(scalars, mpd.Series)
        if isinstance(scalars, ModinExtensionArray):
            if dtype is None:
                if copy is False:
                    return scalars[:]
                return scalars.copy()
            return scalars.astype(dtype, copy=copy)

        ser = mpd.Series(scalars, dtype=dtype, copy=copy)
        return cls(ser)

    def __getitem__(self, key):
        if isinstance(key, tuple):
            key = unpack_tuple_and_ellipses(key)
        res = self._modin_ser.iloc[key]
        if isinstance(key, int):
            return res
        return type(self)(res)

    def __setitem__(self, key, value):
        # This is super-inefficient; we do the setting first on a copy
        #  to make sure it doesn't alter the dtype.
        ser = self._modin_ser
        obj = ser.copy()
        obj.iloc[key] = value
        if obj.dtype != self.dtype:
            raise ValueError(f"Cannot set value into array with dtype={self.dtype}")
        ser.iloc[key] = value

    def __iter__(self):
        ser = self._modin_ser
        for i in range(len(ser)):
            yield ser.iloc[i]

    def _cmp_method(self, other, op):
        if isinstance(other, ModinExtensionArray):
            # TODO: probably need to align rather than assert
            assert other._modin_ser.index.equals(self._modin_ser.index)
            other = other._modin_ser
        res = op(self._modin_ser, other)
        return type(self)(res)

    _arith_method = _cmp_method

    @property
    def dtype(self) -> ModinDtype:
        return ModinDtype(self._modin_ser.dtype)

    @property
    def nbytes(self) -> int:
        return self._modin_ser.nbytes

    __array__ = wrap_modin_series_op("__array__")
    __len__ = wrap_modin_series_op("__len__")
    to_numpy = wrap_modin_series_op("to_numpy")
    isna = wrap_modin_series_op("isna")
    fillna = wrap_modin_series_op("fillna")
    any = wrap_modin_series_op("any")
    all = wrap_modin_series_op("all")
    __invert__ = wrap_modin_series_op("__invert__")
    searchsorted = wrap_modin_series_op("searchsorted")
    factorize = wrap_modin_series_op("factorize")

    def astype(self, dtype, copy=True):
        dtype = pandas_dtype(dtype)
        if dtype == self.dtype:
            if not copy:
                return self
            return self.copy()

        res = self._modin_ser.astype(dtype, copy=copy)
        if dtype == object:
            return np.asarray(res)
        return type(self)(res)

    def unique(self):
        # TODO: once GH#6227 is fixed this can just use wrap_modin_series_op
        res = self._modin_ser.unique()
        res = type(self)._from_sequence(res, dtype=self.dtype)
        return res

    def copy(self):
        # once GH#6219 is fixed this can just use wrap_modin_series_op 
        ser = self._modin_ser
        res = ser.copy().astype(ser.dtype, copy=False)
        return type(self)(res)

    def equals(self, other) -> bool:
        if type(self) != type(other):
            return False
        ser = self._modin_ser
        oser = other._modin_ser
        if not ser.index.equals(oser.index):
            raise NotImplementedError("Need to reindex")
        return bool(ser.equals(oser))

    def value_counts(self, dropna: bool = True):
        return self._modin_ser.value_counts(dropna=dropna)._to_pandas()

    def take(self, indices, allow_fill: bool = False, fill_value=None):
        ser = self._modin_ser
        if allow_fill:
            ser.index = range(len(ser))
            res = ser.reindex(indices, fill_value=fill_value)
        else:
            res = ser.take(indices)
        return type(self)(res)

    def _reduce(self, name: str, *, skipna: bool = True, **kwargs):
        return getattr(self._modin_ser, name)(skipna=skipna, **kwargs)

    @classmethod
    def _concat_same_type(cls, to_concat):
        assert len(to_concat) > 0
        first = to_concat[0]
        assert all(isinstance(x, cls) for x in to_concat)
        assert all(x.dtype == first.dtype for x in to_concat)

        sers = [x._modin_ser for x in to_concat]
        res = mpd.concat(sers, axis=0, ignore_index=True)
        return cls(res)
