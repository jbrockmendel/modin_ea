"""
This file contains a minimal set of tests for compliance with the extension
array interface test suite, and should contain no other tests.
The test suite for the full functionality of the array is located in
`pandas/tests/arrays/`.
The tests in this file are inherited from the BaseExtensionTests, and only
minimal tweaks should be applied to get the tests passing (by overwriting a
parent method).
Additional tests should either be added to one of the BaseExtensionTests
classes (if they are relevant for the extension interface for all dtypes), or
be added to the array-specific tests in `pandas/tests/arrays/`.
"""
import operator
from datetime import (
    date,
    datetime,
    time,
    timedelta,
)
from decimal import Decimal
from pandas.compat import pa_version_under7p0, pa_version_under8p0
import numpy as np
import pytest

import pandas as pd
import pandas._testing as tm
from pandas.tests.extension import base
from pandas.tests.extension.conftest import (
    as_series,
    as_frame,
    use_numpy,
    invalid_scalar,
    na_cmp,
    box_in_series,
    fillna_method,
    as_array,
    groupby_apply_op,
    data_repeated,
)
from pandas.conftest import (
    comparison_op,
    all_arithmetic_operators,
    all_boolean_reductions,
    all_numeric_reductions,
)

pa = pytest.importorskip("pyarrow", minversion="7.0.0")

from ..modin_ea import ModinDtype, ModinExtensionArray
import modin.pandas as mpd


pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:The default value of numeric_only:FutureWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:The `na_sentinel` argument of:DeprecationWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:.*for empty DataFrame is not currently supported by PandasOnRay:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:searchsorted is not currently supported by PandasOnRay:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:`Series.searchsorted` is not currently supported by PandasOnRay:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:`Series.factorize` is not currently supported by PandasOnRay:UserWarning"
    ),
    pytest.mark.filterwarnings(
        "ignore:<function Series.divmod> is not currently supported by PandasOnRay:UserWarning"
    ),
]


# TODO: tm.ALL_NUMPY_DTYPES, non-arrow EA dtypes
@pytest.fixture(params=tm.ALL_PYARROW_DTYPES[:1], ids=str) 
def dtype(request):
    pd_dtype = pd.ArrowDtype(pyarrow_dtype=request.param)
    return ModinDtype(pd_dtype)


@pytest.fixture
def data(dtype):
    pd_dtype = dtype._wrapped_dtype
    pa_dtype = pd_dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        data = [True, False] * 4 + [None] + [True, False] * 44 + [None] + [True, False]
    elif pa.types.is_floating(pa_dtype):
        data = [1.0, 0.0] * 4 + [None] + [-2.0, -1.0] * 44 + [None] + [0.5, 99.5]
    elif pa.types.is_signed_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [-2, -1] * 44 + [None] + [1, 99]
    elif pa.types.is_unsigned_integer(pa_dtype):
        data = [1, 0] * 4 + [None] + [2, 1] * 44 + [None] + [1, 99]
    elif pa.types.is_decimal(pa_dtype):
        data = (
            [Decimal("1"), Decimal("0.0")] * 4
            + [None]
            + [Decimal("-2.0"), Decimal("-1.0")] * 44
            + [None]
            + [Decimal("0.5"), Decimal("33.123")]
        )
    elif pa.types.is_date(pa_dtype):
        data = (
            [date(2022, 1, 1), date(1999, 12, 31)] * 4
            + [None]
            + [date(2022, 1, 1), date(2022, 1, 1)] * 44
            + [None]
            + [date(1999, 12, 31), date(1999, 12, 31)]
        )
    elif pa.types.is_timestamp(pa_dtype):
        data = (
            [datetime(2020, 1, 1, 1, 1, 1, 1), datetime(1999, 1, 1, 1, 1, 1, 1)] * 4
            + [None]
            + [datetime(2020, 1, 1, 1), datetime(1999, 1, 1, 1)] * 44
            + [None]
            + [datetime(2020, 1, 1), datetime(1999, 1, 1)]
        )
    elif pa.types.is_duration(pa_dtype):
        data = (
            [timedelta(1), timedelta(1, 1)] * 4
            + [None]
            + [timedelta(-1), timedelta(0)] * 44
            + [None]
            + [timedelta(-10), timedelta(10)]
        )
    elif pa.types.is_time(pa_dtype):
        data = (
            [time(12, 0), time(0, 12)] * 4
            + [None]
            + [time(0, 0), time(1, 1)] * 44
            + [None]
            + [time(0, 5), time(5, 0)]
        )
    elif pa.types.is_string(pa_dtype):
        data = ["a", "b"] * 4 + [None] + ["1", "2"] * 44 + [None] + ["!", ">"]
    elif pa.types.is_binary(pa_dtype):
        data = [b"a", b"b"] * 4 + [None] + [b"1", b"2"] * 44 + [None] + [b"!", b">"]
    else:
        raise NotImplementedError

    arr = pd.array(data, dtype=pd_dtype)
    pd_ser = pd.Series(arr)
    modin_ser = mpd.Series(pd_ser)
    modin_arr = ModinExtensionArray(modin_ser)
    return modin_arr


@pytest.fixture
def data_missing(data):
    """Length-2 array with [NA, Valid]"""
    return type(data)._from_sequence([None, data[0]], dtype=data.dtype)


@pytest.fixture(params=["data", "data_missing"])
def all_data(request, data, data_missing):
    """Parametrized fixture returning 'data' or 'data_missing' integer arrays.

    Used to test dtype conversion with and without missing values.
    """
    if request.param == "data":
        return data
    elif request.param == "data_missing":
        return data_missing


@pytest.fixture
def data_for_grouping(dtype):
    """
    Data for factorization, grouping, and unique tests.

    Expected to be like [B, B, NA, NA, A, A, B, C]

    Where A < B < C and NA is missing
    """
    pa_dtype = dtype._wrapped_dtype.pyarrow_dtype
    if pa.types.is_boolean(pa_dtype):
        A = False
        B = True
        C = True
    elif pa.types.is_floating(pa_dtype):
        A = -1.1
        B = 0.0
        C = 1.1
    elif pa.types.is_signed_integer(pa_dtype):
        A = -1
        B = 0
        C = 1
    elif pa.types.is_unsigned_integer(pa_dtype):
        A = 0
        B = 1
        C = 10
    elif pa.types.is_date(pa_dtype):
        A = date(1999, 12, 31)
        B = date(2010, 1, 1)
        C = date(2022, 1, 1)
    elif pa.types.is_timestamp(pa_dtype):
        A = datetime(1999, 1, 1, 1, 1, 1, 1)
        B = datetime(2020, 1, 1)
        C = datetime(2020, 1, 1, 1)
    elif pa.types.is_duration(pa_dtype):
        A = timedelta(-1)
        B = timedelta(0)
        C = timedelta(1, 4)
    elif pa.types.is_time(pa_dtype):
        A = time(0, 0)
        B = time(0, 12)
        C = time(12, 12)
    elif pa.types.is_string(pa_dtype):
        A = "a"
        B = "b"
        C = "c"
    elif pa.types.is_binary(pa_dtype):
        A = b"a"
        B = b"b"
        C = b"c"
    elif pa.types.is_decimal(pa_dtype):
        A = Decimal("-1.1")
        B = Decimal("0.0")
        C = Decimal("1.1")
    else:
        raise NotImplementedError
    return pd.array([B, B, None, None, A, A, B, C], dtype=dtype)


@pytest.fixture
def data_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, C, A] with
    A < B < C
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[7], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_missing_for_sorting(data_for_grouping):
    """
    Length-3 array with a known sort order.

    This should be three items [B, NA, A] with
    A < B and NA missing.
    """
    return type(data_for_grouping)._from_sequence(
        [data_for_grouping[0], data_for_grouping[2], data_for_grouping[4]],
        dtype=data_for_grouping.dtype,
    )


@pytest.fixture
def data_for_twos(data):
    """Length-100 array in which all the elements are two."""
    pa_dtype = data.dtype._wrapped_dtype.pyarrow_dtype
    if pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
        return pd.array([2] * 100, dtype=data.dtype)
    # tests will be xfailed where 2 is not a valid scalar for pa_dtype
    return data


@pytest.fixture
def na_value():
    """The scalar missing value for this type. Default 'None'"""
    return pd.NA


@pytest.fixture(params=[None, lambda x: x])
def sort_by_key(request):
    """
    Simple fixture for testing keys in sorting methods.
    Tests None (no key) and the identity key.
    """
    return request.param


class TestBaseCasting(base.BaseCastingTests):
    pass
    #def test_astype_object_series(self, all_data):
    #    # The base class expects .astype(object) to cast to np.dtype(object), not
    #    #  ModinDtype(np.dtype(object)
    #    ser = pd.Series(all_data, name="A")
    #    result = ser.astype(object)
    #    assert result.dtype == ModinDtype(np.dtype(object))

    '''
    def test_astype_object_frame(self, all_data):
        # The base class expects .astype(object) to cast to np.dtype(object), not
        #  ModinDtype(np.dtype(object)
        df = pd.DataFrame({"A": all_data})

        result = df.astype(object)
        assert isinstance(result._mgr.arrays[0], np.ndarray)
        assert result._mgr.arrays[0].dtype == ModinDtype(np.dtype(object))

        # check that we can compare the dtypes
        comp = result.dtypes == df.dtypes
        assert not comp.any()
    '''


class TestConstructors(base.BaseConstructorsTests):
    pass


class TestGetitemTests(base.BaseGetitemTests):
    pass


# TODO: enable once modin supports pandas 2.0
# class TestBaseAccumulateTests(base.BaseAccumulateTests):
#    pass


class TestBaseNumericReduce(base.BaseNumericReduceTests):
    def check_reduce(self, ser, op_name, skipna):
        pa_dtype = ser.dtype._wrapped_dtype.pyarrow_dtype
        if op_name == "count":
            result = getattr(ser, op_name)()
        else:
            result = getattr(ser, op_name)(skipna=skipna)
        if pa.types.is_boolean(pa_dtype):
            # Can't convert if ser contains NA
            pytest.skip(
                "pandas boolean data with NA does not fully support all reductions"
            )
        elif pa.types.is_integer(pa_dtype) or pa.types.is_floating(pa_dtype):
            ser = ser.astype("Float64")
        if op_name == "count":
            expected = getattr(ser, op_name)()
        else:
            expected = getattr(ser, op_name)(skipna=skipna)
        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize("skipna", [True, False])
    def test_reduce_series(self, data, all_numeric_reductions, skipna, request):
        pa_dtype = data.dtype._wrapped_dtype.pyarrow_dtype
        opname = all_numeric_reductions

        ser = pd.Series(data)

        should_work = True
        if pa.types.is_temporal(pa_dtype) and opname in [
            "sum",
            "var",
            "skew",
            "kurt",
            "prod",
        ]:
            if pa.types.is_duration(pa_dtype) and opname in ["sum"]:
                # summing timedeltas is one case that *is* well-defined
                pass
            else:
                should_work = False
        elif (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ) and opname in [
            "sum",
            "mean",
            "median",
            "prod",
            "std",
            "sem",
            "var",
            "skew",
            "kurt",
        ]:
            should_work = False

        if not should_work:
            # matching the non-pyarrow versions, these operations *should* not
            #  work for these dtypes
            msg = f"does not support reduction '{opname}'"
            with pytest.raises(TypeError, match=msg):
                getattr(ser, opname)(skipna=skipna)

            return

        xfail_mark = pytest.mark.xfail(
            raises=TypeError,
            reason=(
                f"{all_numeric_reductions} is not implemented in "
                f"pyarrow={pa.__version__} for {pa_dtype}"
            ),
        )
        if all_numeric_reductions in {"skew", "kurt"}:
            request.node.add_marker(xfail_mark)
        elif (
            all_numeric_reductions in {"var", "std", "median"}
            and pa_version_under7p0
            and pa.types.is_decimal(pa_dtype)
        ):
            request.node.add_marker(xfail_mark)
        elif all_numeric_reductions == "sem" and pa_version_under8p0:
            request.node.add_marker(xfail_mark)

        elif pa.types.is_boolean(pa_dtype) and all_numeric_reductions in {
            "sem",
            "std",
            "var",
            "median",
        }:
            request.node.add_marker(xfail_mark)
        super().test_reduce_series(data, all_numeric_reductions, skipna)


class TestBaseBooleanReduce(base.BaseBooleanReduceTests):
    pass


class TestBaseGroupby(base.BaseGroupbyTests):
    pass


class TestBaseDtype(base.BaseDtypeTests):
    pass


class TestBaseIndex(base.BaseIndexTests):
    pass


class TestBaseInterface(base.BaseInterfaceTests):
    pass


class TestBaseMissing(base.BaseMissingTests):
    pass


class TestBasePrinting(base.BasePrintingTests):
    pass


class TestBaseReshaping(base.BaseReshapingTests):
    pass


class TestBaseSetitem(base.BaseSetitemTests):
    pass


class TestBaseParsing(base.BaseParsingTests):
    pass


class TestBaseUnaryOps(base.BaseUnaryOpsTests):
    pass


class TestBaseMethods(base.BaseMethodsTests):
    @pytest.mark.xfail(reason="GH#6229")
    @pytest.mark.parametrize("box", [pd.array, pd.Series, pd.DataFrame])
    def test_equals(self, data, na_value, as_series, box):
        super().test_equals(data, na_value, as_series, box)


class TestBaseArithmeticOps(base.BaseArithmeticOpsTests):
    def get_op_from_name(self, op_name):
        short_opname = op_name.strip("_")
        if short_opname == "rtruediv":
            # use the numpy version that won't raise on division by zero
            return lambda x, y: np.divide(y, x)
        elif short_opname == "rfloordiv":
            return lambda x, y: np.floor_divide(y, x)

        return tm.get_op_from_name(op_name)

    def _patch_combine(self, obj, other, op):
        # BaseOpsUtil._combine can upcast expected dtype
        # (because it generates expected on python scalars)
        # while ArrowExtensionArray maintains original type
        expected = base.BaseArithmeticOpsTests._combine(self, obj, other, op)
        was_frame = False
        if isinstance(expected, pd.DataFrame):
            was_frame = True
            expected_data = expected.iloc[:, 0]
            original_dtype = obj.iloc[:, 0].dtype
        else:
            expected_data = expected
            original_dtype = obj.dtype

        pa_expected = pa.array(np.asarray(expected_data._values))

        if pa.types.is_duration(pa_expected.type):
            orig_pa_type = original_dtype._wrapped_dtype.pyarrow_dtype
            if pa.types.is_date(orig_pa_type):
                if pa.types.is_date64(orig_pa_type):
                    # TODO: why is this different vs date32?
                    unit = "ms"
                else:
                    unit = "s"
            else:
                # pyarrow sees sequence of datetime/timedelta objects and defaults
                #  to "us" but the non-pointwise op retains unit
                # timestamp or duration
                unit = orig_pa_type.unit
                if type(other) in [datetime, timedelta] and unit in ["s", "ms"]:
                    # pydatetime/pytimedelta objects have microsecond reso, so we
                    #  take the higher reso of the original and microsecond. Note
                    #  this matches what we would do with DatetimeArray/TimedeltaArray
                    unit = "us"

            pa_expected = pa_expected.cast(f"duration[{unit}]")
        else:
            pa_expected = pa_expected.cast(original_dtype.pyarrow_dtype)

        pd_expected = type(expected_data._values)(pa_expected)
        if was_frame:
            expected = pd.DataFrame(
                pd_expected, index=expected.index, columns=expected.columns
            )
        else:
            expected = pd.Series(pd_expected)
        return expected

    def _is_temporal_supported(self, opname, pa_dtype):
        return not pa_version_under8p0 and (
            opname in ("__add__", "__radd__")
            and pa.types.is_duration(pa_dtype)
            or opname in ("__sub__", "__rsub__")
            and pa.types.is_temporal(pa_dtype)
        )

    def _get_scalar_exception(self, opname, pa_dtype):
        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)
        if opname in {
            "__mod__",
            "__rmod__",
        }:
            exc = NotImplementedError
        elif arrow_temporal_supported:
            exc = None
        elif opname in ["__add__", "__radd__"] and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            exc = None
        elif not (
            pa.types.is_floating(pa_dtype)
            or pa.types.is_integer(pa_dtype)
            or pa.types.is_decimal(pa_dtype)
        ):
            exc = pa.ArrowNotImplementedError
        else:
            exc = None
        return exc

    def _get_arith_xfail_marker(self, opname, pa_dtype):
        mark = None

        arrow_temporal_supported = self._is_temporal_supported(opname, pa_dtype)

        if (
            opname == "__rpow__"
            and (
                pa.types.is_floating(pa_dtype)
                or pa.types.is_integer(pa_dtype)
                or pa.types.is_decimal(pa_dtype)
            )
            and not pa_version_under7p0
        ):
            mark = pytest.mark.xfail(
                reason=(
                    f"GH#29997: 1**pandas.NA == 1 while 1**pyarrow.NA == NULL "
                    f"for {pa_dtype}"
                )
            )
        elif arrow_temporal_supported and pa.types.is_time(pa_dtype):
            mark = pytest.mark.xfail(
                raises=TypeError,
                reason=(
                    f"{opname} not supported between"
                    f"pd.NA and {pa_dtype} Python scalar"
                ),
            )
        elif (
            opname == "__rfloordiv__"
            and (pa.types.is_integer(pa_dtype) or pa.types.is_decimal(pa_dtype))
            and not pa_version_under7p0
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        elif (
            opname == "__rtruediv__"
            and pa.types.is_decimal(pa_dtype)
            and not pa_version_under7p0
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="divide by 0",
            )
        elif (
            opname == "__pow__"
            and pa.types.is_decimal(pa_dtype)
            and pa_version_under7p0
        ):
            mark = pytest.mark.xfail(
                raises=pa.ArrowInvalid,
                reason="Invalid decimal function: power_checked",
            )

        return mark

    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_arith_frame_with_scalar(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype._wrapped_dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")

        self.frame_scalar_exc = self._get_scalar_exception(
            all_arithmetic_operators, pa_dtype
        )

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.node.add_marker(mark)

        if (
            (
                all_arithmetic_operators == "__floordiv__"
                and pa.types.is_integer(pa_dtype)
            )
            or pa.types.is_duration(pa_dtype)
            or pa.types.is_timestamp(pa_dtype)
            or pa.types.is_date(pa_dtype)
        ):
            # BaseOpsUtil._combine always returns int64, while ArrowExtensionArray does
            # not upcast
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        super().test_arith_frame_with_scalar(data, all_arithmetic_operators)

    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_arith_series_with_array(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype._wrapped_dtype.pyarrow_dtype

        self.series_array_exc = self._get_scalar_exception(
            all_arithmetic_operators, pa_dtype
        )

        if (
            all_arithmetic_operators
            in (
                "__sub__",
                "__rsub__",
            )
            and pa.types.is_unsigned_integer(pa_dtype)
            and not pa_version_under7p0
        ):
            request.node.add_marker(
                pytest.mark.xfail(
                    raises=pa.ArrowInvalid,
                    reason=(
                        f"Implemented pyarrow.compute.subtract_checked "
                        f"which raises on overflow for {pa_dtype}"
                    ),
                )
            )

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.node.add_marker(mark)

        op_name = all_arithmetic_operators
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))

        if (
            pa.types.is_floating(pa_dtype)
            or (
                pa.types.is_integer(pa_dtype)
                and all_arithmetic_operators not in ["__truediv__", "__rtruediv__"]
            )
            or pa.types.is_duration(pa_dtype)
            or pa.types.is_timestamp(pa_dtype)
            or pa.types.is_date(pa_dtype)
        ):
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        self.check_opname(ser, op_name, other, exc=self.series_array_exc)

    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_arith_series_with_scalar(
        self, data, all_arithmetic_operators, request, monkeypatch
    ):
        pa_dtype = data.dtype._wrapped_dtype.pyarrow_dtype

        if all_arithmetic_operators == "__rmod__" and (
            pa.types.is_string(pa_dtype) or pa.types.is_binary(pa_dtype)
        ):
            pytest.skip("Skip testing Python string formatting")

        self.series_scalar_exc = self._get_scalar_exception(
            all_arithmetic_operators, pa_dtype
        )

        mark = self._get_arith_xfail_marker(all_arithmetic_operators, pa_dtype)
        if mark is not None:
            request.node.add_marker(mark)

        if (
            (
                all_arithmetic_operators == "__floordiv__"
                and pa.types.is_integer(pa_dtype)
            )
            or pa.types.is_duration(pa_dtype)
            or pa.types.is_timestamp(pa_dtype)
            or pa.types.is_date(pa_dtype)
        ):
            # BaseOpsUtil._combine always returns int64, while ArrowExtensionArray does
            # not upcast
            monkeypatch.setattr(TestBaseArithmeticOps, "_combine", self._patch_combine)
        super().test_arith_series_with_scalar(data, all_arithmetic_operators)

    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_divmod(self, data):
        super().test_divmod(data)


class TestBaseComparisonOps(base.BaseComparisonOpsTests):
    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_compare_array(self, data, comparison_op, na_value):
        ser = pd.Series(data)
        # pd.Series([ser.iloc[0]] * len(ser)) may not return ArrowExtensionArray
        # since ser.iloc[0] is a python scalar
        other = pd.Series(pd.array([ser.iloc[0]] * len(ser), dtype=data.dtype))
        if comparison_op.__name__ in ["eq", "ne"]:
            # comparison should match point-wise comparisons
            result = comparison_op(ser, other)
            # Series.combine does not calculate the NA mask correctly
            # when comparing over an array
            assert result[8] is na_value
            assert result[97] is na_value
            expected = ser.combine(other, comparison_op)
            expected[8] = na_value
            expected[97] = na_value
            self.assert_series_equal(result, expected)

        else:
            exc = None
            try:
                result = comparison_op(ser, other)
            except Exception as err:
                exc = err

            if exc is None:
                # Didn't error, then should match point-wise behavior
                expected = ser.combine(other, comparison_op)
                self.assert_series_equal(result, expected)
            else:
                with pytest.raises(type(exc)):
                    ser.combine(other, comparison_op)

    @pytest.mark.xfail(
        reason="Modin needs to update to pandas 2.0 to get this to work with "
        "pyarrow dtypes."
    )
    def test_compare_scalar(self, data, comparison_op):
        super().test_compare_scalar(data, comparison_op)
