# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


@pytest.mark.parametrize("data_type", list(pg.gko_types.dtype))
class TestArray:
    ref = pGB.ReferenceExecutor()
    size = 5
    lst = [1.0, 2.0, 3.0, 4.0, 5.0]

    def test_init_empty_array(self, data_type: pg.gko_types.ValueType):
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(self.ref, self.size)
        with pytest.deprecated_call():
            assert arr.get_size() == self.size

    def test_can_fill_array(self, data_type: pg.gko_types.ValueType):
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(self.ref, self.size)
        arr.fill(10)  # implicit conversion to data_type
        assert arr.at(0) == 10  # implicit conversion to data_type

    def test_can_instantiate_array_from_numpy_array(
        self, data_type: pg.gko_types.ValueType
    ):
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(self.ref, np_array)
        assert arr.shape[0] == len(np_array)
        for i in range(self.size):
            expect = self.lst[i]
            assert arr.at(i) == expect  # implicit conversion to data_type

    def test_size_property(self, data_type: pg.gko_types.ValueType):
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(self.ref, np_array)
        with pytest.deprecated_call():
            assert arr.size == len(np_array)
            with pytest.raises(AttributeError):
                arr.size = 10

    def test_shape_property(self, data_type: pg.gko_types.ValueType):
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(self.ref, np_array)
        assert arr.shape == (len(np_array),)
        with pytest.raises(AttributeError):
            arr.shape = (10,)

    def test_can_copy_construct_array(self, data_type: pg.gko_types.ValueType):
        array_cls = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        arr = array_cls(self.ref, np_array)
        arr_copy = array_cls(self.ref, arr)

        assert arr.shape == arr_copy.shape

    def test_can_construct_np_array_from_gko_array(
        self, data_type: pg.gko_types.ValueType
    ):
        array_cls = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        arr = array_cls(self.ref, np_array)
        np_array_copy = np.array(arr)

        assert np.array_equal(np_array, np_array_copy)

    def test_can_reduce_add_array(self, data_type: pg.gko_types.ValueType):
        array_cls = getattr(pGB.base, "array_" + data_type)
        np_array = np.array(self.lst, dtype=data_type.numpy_type)
        arr = array_cls(self.ref, np_array)
        init_value = 0  # implicit conversion to data_type
        expected_value = 15  # implicit conversion to data_type
        assert pGB.base.reduce_add(arr, init_value) == expected_value
