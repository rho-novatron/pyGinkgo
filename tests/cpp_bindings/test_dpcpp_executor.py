# SPDX-FileCopyrightText: 2025 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec


@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestDpcpp:
    def test_array_cuda(self, data_type: pg.gko_types.ValueType):
        if pGB.DpcppExecutor.get_num_devices("all") < 1:
            pytest.skip("SYCL is not available")

        executor = pGB.DpcppExecutor()
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=data_type.numpy_type)
        array_cls = getattr(pGB.base, "array_" + data_type)
        arr = array_cls(executor, np_array)
        arr_copy = array_cls(executor, arr)
        assert arr.get_size() == arr_copy.get_size()
        assert pGB.base.reduce_add(arr, 0.0) == 15.0

    def test_dense_copy_to_host(self, data_type: pg.gko_types.ValueType):
        if pGB.DpcppExecutor.get_num_devices("all") < 1:
            pytest.skip("SYCL is not available")

        master = pGB.ReferenceExecutor()
        executor = pGB.DpcppExecutor(master=master)
        np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=data_type.numpy_type)
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        dense = dense_cls(executor, np_array)
        dense_on_master = dense.copy_to_host()
        assert dense.get_executor() == executor
        assert dense_on_master.get_executor() == master
        assert id(dense) != id(dense_on_master)
        verify_dense_vec(dense_on_master, np_array)

    def test_device_id(self, data_type: pg.gko_types.ValueType):
        if pGB.DpcppExecutor.get_num_devices("all") < 1:
            pytest.skip("SYCL is not available")

        executor = pGB.DpcppExecutor(0)
        assert executor.device_id == 0
        with pytest.raises(AttributeError):
            executor.device_id = 1

    def test_can_synchronize(self, data_type: pg.gko_types.ValueType):
        if pGB.DpcppExecutor.get_num_devices("all") < 1:
            pytest.skip("SYCL is not available")

        executor = pGB.DpcppExecutor(0)
        executor.synchronize()

    def test_master(self, data_type: pg.gko_types.ValueType):
        if pGB.DpcppExecutor.get_num_devices("all") < 1:
            pytest.skip("SYCL is not available")
        master = pGB.ReferenceExecutor()
        executor = pGB.DpcppExecutor(0, master)
        assert id(executor.master) == id(master)
        with pytest.raises(AttributeError):
            executor.master = pGB.ReferenceExecutor()
