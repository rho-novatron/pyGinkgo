# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec


def test_array_hip():
    if pGB.HipExecutor.get_num_devices() < 1:
        pytest.skip("HIP is not available")

    executor = pGB.HipExecutor()
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    arr = pGB.base.array_double(executor, np_array)
    arr_copy = pGB.base.array_double(executor, arr)
    assert arr.shape == arr_copy.shape
    assert pGB.base.reduce_add(arr, 0.0) == 15.0


def test_dense_copy_to_host():
    if pGB.HipExecutor.get_num_devices() < 1:
        pytest.skip("HIP is not available")

    master = pGB.ReferenceExecutor()
    executor = pGB.HipExecutor(master=master)
    np_array = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    dense = pGB.matrix.dense_double(executor, np_array)
    dense_on_master = dense.copy_to_host()
    assert dense.get_executor() == executor
    assert dense_on_master.get_executor() == master
    assert id(dense) != id(dense_on_master)
    verify_dense_vec(dense_on_master, np_array)


def test_device_id():
    if pGB.HipExecutor.get_num_devices() < 1:
        pytest.skip("HIP is not available")

    executor = pGB.HipExecutor(0)
    assert executor.device_id == 0
    # Test if executor.device_id is immutable
    # trying to change it should raise an Attribute error
    with pytest.raises(AttributeError):
        executor.device_id = 1


def test_master():
    if pGB.HipExecutor.get_num_devices() < 1:
        pytest.skip("HIP is not available")

    master = pGB.ReferenceExecutor()
    executor = pGB.HipExecutor(0, master)
    assert id(executor.master) == id(master)
    with pytest.raises(AttributeError):
        executor.master = pGB.ReferenceExecutor()
