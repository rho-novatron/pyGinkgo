# SPDX-FileCopyrightText: 2025 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from . import gko_types
from . import pyGinkgoBindings as pGB
from typing import Optional
    

def device(type: gko_types.DeviceType = "cpu", index: Optional[int] = None) -> pGB.Executor:
    """
    Get Ginkgo executor device.

    Parameters
    ----------
    type : str
        The type of device to set.
        Or use `cuda:index` or `hip:index` to set a specific device index.
    index : int, optional
        The index of the device to set. Default is 0.
        It can only be specified when type doesn't include it already.

    If type is a Ginkgo executor, it will be returned as is.
    """
    if isinstance(type, pGB.Executor):
        return type

    params = type.split(":")

    if len(params) + (index is not None) > 2:
        raise ValueError("Too many index parameters. Only one index is allowed.")
    
    if index is None and len(params) == 2:
        try:
            index = int(params[1])
        except ValueError:
            raise ValueError("Invalid device index. It should be an integer.")
    
    # Making device type case independent
    type = params[0].lower()
    if gko_types.ExecutorType.cpu in type:
        return pGB.ReferenceExecutor()
    elif gko_types.ExecutorType.omp in type:
        return pGB.OmpExecutor()
    
    # All the other types require an index
    if index is None:
        index = 0
    
    if gko_types.ExecutorType.cuda in type:
        return pGB.CudaExecutor(
            device_id=index,
        )
    elif gko_types.ExecutorType.hip in type:
        return pGB.HipExecutor(
            device_id=index,
        )
    elif gko_types.ExecutorType.dpcpp in type:
        return pGB.DpcppExecutor(
            device_id=index,
        )

    raise ValueError(
        f"Unknown device type: {type}." +
        "Valid types are: " +
        ', '.join(t for t in gko_types.ExecutorType)
    )
