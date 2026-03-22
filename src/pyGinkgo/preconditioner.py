# SPDX-FileCopyrightText: 2025 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

from . import gko_types
import pyGinkgo as pg
from . import pyGinkgoBindings as pGB
from .pyGinkgoBindings.preconditioner import *


def Ilu(device: gko_types.DeviceType, matrix: pGB.LinOp):
    executor = pg.device(device)

    # TODO: create a better way to check the type of the matrix
    typization = type(matrix).__name__.split('_')[1:]
    if len(typization) == 1:
        ilu_cls = getattr(pGB.preconditioner, "Ilu_" + typization[0])
    elif len(typization) == 2:
        ilu_cls = getattr(pGB.preconditioner, "Ilu_" + typization[0] + "_" + typization[1])
    else:
        raise ValueError(f"Not a known matrix type: {typization}.")
    
    return ilu_cls(executor, matrix)
