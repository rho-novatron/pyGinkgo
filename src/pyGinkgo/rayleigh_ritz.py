# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pyGinkgo as pg
from pyGinkgo import pyGinkgoBindings as pGB

if pg.torch_avail:
    import torch
import numpy as np


def mul(a, b, dtype="float", device="cpu"):
    """ helper function to perform a @ b """
    dimRes = (a.shape[0], b.shape[1])
    # executor = a.get_executor()
    res = pg.as_tensor(device=device, dim=dimRes, dtype=dtype)
    a.apply(b, res)
    return res

def RR1(X, AX, BX, dtype="float", device="cpu"):
    """
    Computes m least dominant generalized eigenpairs of
    (A,B) with respect to the range of X using a
    Rayleigh-Ritz projection.

    Parameters:
    - X, AX, BX : Input matrices (dense)

    Returns:
    - hX     : Eigenvectors
    - Lambda : Eigenvalues
    """
    XT = X.T()
    # compute G1 = X.T @ AX
    G1 = mul(XT, AX, dtype=dtype, device=device)
    # compute G2 = X.T @ BX
    G2 = mul(XT, BX, dtype=dtype, device=device)

    # compute G2P G2' = L^(-1) @ G1
    # Find L s.t. L @ L.T = G2
    L = pg.factor(G2, kind="Lower")
    _, G2P = pg.solve(L, G1, kind="triangular", solver_args={"type": "Lower"})
    LT = L.T()

    # compute G2PP G2'' = L^(-1) @ G2P.T = L^(-1) @ G1 @ L^(-T)
    _, G2PP = pg.solve(L, G2P.T(), kind="triangular", solver_args={"type": "Lower"})

    Lambda, hY = pg.eigen_solve(G2PP)

    _, hX = pg.solve(LT, hY, kind="triangular", solver_args={"type": "Upper"})
    return hX, Lambda
