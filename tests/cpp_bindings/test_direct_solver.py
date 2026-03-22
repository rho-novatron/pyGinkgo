# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB


@pytest.mark.parametrize("solver_name", ["direct"])
@pytest.mark.parametrize("data_type", list(pg.gko_types.ValueType))
class TestDirectSolverBinding:
    ref = pGB.ReferenceExecutor()
    values = [1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
    dim = (3, 3)

    solver_args = {"direct": {"factorization": "Cholesky"}}

    def test_direct_solver(self, solver_name, data_type: pg.gko_types.ValueType):
        dense_cls = getattr(pGB.matrix, "dense_" + data_type)
        mtx = dense_cls(
            self.ref, (3, 3), np.array(self.values, dtype=data_type.numpy_type), 3
        )
        exp = np.array([1, 1 / 2.0, 1 / 3.0], dtype=data_type.numpy_type)
        solver_cls = getattr(pGB.solver, f"{solver_name}_{data_type}_int32")
        args = self.solver_args[solver_name]
        solver = solver_cls(exec=self.ref, system_matrix=mtx, **args)

        rhs = dense_cls(mtx.get_executor(), (self.dim[0], 1))
        rhs.fill(1.0)
        x = dense_cls(mtx.get_executor(), (self.dim[0], 1))
        x.fill(0.0)
        solver.apply(rhs, x)

        res = np.array(x)
        assert np.all(exp) == np.all(res)
