# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import os
import pytest
import numpy as np

import pyGinkgo as pg
import pyGinkgo.pyGinkgoBindings as pGB

d_type_map = {
    "half": np.float16,
    "float": np.float32,
    "double": np.float64,
}


# TODO: it takes 6s to run test_can_solve_with_solver_args_gmres[float] and test_can_default_solve[float]
class TestSolve:
    # A bit better parametrization approach using fixtures
    # It allows to reduce amount of code repetition
    # https://stackoverflow.com/a/50135020/8302811
    @pytest.fixture(autouse=True, params=list(d_type_map.keys()))
    def __post_init__(self, request):
        data_type = request.param
        self.executor = pGB.ReferenceExecutor()
        self.fn = os.path.dirname(os.path.realpath(__file__)) + "/fv1.mtx"

        self.reader_cls = getattr(pGB.matrix, f"read_Coo_{data_type}_int32")
        self.mtx = self.reader_cls(self.fn, self.executor)
        self.dim = self.mtx.shape
        self.rows = self.dim[0]
        self.cols = self.dim[1]

        self.dense_cls = getattr(pGB.matrix, f"dense_{data_type}")
        self.rhs = self.dense_cls(self.executor, (self.rows, 1))
        self.rhs.fill(1.0)
        self.initial_guess = self.dense_cls(self.executor, (self.rows, 1))
        self.initial_guess.fill(0.0)

    def test_can_default_solve(self):
        logger, result = pg.solve(self.mtx, self.rhs, self.initial_guess)

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

    def test_can_solve_with_solver_args_pcg(self):
        solver_args = {
            "type": "solver::Cg",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }
        logger, result = pg.solve(
            self.mtx, self.rhs, self.initial_guess, solver_args=solver_args
        )

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

        # Check if result entries are non zero
        npresult = np.array(result)
        assert len(npresult) > 0
        assert len(np.nonzero(npresult)[0]) == len(npresult)

    def test_can_solve_with_solver_args_gmres(self):
        solver_args = {
            "type": "solver::Gmres",
            "preconditioner": {
                "type": "preconditioner::Ilu",
                "reverse_apply": False,
                "factorization": {"type": "factorization::ParIlu"},
            },
            "criteria": [
                {"type": "Iteration", "max_iters": 1000},
                {"type": "ResidualNorm", "reduction_factor": 1e-7},
            ],
        }
        logger, result = pg.solve(
            self.mtx, self.rhs, self.initial_guess, solver_args=solver_args
        )

        assert logger.has_converged()
        assert logger.get_num_iterations() < 1000

        # Check if result entries are non zero
        npresult = np.array(result)
        assert len(npresult) > 0
        assert len(np.nonzero(npresult)[0]) == len(npresult)

    def test_generate_solver_is_the_same_as_solve(self):
        solver_args = {
            "type": "solver::Cg",
            "criteria": [
                {"type": "Iteration", "max_iters": 2},
            ],
        }

        # prepare the two same initial guess.
        initial_guess_1 = self.dense_cls(self.executor, (self.rows, 1))
        initial_guess_1.fill(0.0)
        initial_guess_2 = self.dense_cls(self.executor, (self.rows, 1))
        initial_guess_2.fill(0.0)
        logger, result = pg.solve(
            self.mtx, self.rhs, initial_guess_1, solver_args=solver_args
        )

        solver = pg.generate_solver(self.mtx, solver_args=solver_args)
        solver.apply(self.rhs, initial_guess_2)

        assert logger.get_num_iterations() == 2

        npresult_1_host = np.array(result.copy_to_host())
        npresult_2_host = np.array(initial_guess_2.copy_to_host())

        assert (
            np.linalg.norm(npresult_1_host - npresult_2_host)
            / np.linalg.norm(npresult_1_host)
            < 1e-7
        )
