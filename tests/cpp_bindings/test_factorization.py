# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import numpy as np

import pyGinkgo.pyGinkgoBindings as pGB

from test_utils import verify_dense_vec


class TestFactorizationBinding:
    ref = pGB.ReferenceExecutor()
    values = [[2.0, 1.0, 1.0], [1.0, 2.0, 1.0], [1.0, 1.0, 2.0]]

    def test_factorization(self):
        np_array = np.array(self.values, dtype=np.float32)
        dense = pGB.matrix.dense_float(self.ref, np_array)
        factorization = pGB.factorization.factorization(
            self.ref, dense.convert_to_csr()
        )

        lower = factorization.get_lower_factor()
        upper = factorization.get_upper_factor()

        mul_res = pGB.matrix.dense_float(
            self.ref, np.zeros((3, 3), dtype=np.float32)
        ).convert_to_csr()
        lower.apply(upper, mul_res)

        # lower @ upper == dense
        verify_dense_vec(
            mul_res.convert_to_dense(), np_array.reshape(-1), precision=1e-6
        )
