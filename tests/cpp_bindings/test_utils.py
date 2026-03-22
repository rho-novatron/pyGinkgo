# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

import pytest
import numpy as np
from typing import Union


def verify_dense_vec(mtx, values: Union[list, np.ndarray], precision=0.0):
    """ """
    assert mtx.get_num_stored_elements() == len(values)
    for i in range(len(values)):
        assert mtx.at(i) == pytest.approx(values[i], abs=precision)

    assert mtx.at(2, 0) == mtx.at(2)
    # test if it can be called several times
    assert mtx.at(2, 0) == mtx.at(2)
