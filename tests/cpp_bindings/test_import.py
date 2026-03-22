# SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
#
# SPDX-License-Identifier: MIT

# tests/test_import.py


def test_import():
    try:
        import pyGinkgo  # noqa: F401
        import pyGinkgo.pyGinkgoBindings  # noqa: F401

        assert True  # If import succeeds, the test passes

    except ImportError:
        assert False  # If import fails, the test fails
