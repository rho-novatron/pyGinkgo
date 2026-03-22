// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#pragma once

#include <ginkgo/ginkgo.hpp>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

using ValueType = double;
using IndexType = int;

using int32 = gko::int32;
using int64 = gko::int64;

using half = gko::half;

// Declaration of half-precision type for the numpy
// Used solution from:
// https://github.com/PaddlePaddle/Paddle/blob/d2a6b77dd90fa749cf2602378b3b506d628f1061/paddle/fluid/pybind/inference_api.cc#L59-L85
namespace PYBIND11_NAMESPACE {
namespace detail {
// Note: use same enum number of float16 in numpy.
// import numpy as np
// print np.dtype(np.float16).num  # 23
constexpr int NPY_FLOAT16_ = 23;

template <>
struct npy_format_descriptor<half> {
    static py::dtype dtype() { return py::dtype(NPY_FLOAT16_); }
    static std::string format()
    {
        // Note: "e" represents float16.
        // Details at:
        // https://docs.python.org/3/library/struct.html#:~:text=(3)-,e
        return "e";
    }
    static constexpr auto name = _("numpy.float16");
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE

// https://pybind11.readthedocs.io/en/stable/advanced/cast/custom.html
namespace PYBIND11_NAMESPACE {
namespace detail {
template <>
struct type_caster<half> {
public:
    /**
     * This macro establishes the name 'half' in
     * function signatures and declares a local variable
     * 'value' of type half
     */
    PYBIND11_TYPE_CASTER(half, const_name("numpy.float16"));

    /**
     * Conversion part 1 (Python->C++): convert a PyObject into a half
     * instance or return false upon failure. The second argument
     * indicates whether implicit conversions should be applied.
     */
    bool load(handle src, bool)
    {
        /* Extract PyObject from handle */
        PyObject *source = src.ptr();

        /* Try converting into a Python float value */
        PyObject *tmp = PyNumber_Float(source);
        if (!tmp) return false;

        // TODO: fix double conversion on ginkgo side
        value = (float)PyFloat_AsDouble(source);
        Py_DECREF(tmp);

        /* Ensure return code was OK (to avoid out-of-range errors etc) */
        return !(value == -1 && !PyErr_Occurred());
    }

    /**
     * Conversion part 2 (C++ -> Python): convert an half instance into
     * a Python object. The second and third arguments are used to
     * indicate the return value policy and parent object (for
     * ``return_value_policy::reference_internal``) and are generally
     * ignored by implicit casters.
     */
    static handle cast(half src, return_value_policy /* policy */,
                       handle /* parent */)
    {
        // Based on
        // https://github.com/pybind/pybind11/issues/1288#issuecomment-373942774
        return py::dtype(detail::NPY_FLOAT16_)
            .attr("type")((float)src)
            .release();
    }
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
