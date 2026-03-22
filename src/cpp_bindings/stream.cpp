// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;

void add_stream_classes(py::module_ &root_module)
{
#ifdef GINKGO_BUILD_CUDA
    py::class_<gko::cuda_stream, std::shared_ptr<gko::cuda_stream>>(
        root_module, "cuda_stream")
        // The comments are copied from the source of the class at
        // ginkgo-src/include/ginkgo/core/base/stream.hpp
        .def(
            py::init(),
            "Creates an empty stream wrapper, representing the default stream.")
        .def(py::init<int>(),
             "Creates a new custom CUDA stream on the given CUDA device ID");
#endif
#ifdef GINKGO_BUILD_HIP
    py::class_<gko::hip_stream, std::shared_ptr<gko::hip_stream>>(root_module,
                                                                  "hip_stream")
        // The comments are copied from the source of the class at
        // ginkgo-src/include/ginkgo/core/base/stream.hpp
        .def(
            py::init(),
            "Creates an empty stream wrapper, representing the default stream.")
        .def(py::init<int>(),
             "Creates a new custom HIP stream on the given HIP device ID");
#endif
}
