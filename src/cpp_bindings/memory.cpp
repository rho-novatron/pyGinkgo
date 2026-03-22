// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

// TODO: might require placement in a separate dedicated folder
//      (in this case, something like 'base' folder)

#include "python.hpp"

namespace py = pybind11;

void add_allocator_classes(py::module_ &root_module)
{
    py::class_<gko::Allocator, std::shared_ptr<gko::Allocator>>(root_module,
                                                                "Allocator");

#ifdef GINKGO_BUILD_CUDA
    // TODO: implement CudaAllocatorBase binding,
    //    to be able to define custom allocator
    py::class_<gko::CudaAllocatorBase, gko::Allocator,
               std::shared_ptr<gko::CudaAllocatorBase>>(root_module,
                                                        "CudaAllocatorBase");

    py::class_<gko::CudaAllocator, gko::CudaAllocatorBase,
               std::shared_ptr<gko::CudaAllocator>>(root_module,
                                                    "CudaAllocator")
        .def(py::init());
#endif
#ifdef GINKGO_BUILD_HIP
    // TODO: implement HipAllocatorBase binding,
    //    to be able to define custom allocator
    py::class_<gko::HipAllocatorBase, gko::Allocator,
               std::shared_ptr<gko::HipAllocatorBase>>(root_module,
                                                       "HipAllocatorBase");

    py::class_<gko::HipAllocator, gko::HipAllocatorBase,
               std::shared_ptr<gko::HipAllocator>>(root_module, "HipAllocator")
        .def(py::init());
#endif
}
