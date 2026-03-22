// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

// TODO: might require placement in a separate dedicated folder

#include "python.hpp"

namespace py = pybind11;

void add_executor_classes(py::module_ &root_module)
{
    py::class_<gko::Executor, std::shared_ptr<gko::Executor>>(root_module,
                                                              "Executor")
        .def("synchronize", &gko::Executor::synchronize,
             "Synchronizes the executor");

    // CPU executors
    py::class_<gko::detail::ExecutorBase<gko::OmpExecutor>, gko::Executor,
               std::shared_ptr<gko::detail::ExecutorBase<gko::OmpExecutor>>>(
        root_module, "OmpExecutorBase");

    py::class_<gko::OmpExecutor, gko::detail::ExecutorBase<gko::OmpExecutor>,
               std::shared_ptr<gko::OmpExecutor>>(root_module, "OmpExecutor")
        .def(py::init([]() { return gko::OmpExecutor::create(); }));

    py::class_<gko::ReferenceExecutor, gko::OmpExecutor,
               std::shared_ptr<gko::ReferenceExecutor>>(root_module,
                                                        "ReferenceExecutor")
        .def(py::init([]() { return gko::ReferenceExecutor::create(); }));

// GPU executors
#ifdef GINKGO_BUILD_CUDA
    py::class_<gko::detail::ExecutorBase<gko::CudaExecutor>, gko::Executor,
               std::shared_ptr<gko::detail::ExecutorBase<gko::CudaExecutor>>>(
        root_module, "CudaExecutorBase");

    py::class_<gko::CudaExecutor, gko::detail::ExecutorBase<gko::CudaExecutor>,
               std::shared_ptr<gko::CudaExecutor>>(root_module, "CudaExecutor")
        .def(py::init([](int dev_id, std::shared_ptr<gko::Executor> master,
                         std::shared_ptr<gko::CudaAllocatorBase> alloc,
                         std::shared_ptr<gko::cuda_stream> stream) {
                 return gko::CudaExecutor::create(dev_id, master, alloc,
                                                  stream->get());
             }),
             // The first two are the deviation from the original library
             py::arg_v("device_id", 0),
             py::arg_v("master", gko::ReferenceExecutor::create(),
                       "ReferenceExecutor()"),
             py::arg_v("allocator", std::make_shared<gko::CudaAllocator>(),
                       "CudaAllocator()"),
             py::arg_v("stream", std::make_shared<gko::cuda_stream>(),
                       "cuda_stream()"),
             "The default arguments are:\n"
             "    device_id: 0\n"
             "    master: ReferenceExecutor()\n"
             "    allocator: CudaAllocator()\n"
             "    stream: cuda_stream()")
        .def_property_readonly(
            "master",
            [](std::shared_ptr<gko::CudaExecutor> self)
                -> std::shared_ptr<gko::Executor>
            // Hopefully there would be no problems with converting val to lval
            { return self->get_master(); })
        .def_property_readonly("device_id", &gko::CudaExecutor::get_device_id)
        .def_static("get_num_devices", &gko::CudaExecutor::get_num_devices);
#endif
#ifdef GINKGO_BUILD_HIP
    py::class_<gko::detail::ExecutorBase<gko::HipExecutor>, gko::Executor,
               std::shared_ptr<gko::detail::ExecutorBase<gko::HipExecutor>>>(
        root_module, "HipExecutorBase");

    py::class_<gko::HipExecutor, gko::detail::ExecutorBase<gko::HipExecutor>,
               std::shared_ptr<gko::HipExecutor>>(root_module, "HipExecutor")
        .def(py::init([](int dev_id, std::shared_ptr<gko::Executor> master,
                         std::shared_ptr<gko::HipAllocatorBase> alloc,
                         std::shared_ptr<gko::hip_stream> stream) {
                 return gko::HipExecutor::create(dev_id, master, alloc,
                                                 stream->get());
             }),
             // The first two are the deviation from the original library
             py::arg_v("device_id", 0),
             py::arg_v("master", gko::ReferenceExecutor::create(),
                       "ReferenceExecutor()"),
             py::arg_v("allocator", std::make_shared<gko::HipAllocator>(),
                       "HipAllocator()"),
             py::arg_v("stream", std::make_shared<gko::hip_stream>(),
                       "hip_stream()"),
             "The default arguments are:\n"
             "    device_id: 0\n"
             "    master: ReferenceExecutor()\n"
             "    allocator: HipAllocator()\n"
             "    stream: hip_stream()")
        .def_property_readonly(
            "master",
            [](std::shared_ptr<gko::HipExecutor> self)
                -> std::shared_ptr<gko::Executor>
            // Hopefully there would be no problems with converting val to lval
            { return self->get_master(); })
        .def_property_readonly("device_id", &gko::HipExecutor::get_device_id)
        .def_static("get_num_devices", &gko::HipExecutor::get_num_devices);
#endif
#ifdef GINKGO_BUILD_SYCL
    py::class_<gko::detail::ExecutorBase<gko::DpcppExecutor>, gko::Executor,
               std::shared_ptr<gko::detail::ExecutorBase<gko::DpcppExecutor>>>(
        root_module, "DpcppExecutorBase");

    py::class_<gko::DpcppExecutor,
               gko::detail::ExecutorBase<gko::DpcppExecutor>,
               std::shared_ptr<gko::DpcppExecutor>>(root_module,
                                                    "DpcppExecutor")
        .def(py::init([](int dev_id, std::shared_ptr<gko::Executor> master,
                         std::string device_type,
                         dpcpp_queue_property property) {
                 return gko::DpcppExecutor::create(dev_id, master, device_type,
                                                   property);
             }),
             // The first two are the deviation from the original library
             py::arg_v("device_id", 0),
             py::arg_v("master", gko::ReferenceExecutor::create(),
                       "ReferenceExecutor()"),
             py::arg_v("device_type", "all", "all"),
             py::arg_v("dpcpp_queue_property", dpcpp_queue_property::in_order,
                       "in_order"),
             "The default arguments are:\n"
             "    device_id: 0\n"
             "    master: ReferenceExecutor()\n"
             "    device_type: all\n"
             "    dpcpp_queue_property: in_order")
        .def_property_readonly(
            "master",
            [](std::shared_ptr<gko::DpcppExecutor> self)
                -> std::shared_ptr<gko::Executor>
            // Hopefully there would be no problems with converting val to lval
            { return self->get_master(); })
        .def_property_readonly("device_id", &gko::DpcppExecutor::get_device_id)
        .def_static("get_num_devices", &gko::DpcppExecutor::get_num_devices);
#endif
}
