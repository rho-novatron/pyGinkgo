// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "python.hpp"
#include "utils.hpp"

template <typename ValueType>
void init_logger(py::module_ &logger_module, const std::string value_type)
{
    std::string pyclass_name = "convergence_logger_" + value_type;
    py::class_<gko::log::Convergence<ValueType>,
               std::shared_ptr<gko::log::Convergence<ValueType>>>(
        logger_module, pyclass_name.c_str())
        .def(
            py::init([]() {
                return gko::share(gko::log::Convergence<ValueType>::create());
            }),
            "Creates an empty stream wrapper, representing the default stream.")
        .def("has_converged", &gko::log::Convergence<ValueType>::has_converged,
             "Returns whether the solver has converged")
        .def(
            "get_residual_norm",
            [](gko::log::Convergence<ValueType> &o) {
                auto res_norm = gko::as<gko::matrix::Dense<ValueType>>(
                    o.get_residual_norm());
                auto res_norm_host = gko::matrix::Dense<ValueType>::create(
                    res_norm->get_executor()->get_master(), gko::dim<2>{1});
                res_norm_host->copy_from(res_norm);
                return res_norm_host->at(0);
            },
            "Returns the residual norm")
        .def("get_num_iterations",
             &gko::log::Convergence<ValueType>::get_num_iterations,
             "Returns whether the number of iterations");
}

void init_logger_all_types(py::module_ &logger_module)
{
#define DECLARE_LOGGER(ValueType) \
    init_logger<ValueType>(logger_module, #ValueType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_TYPE(DECLARE_LOGGER);
}
