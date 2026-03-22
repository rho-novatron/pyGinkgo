// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"


void init_factorization(py::module_ &module_factorization)
{
    py::class_<gko::experimental::factorization::Factorization<float, int>,
               std::shared_ptr<
                   gko::experimental::factorization::Factorization<float, int>>,
               gko::LinOp>(module_factorization, "factorization")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            auto fact =
                gko::experimental::factorization::Cholesky<float, int>::build()
                    .on(exec);
            return gko::share(fact->generate(system_matrix)->unpack());
        }))
        .def(
            "get_upper_factor",
            [](gko::experimental::factorization::Factorization<float, int> &m) {
                return m.get_upper_factor();
            },
            "Returns upper factors")
        .def(
            "get_lower_factor",
            [](gko::experimental::factorization::Factorization<float, int> &m) {
                return m.get_lower_factor();
            },
            "Returns lower factors");
}
