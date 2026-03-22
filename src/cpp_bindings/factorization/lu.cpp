// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"


void init_lu_factorization(py::module_ &module_factorization)
{
    py::class_<gko::experimental::factorization::Lu<ValueType, IndexType>,
               std::shared_ptr<gko::experimental::factorization::Factorization<
                   ValueType, IndexType>>>(module_factorization, "lu")
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
            /*auto mtx =*/
            /*    gko::as<gko::matrix::Csr<ValueType,
             * IndexType>>(system_matrix);*/
            auto fact = gko::experimental::factorization::Lu<ValueType,
                                                             IndexType>::build()
                            .on(exec);
            return gko::share(fact->generate(system_matrix));
        }));
}
