// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"

template <typename ValueType, typename IndexType>
void init_direct(py::module_ &module_solver, const std::string value_type,
                 const std::string index_type)
{
    std::string pyclass_name = "direct_" + value_type + "_" + index_type;
    std::string repr_str = "pygko.solver." + pyclass_name + " object";
    py::class_<gko::experimental::solver::Direct<ValueType, IndexType>,
               std::shared_ptr<
                   gko::experimental::solver::Direct<ValueType, IndexType>>,
               gko::LinOp>(module_solver, pyclass_name.c_str())
        .def(
            py::init([](std::shared_ptr<gko::Executor> exec,
                        std::shared_ptr<const gko::LinOp> system_matrix,
                        std::string factorization) {
                std::shared_ptr<gko::LinOpFactory> factorization_fact;

                if (factorization == "Cholesky") {
                    factorization_fact =
                        gko::share(gko::experimental::factorization::Cholesky<
                                       ValueType, IndexType>::build()
                                       .with_skip_sorting(true)
                                       .on(exec));
                }
                if (factorization == "LU") {
                    factorization_fact = gko::share(
                        gko::experimental::factorization::Lu<ValueType,
                                                             IndexType>::build()
                            .with_skip_sorting(true)
                            .on(exec));
                }

                auto solver_fact = gko::share(
                    gko::experimental::solver::Direct<ValueType,
                                                      IndexType>::build()
                        .with_factorization(factorization_fact)
                        .on(exec));
                return gko::share(solver_fact->generate(system_matrix));
            }),
            py::arg("exec"), py::arg("system_matrix"), py::arg("factorization"))
        .def("__repr__",
             [=](const gko::experimental::solver::Direct<ValueType, IndexType>
                     &o) { return repr_str; })
        .def(
            "apply",
            [](const gko::experimental::solver::Direct<ValueType, IndexType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}

void init_direct_all_types(py::module_ &module_solver)
{
#define DECLARE_DIRECT_SOLVER(ValueType, IndexType) \
    init_direct<ValueType, IndexType>(module_solver, #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_DIRECT_SOLVER);
}
