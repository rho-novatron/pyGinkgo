// SPDX-FileCopyrightText: 2024 - 2026 pyGinkgo authors
//
// SPDX-License-Identifier: MIT

#include "../python.hpp"
#include "../utils.hpp"

template <template <typename ValueType, typename IndexType> class TrsType,
          typename ValueType, typename IndexType>
void init_trs(py::module_ &module_solver, const std::string trs_type,
              const std::string value_type, const std::string index_type)
{
    const std::string pyclass_name =
        trs_type + "_" + value_type + "_" + index_type;
    const std::string repr_str = "pygko.solver." + pyclass_name + " object";
    py::class_<TrsType<ValueType, IndexType>,
               std::shared_ptr<TrsType<ValueType, IndexType>>, gko::LinOp>(
        module_solver, pyclass_name.c_str())
        .def(py::init([](std::shared_ptr<gko::Executor> exec,
                         std::shared_ptr<const gko::LinOp> system_matrix) {
                 auto solver_fact = gko::share(
                     TrsType<ValueType, IndexType>::build().on(exec));
                 return gko::share(solver_fact->generate(system_matrix));
             }),
             py::arg("exec"), py::arg("system_matrix"))
        .def("__repr__",
             [=](const TrsType<ValueType, IndexType> &o) { return repr_str; })
        .def(
            "apply",
            [](const TrsType<ValueType, IndexType> &d,
               std::shared_ptr<const gko::LinOp> b,
               std::shared_ptr<gko::LinOp> x) { d.apply(b, x); },
            "");
}

void init_trs_all_types(py::module_ &module_solver)
{
#define DECLARE_UPPER_TRS_SOLVER(ValueType, IndexType)     \
    init_trs<gko::solver::UpperTrs, ValueType, IndexType>( \
        module_solver, "UpperTrs", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_UPPER_TRS_SOLVER);

#define DECLARE_LOWER_TRS_SOLVER(ValueType, IndexType)     \
    init_trs<gko::solver::LowerTrs, ValueType, IndexType>( \
        module_solver, "LowerTrs", #ValueType, #IndexType);
    PYGKO_INSTANTIATE_FOR_EACH_NON_COMPLEX_VALUE_AND_INDEX_TYPE(
        DECLARE_LOWER_TRS_SOLVER);
}
