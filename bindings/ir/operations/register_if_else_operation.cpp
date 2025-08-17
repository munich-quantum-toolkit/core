/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/Register.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/Operation.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/native_enum.h>
// clang-format on

#include <cstdint>
#include <memory>
#include <sstream>

namespace mqt {

namespace py = pybind11;
using namespace pybind11::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerIfElseOperation(const py::module& m) {
  py::native_enum<qc::ComparisonKind>(
      m, "ComparisonKind", "enum.Enum",
      "Enumeration of comparison types for classic-controlled operations.")
      .value("eq", qc::ComparisonKind::Eq)
      .value("neq", qc::ComparisonKind::Neq)
      .value("lt", qc::ComparisonKind::Lt)
      .value("leq", qc::ComparisonKind::Leq)
      .value("gt", qc::ComparisonKind::Gt)
      .value("geq", qc::ComparisonKind::Geq)
      .export_values()
      .finalize();

  auto ifElse =
      py::class_<qc::IfElseOperation, qc::Operation>(m, "IfElseOperation");

  ifElse.def(py::init([](qc::Operation* thenOp, qc::Operation* elseOp,
                         qc::ClassicalRegister controlReg,
                         const std::uint64_t expectedVal,
                         const qc::ComparisonKind kind) {
               std::unique_ptr<qc::Operation> thenPtr =
                   thenOp ? thenOp->clone() : nullptr;
               std::unique_ptr<qc::Operation> elsePtr =
                   elseOp ? elseOp->clone() : nullptr;
               return std::make_unique<qc::IfElseOperation>(
                   std::move(thenPtr), std::move(elsePtr), controlReg,
                   expectedVal, kind);
             }),
             "then_operation"_a, "else_operation"_a, "control_register"_a,
             "expected_value"_a = 1U,
             "comparison_kind"_a = qc::ComparisonKind::Eq);
  ifElse.def(
      py::init([](qc::Operation* thenOp, qc::Operation* elseOp, qc::Bit cBit,
                  std::uint64_t expectedVal, qc::ComparisonKind kind) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        return std::make_unique<qc::IfElseOperation>(
            std::move(thenPtr), std::move(elsePtr), cBit, expectedVal, kind);
      }),
      "then_operation"_a, "else_operation"_a, "control_bit"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq);
  ifElse.def_property_readonly("then_operation",
                               &qc::IfElseOperation::getThenBranch,
                               py::return_value_policy::reference_internal);
  ifElse.def_property_readonly("else_operation",
                               &qc::IfElseOperation::getElseBranch,
                               py::return_value_policy::reference_internal);
  ifElse.def_property_readonly("control_register",
                               &qc::IfElseOperation::getControlRegister);
  ifElse.def_property_readonly("control_bit",
                               &qc::IfElseOperation::getControlBit);
  ifElse.def_property_readonly("expected_value",
                               &qc::IfElseOperation::getExpectedValue);
  ifElse.def_property_readonly("comparison_kind",
                               &qc::IfElseOperation::getComparisonKind);
  ifElse.def("__repr__", [](const qc::IfElseOperation& op) {
    std::stringstream ss;
    ss << "IfElseOperation(<...op...>, ";
    if (const auto& controlReg = op.getControlRegister();
        controlReg.has_value()) {
      ss << "control_register=ClassicalRegister(" << controlReg->getSize()
         << ", " << controlReg->getStartIndex() << ", " << controlReg->getName()
         << "), ";
    }
    if (const auto& controlBit = op.getControlBit(); controlBit.has_value()) {
      ss << "control_bit=" << controlBit.value() << ", ";
    }
    ss << "expected_value=" << op.getExpectedValue() << ", "
       << "comparison_kind='" << op.getComparisonKind() << "')";
    return ss.str();
  });
}

} // namespace mqt
