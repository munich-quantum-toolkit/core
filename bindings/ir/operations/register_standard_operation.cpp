/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "ir/Definitions.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "ir/operations/StandardOperation.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h> // NOLINT(misc-include-cleaner)
#include <sstream>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerStandardOperation(const nb::module_& m) {
  nb::class_<qc::StandardOperation, qc::Operation>(
      m, "StandardOperation", R"pb(Standard quantum operation.

This class is used to represent all standard quantum operations, i.e., operations that are unitary.
This includes all possible quantum gates.
Such Operations are defined by their :class:`OpType`, the qubits (controls and targets) they act on, and their parameters.

Args:
    control: The control qubit(s) of the operation (if any).
    target: The target qubit(s) of the operation.
    op_type: The type of the operation.
    params: The parameters of the operation (if any).)pb")

      .def(nb::init<>())
      .def(nb::init<qc::Qubit, qc::OpType, std::vector<qc::fp>>(), "target"_a,
           "op_type"_a, "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<const qc::Targets&, qc::OpType, std::vector<qc::fp>>(),
           "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<qc::Control, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&>(),
           "control"_a, "target"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<qc::Control, const qc::Targets&, qc::OpType,
                    const std::vector<qc::fp>&>(),
           "control"_a, "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<const qc::Controls&, qc::Qubit, qc::OpType,
                    const std::vector<qc::fp>&>(),
           "controls"_a, "target"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<const qc::Controls&, const qc::Targets&, qc::OpType,
                    std::vector<qc::fp>>(),
           "controls"_a, "targets"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def(nb::init<const qc::Controls&, qc::Qubit, qc::Qubit, qc::OpType,
                    std::vector<qc::fp>>(),
           "controls"_a, "target0"_a, "target1"_a, "op_type"_a,
           "params"_a.sig("...") = std::vector<qc::fp>{})
      .def("__repr__", [](const qc::StandardOperation& op) {
        std::stringstream ss;
        ss << "StandardOperation(";
        const auto& controls = op.getControls();
        if (controls.size() == 1U) {
          ss << "control=";
          const auto& control = *controls.begin();
          ss << control.toString() << ", ";
        } else if (!controls.empty()) {
          ss << "controls={";
          for (const auto& control : controls) {
            ss << control.toString() << ", ";
          }
          ss << "}, ";
        }
        const auto& targets = op.getTargets();
        if (targets.size() == 1U) {
          ss << "target=" << targets.front() << ", ";
        } else if (!targets.empty()) {
          ss << "targets=[";
          for (const auto& target : targets) {
            ss << target << ", ";
          }
          ss << "], ";
        }
        ss << "op_type=" << toString(op.getType());
        const auto& params = op.getParameter();
        if (!params.empty()) {
          ss << ", params=[";
          for (const auto& param : params) {
            ss << param << ", ";
          }
          ss << "]";
        }
        ss << ")";
        return ss.str();
      });
}
} // namespace mqt
