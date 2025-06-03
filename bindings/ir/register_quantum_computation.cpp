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
#include "ir/QuantumComputation.hpp"
#include "ir/operations/ClassicControlledOperation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "qasm3/Importer.hpp"

// These includes must be the first includes for any bindings code
// clang-format off
#include <pybind11/pybind11.h>
<<<<<<< HEAD
#include <pybind11/stl.h> // NOLINT(misc-include-cleaner)

#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
=======
#include <pybind11/stl.h>
>>>>>>> 59e09070 (Remove pybind11.hpp)
// clang-format on

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace mqt {

namespace py = pybind11;
<<<<<<< HEAD
using namespace pybind11::literals;
=======
using namespace py::literals;
>>>>>>> 59e09070 (Remove pybind11.hpp)

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQuantumComputation(py::module& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || static_cast<SizeType>(i) >= size) {
      throw py::index_error();
    }
    return i;
  };

  auto qc = py::class_<qc::QuantumComputation>(m, "QuantumComputation");

  ///---------------------------------------------------------------------------
  ///                           \n Constructors \n
  ///---------------------------------------------------------------------------
  qc.def(py::init<std::size_t, std::size_t, std::size_t>(), "nq"_a = 0U,
         "nc"_a = 0U, "seed"_a = 0U);

  // expose the static constructor from qasm strings or files
  qc.def_static("from_qasm_str", &qasm3::Importer::imports, "qasm"_a);
  qc.def_static("from_qasm", &qasm3::Importer::importf, "filename"_a);

  ///---------------------------------------------------------------------------
  ///                       \n General Properties \n
  ///---------------------------------------------------------------------------

  qc.def_property("name", &qc::QuantumComputation::getName,
                  &qc::QuantumComputation::setName);
  qc.def_property_readonly("num_qubits", &qc::QuantumComputation::getNqubits);
  qc.def_property_readonly("num_ancilla_qubits",
                           &qc::QuantumComputation::getNancillae);
  qc.def_property_readonly("num_garbage_qubits",
                           &qc::QuantumComputation::getNgarbageQubits);
  qc.def_property_readonly("num_measured_qubits",
                           &qc::QuantumComputation::getNmeasuredQubits);
  qc.def_property_readonly("num_data_qubits",
                           &qc::QuantumComputation::getNqubitsWithoutAncillae);
  qc.def_property_readonly("num_classical_bits",
                           &qc::QuantumComputation::getNcbits);
  qc.def_property_readonly("num_ops", &qc::QuantumComputation::getNops);
  qc.def("num_single_qubit_ops", &qc::QuantumComputation::getNsingleQubitOps);
  qc.def("num_total_ops", &qc::QuantumComputation::getNindividualOps);
  qc.def("depth", &qc::QuantumComputation::getDepth);
  qc.def_property("global_phase", &qc::QuantumComputation::getGlobalPhase,
                  &qc::QuantumComputation::gphase);
  qc.def("invert", &qc::QuantumComputation::invert);
  qc.def("to_operation", &qc::QuantumComputation::asOperation);

  ///---------------------------------------------------------------------------
  ///                  \n Mutable Sequence Interface \n
  ///---------------------------------------------------------------------------

  qc.def(
      "__getitem__",
      [&wrap](const qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        return circ.at(static_cast<SizeType>(i)).get();
      },
      py::return_value_policy::reference_internal, "idx"_a);
  qc.def(
      "__getitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        auto ops = std::vector<qc::Operation*>();
        ops.reserve(sliceLength);
        for (std::size_t i = start; i < stop; i += step) {
          ops.emplace_back(circ.at(i).get());
        }
        return ops;
      },
      py::return_value_policy::reference_internal, "slice"_a);
  qc.def(
      "__setitem__",
      [&wrap](qc::QuantumComputation& circ, DiffType i,
              const qc::Operation& op) {
        i = wrap(i, circ.getNops());
        circ.at(static_cast<SizeType>(i)) = op.clone();
      },
      "idx"_a, "op"_a);
  qc.def(
      "__setitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice,
         const std::vector<qc::Operation*>& ops) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        if (sliceLength != ops.size()) {
          throw std::runtime_error(
              "Length of slice and number of operations do not match.");
        }
        for (std::size_t i = 0; i < sliceLength; ++i) {
          circ.at(start) = ops[i]->clone();
          start += step;
        }
      },
      "slice"_a, "ops"_a);
  qc.def(
      "__delitem__",
      [&wrap](qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        circ.erase(circ.begin() + i);
      },
      "idx"_a);
  qc.def(
      "__delitem__",
      [](qc::QuantumComputation& circ, const py::slice& slice) {
        std::size_t start{};
        std::size_t stop{};
        std::size_t step{};
        std::size_t sliceLength{};
        if (!slice.compute(circ.getNops(), &start, &stop, &step,
                           &sliceLength)) {
          throw py::error_already_set();
        }
        // delete in reverse order to not invalidate indices
        for (std::size_t i = sliceLength; i > 0; --i) {
          circ.erase(circ.begin() +
                     static_cast<int64_t>(start + ((i - 1) * step)));
        }
      },
      "slice"_a);
  qc.def("__len__", &qc::QuantumComputation::getNops);
  qc.def(
      "insert",
      [](qc::QuantumComputation& circ, std::size_t idx,
         const qc::Operation& op) {
        circ.insert(circ.begin() + static_cast<int64_t>(idx), op.clone());
      },
      "idx"_a, "op"_a);
  qc.def(
      "append",
      [](qc::QuantumComputation& circ, const qc::Operation& op) {
        circ.emplace_back(op.clone());
      },
      "op"_a);
  qc.def("reverse", &qc::QuantumComputation::reverse);
  qc.def("clear", py::overload_cast<>(&qc::QuantumComputation::reset));

  ///---------------------------------------------------------------------------
  ///                         \n (Qu)Bit Registers \n
  ///---------------------------------------------------------------------------

  qc.def("add_qubit_register", &qc::QuantumComputation::addQubitRegister, "n"_a,
         "name"_a = "q");
  qc.def("add_classical_register",
         &qc::QuantumComputation::addClassicalRegister, "n"_a, "name"_a = "c");
  qc.def("add_ancillary_register",
         &qc::QuantumComputation::addAncillaryRegister, "n"_a,
         "name"_a = "anc");
  qc.def("unify_quantum_registers",
         &qc::QuantumComputation::unifyQuantumRegisters, "name"_a = "q");

  qc.def_property_readonly("qregs",
                           &qc::QuantumComputation::getQuantumRegisters);
  qc.def_property_readonly("cregs",
                           &qc::QuantumComputation::getClassicalRegisters);
  qc.def_property_readonly("ancregs",
                           &qc::QuantumComputation::getAncillaRegisters);

  ///---------------------------------------------------------------------------
  ///               \n Input Layout and Output Permutation \n
  ///---------------------------------------------------------------------------

  qc.def_readwrite("initial_layout", &qc::QuantumComputation::initialLayout);
  qc.def_readwrite("output_permutation",
                   &qc::QuantumComputation::outputPermutation);
  qc.def("initialize_io_mapping", &qc::QuantumComputation::initializeIOMapping);

  ///---------------------------------------------------------------------------
  ///                  \n Ancillary and Garbage Handling \n
  ///---------------------------------------------------------------------------

  qc.def_property_readonly(
      "ancillary", py::overload_cast<>(&qc::QuantumComputation::getAncillary));
  qc.def("set_circuit_qubit_ancillary",
         &qc::QuantumComputation::setLogicalQubitAncillary, "q"_a);
  qc.def("se_circuit_qubits_ancillary",
         &qc::QuantumComputation::setLogicalQubitsAncillary, "q_min"_a,
         "q_max"_a);
  qc.def("is_circuit_qubit_ancillary",
         &qc::QuantumComputation::logicalQubitIsAncillary, "q"_a);
  qc.def_property_readonly(
      "garbage", py::overload_cast<>(&qc::QuantumComputation::getGarbage));
  qc.def("set_circuit_qubit_garbage",
         &qc::QuantumComputation::setLogicalQubitGarbage, "q"_a);
  qc.def("set_circuit_qubits_garbage",
         &qc::QuantumComputation::setLogicalQubitsGarbage, "q_min"_a,
         "q_max"_a);
  qc.def("is_circuit_qubit_garbage",
         &qc::QuantumComputation::logicalQubitIsGarbage, "q"_a);

  ///---------------------------------------------------------------------------
  ///                    \n Symbolic Circuit Handling \n
  ///---------------------------------------------------------------------------

  qc.def_property_readonly("variables", &qc::QuantumComputation::getVariables);
  qc.def("add_variable", &qc::QuantumComputation::addVariable, "var"_a);
  qc.def(
      "add_variables",
      [](qc::QuantumComputation& circ,
         const std::vector<qc::SymbolOrNumber>& vars) {
        for (const auto& var : vars) {
          circ.addVariable(var);
        }
      },
      "vars_"_a);
  qc.def("is_variable_free", &qc::QuantumComputation::isVariableFree);
  qc.def("instantiate", &qc::QuantumComputation::instantiate, "assignment"_a);
  qc.def("instantiate_inplace", &qc::QuantumComputation::instantiateInplace,
         "assignment"_a);

  ///---------------------------------------------------------------------------
  ///                       \n Output Handling \n
  ///---------------------------------------------------------------------------

  qc.def("qasm2_str",
         [](const qc::QuantumComputation& circ) { return circ.toQASM(false); });
  qc.def(
      "qasm2",
      [](const qc::QuantumComputation& circ, const std::string& filename) {
        circ.dump(filename, qc::Format::OpenQASM2);
      },
      "filename"_a);
  qc.def("qasm3_str",
         [](const qc::QuantumComputation& circ) { return circ.toQASM(true); });
  qc.def(
      "qasm3",
      [](const qc::QuantumComputation& circ, const std::string& filename) {
        circ.dump(filename, qc::Format::OpenQASM3);
      },
      "filename"_a);
  qc.def("__str__", [](const qc::QuantumComputation& circ) {
    auto ss = std::stringstream();
    circ.print(ss);
    return ss.str();
  });
  qc.def("__repr__", [](const qc::QuantumComputation& circ) {
    auto ss = std::stringstream();
    ss << "QuantumComputation(num_qubits=" << circ.getNqubits()
       << ", num_bits=" << circ.getNcbits() << ", num_ops=" << circ.getNops()
       << ")";
    circ.print(ss);
    return ss.str();
  });

  ///---------------------------------------------------------------------------
  ///                            \n Operations \n
  ///---------------------------------------------------------------------------

#define DEFINE_SINGLE_TARGET_OPERATION(op)                                     \
  qc.def(#op, &qc::QuantumComputation::op, "q"_a);                             \
  qc.def("c" #op, &qc::QuantumComputation::c##op, "control"_a, "target"_a);    \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, "controls"_a, "target"_a);

  DEFINE_SINGLE_TARGET_OPERATION(i)
  DEFINE_SINGLE_TARGET_OPERATION(x)
  DEFINE_SINGLE_TARGET_OPERATION(y)
  DEFINE_SINGLE_TARGET_OPERATION(z)
  DEFINE_SINGLE_TARGET_OPERATION(h)
  DEFINE_SINGLE_TARGET_OPERATION(s)
  DEFINE_SINGLE_TARGET_OPERATION(sdg)
  DEFINE_SINGLE_TARGET_OPERATION(t)
  DEFINE_SINGLE_TARGET_OPERATION(tdg)
  DEFINE_SINGLE_TARGET_OPERATION(v)
  DEFINE_SINGLE_TARGET_OPERATION(vdg)
  DEFINE_SINGLE_TARGET_OPERATION(sx)
  DEFINE_SINGLE_TARGET_OPERATION(sxdg)

#define DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(op, param)             \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param), "q"_a);            \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param),             \
         "control"_a, "target"_a);                                             \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param),           \
         "controls"_a, "target"_a);

  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rx, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(ry, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(rz, theta)
  DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION(p, theta)

#define DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)       \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         "q"_a);                                                               \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), "control"_a, "target"_a);                           \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), "controls"_a, "target"_a);

  DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION(u2, phi, lambda_)

#define DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(op, param0, param1,     \
                                                       param2)                 \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         py::arg(#param2), "q"_a);                                             \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), py::arg(#param2), "control"_a, "target"_a);         \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), py::arg(#param2), "controls"_a, "target"_a);

  DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION(u, theta, phi, lambda_)

#define DEFINE_TWO_TARGET_OPERATION(op)                                        \
  qc.def(#op, &qc::QuantumComputation::op, "target1"_a, "target2"_a);          \
  qc.def("c" #op, &qc::QuantumComputation::c##op, "control"_a, "target1"_a,    \
         "target2"_a);                                                         \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, "controls"_a, "target1"_a, \
         "target2"_a);

  DEFINE_TWO_TARGET_OPERATION(swap)
  DEFINE_TWO_TARGET_OPERATION(dcx)
  DEFINE_TWO_TARGET_OPERATION(ecr)
  DEFINE_TWO_TARGET_OPERATION(iswap)
  DEFINE_TWO_TARGET_OPERATION(peres)
  DEFINE_TWO_TARGET_OPERATION(peresdg)

#define DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(op, param)                \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param), "target1"_a,       \
         "target2"_a);                                                         \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param),             \
         "control"_a, "target1"_a, "target2"_a);                               \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param),           \
         "controls"_a, "target1"_a, "target2"_a);

  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rxx, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(ryy, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzz, theta)
  DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION(rzx, theta)

#define DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(op, param0, param1)          \
  qc.def(#op, &qc::QuantumComputation::op, py::arg(#param0), py::arg(#param1), \
         "target1"_a, "target2"_a);                                            \
  qc.def("c" #op, &qc::QuantumComputation::c##op, py::arg(#param0),            \
         py::arg(#param1), "control"_a, "target1"_a, "target2"_a);             \
  qc.def("mc" #op, &qc::QuantumComputation::mc##op, py::arg(#param0),          \
         py::arg(#param1), "controls"_a, "target1"_a, "target2"_a);

  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_minus_yy, theta, beta)
  DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION(xx_plus_yy, theta, beta)

#undef DEFINE_SINGLE_TARGET_OPERATION
#undef DEFINE_SINGLE_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_TWO_PARAMETER_OPERATION
#undef DEFINE_SINGLE_TARGET_THREE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_OPERATION
#undef DEFINE_TWO_TARGET_SINGLE_PARAMETER_OPERATION
#undef DEFINE_TWO_TARGET_TWO_PARAMETER_OPERATION

  qc.def("gphase", &qc::QuantumComputation::gphase, "phase"_a);

  qc.def("measure",
         py::overload_cast<qc::Qubit, std::size_t>(
             &qc::QuantumComputation::measure),
         "qubit"_a, "cbit"_a);
  qc.def("measure",
         py::overload_cast<const std::vector<qc::Qubit>&,
                           const std::vector<qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "qubits"_a, "cbits"_a);
  qc.def("measure_all", &qc::QuantumComputation::measureAll, py::kw_only(),
         "add_bits"_a = true);

  qc.def("reset", py::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset),
         "q"_a);
  qc.def("reset",
         py::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::reset),
         "qubits"_a);

  qc.def("barrier", py::overload_cast<>(&qc::QuantumComputation::barrier));
  qc.def("barrier",
         py::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier), "q"_a);
  qc.def("barrier", py::overload_cast<const std::vector<qc::Qubit>&>(
                        &qc::QuantumComputation::barrier));

  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "creg"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{});
  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "control"_a, "creg"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{});
  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Controls&,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "controls"_a, "creg"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{});
  qc.def("classic_controlled",
         py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Bit,
                           const std::uint64_t, const qc::ComparisonKind,
                           const std::vector<qc::fp>&>(
             &qc::QuantumComputation::classicControlled),
         "op"_a, "target"_a, "cbit"_a, "expected_value"_a = 1U,
         "comparison_kind"_a = qc::ComparisonKind::Eq,
         "params"_a = std::vector<qc::fp>{});
  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                        const qc::Bit, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "control"_a, "cbit"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{});
  qc.def(
      "classic_controlled",
      py::overload_cast<const qc::OpType, const qc::Qubit, const qc::Controls&,
                        const qc::Bit, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::classicControlled),
      "op"_a, "target"_a, "controls"_a, "cbit"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{});
}
} // namespace mqt
