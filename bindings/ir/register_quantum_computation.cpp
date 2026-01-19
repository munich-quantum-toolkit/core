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
#include "ir/QuantumComputation.hpp"
#include "ir/operations/Control.hpp"
#include "ir/operations/Expression.hpp"
#include "ir/operations/IfElseOperation.hpp"
#include "ir/operations/OpType.hpp"
#include "ir/operations/Operation.hpp"
#include "qasm3/Importer.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <nanobind/nanobind.h>
#include <nanobind/stl/set.h>           // NOLINT(misc-include-cleaner)
#include <nanobind/stl/string.h>        // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unique_ptr.h>    // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_map.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/unordered_set.h> // NOLINT(misc-include-cleaner)
#include <nanobind/stl/variant.h>       // NOLINT(misc-include-cleaner)
#include <nanobind/stl/vector.h>        // NOLINT(misc-include-cleaner)
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace mqt {

namespace nb = nanobind;
using namespace nb::literals;

using DiffType = std::vector<std::unique_ptr<qc::Operation>>::difference_type;
using SizeType = std::vector<std::unique_ptr<qc::Operation>>::size_type;

// NOLINTNEXTLINE(misc-use-internal-linkage)
void registerQuantumComputation(const nb::module_& m) {
  auto wrap = [](DiffType i, const SizeType size) {
    if (i < 0) {
      i += static_cast<DiffType>(size);
    }
    if (i < 0 || std::cmp_greater_equal(i, size)) {
      throw nb::index_error();
    }
    return i;
  };

  auto qc = nb::class_<qc::QuantumComputation>(
      m, "QuantumComputation",
      nb::sig("class "
              "QuantumComputation(collections.abc.MutableSequence[mqt.core.ir."
              "operations.Operation])"),
      R"pb(The main class for representing quantum computations within the MQT.

Acts as mutable sequence of :class:`~mqt.core.ir.operations.Operation` objects, which represent the individual operations in the quantum computation.

Args:
    nq: The number of qubits in the quantum computation.
    nc: The number of classical bits in the quantum computation.
    seed: The seed to use for the internal random number generator.)pb");

  ///---------------------------------------------------------------------------
  ///                           \n Constructors \n
  ///---------------------------------------------------------------------------
  qc.def(nb::init<std::size_t, std::size_t, std::size_t>(), "nq"_a = 0U,
         "nc"_a = 0U, "seed"_a = 0U);

  // expose the static constructor from qasm strings or files
  qc.def_static("from_qasm_str", &qasm3::Importer::imports, "qasm"_a,
                R"pb(Create a QuantumComputation object from an OpenQASM string.

Args:
    qasm: The OpenQASM string to create the QuantumComputation object from.

Returns:
    The QuantumComputation object created from the OpenQASM string.)pb");

  qc.def_static("from_qasm", &qasm3::Importer::importf, "filename"_a,
                R"pb(Create a QuantumComputation object from an OpenQASM file.

Args:
    filename: The filename of the OpenQASM file to create the QuantumComputation object from.

Returns:
    The QuantumComputation object created from the OpenQASM file.)pb");

  ///---------------------------------------------------------------------------
  ///                       \n General Properties \n
  ///---------------------------------------------------------------------------

  qc.def_prop_rw("name", &qc::QuantumComputation::getName,
                 &qc::QuantumComputation::setName,
                 "The name of the quantum computation.");

  qc.def_prop_ro("num_qubits", &qc::QuantumComputation::getNqubits,
                 "The total number of qubits in the quantum computation.");

  qc.def_prop_ro("num_ancilla_qubits", &qc::QuantumComputation::getNancillae,
                 R"pb(The number of ancilla qubits in the quantum computation.

Note:
    Ancilla qubits are qubits that always start in a fixed state (usually :math:`|0\\rangle`).)pb");

  qc.def_prop_ro("num_garbage_qubits",
                 &qc::QuantumComputation::getNgarbageQubits,
                 R"pb(The number of garbage qubits in the quantum computation.

Note:
    Garbage qubits are qubits whose final state is not relevant for the computation.)pb");

  qc.def_prop_ro(
      "num_measured_qubits", &qc::QuantumComputation::getNmeasuredQubits,
      R"pb(The number of qubits that are measured in the quantum computation.

Computed as :math:`| \text{qubits} | - | \text{garbage} |`.)pb");

  qc.def_prop_ro("num_data_qubits",
                 &qc::QuantumComputation::getNqubitsWithoutAncillae,
                 R"pb(The number of data qubits in the quantum computation.

Computed as :math:`| \text{qubits} | - | \text{ancilla} |`.)pb");

  qc.def_prop_ro("num_classical_bits", &qc::QuantumComputation::getNcbits,
                 "The number of classical bits in the quantum computation.");

  qc.def_prop_ro("num_ops", &qc::QuantumComputation::getNops,
                 "The number of operations in the quantum computation.");

  qc.def("num_single_qubit_ops", &qc::QuantumComputation::getNsingleQubitOps,
         "Return the number of single-qubit operations in the quantum "
         "computation.");

  qc.def("num_total_ops", &qc::QuantumComputation::getNindividualOps,
         R"pb(Return the total number of operations in the quantum computation.

Recursively counts sub-operations (e.g., from :class:`~mqt.core.ir.operations.CompoundOperation` objects).)pb");

  qc.def("depth", &qc::QuantumComputation::getDepth,
         "Return the depth of the quantum computation.");

  qc.def_prop_rw("global_phase", &qc::QuantumComputation::getGlobalPhase,
                 &qc::QuantumComputation::gphase,
                 "The global phase of the quantum computation.");

  qc.def("invert", &qc::QuantumComputation::invert,
         "Invert the quantum computation in-place by inverting each operation "
         "and reversing the order of operations.");

  qc.def("to_operation", &qc::QuantumComputation::asOperation,
         R"pb(Convert the quantum computation to a single operation.

This gives ownership of the operations to the resulting operation, so the quantum computation will be empty after this operation.

When the quantum computation contains more than one operation, the resulting operation is a :class:`~mqt.core.ir.operations.CompoundOperation`.

Returns:
    The operation representing the quantum computation.)pb");

  ///---------------------------------------------------------------------------
  ///                  \n Mutable Sequence Interface \n
  ///---------------------------------------------------------------------------

  qc.def(
      "__getitem__",
      [wrap](const qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        return circ.at(static_cast<SizeType>(i)).get();
      },
      nb::rv_policy::reference_internal, "index"_a,
      R"pb(Get the operation at the given index.

Note:
    This gives write access to the operation at the given index.

Args:
    index: The index of the operation to get.

Returns:
    The operation at the given index.)pb");

  qc.def(
      "__getitem__",
      [](qc::QuantumComputation& circ, const nb::slice& slice) {
        auto [start, stop, step, sliceLength] = slice.compute(circ.getNops());
        auto ops = std::vector<qc::Operation*>();
        ops.reserve(sliceLength);
        for (std::size_t i = 0; i < sliceLength; ++i) {
          auto idx =
              static_cast<DiffType>(start) + (static_cast<DiffType>(i) * step);
          ops.emplace_back(circ.at(static_cast<SizeType>(idx)).get());
        }
        return ops;
      },
      nb::rv_policy::reference_internal, "index"_a,
      R"pb(Get a slice of operations from the quantum computation.

Note:
    This gives write access to the operations in the given slice.

Args:
    index: The slice of operations to get.

Returns:
    The operations in the given slice.)pb");

  qc.def(
      "__setitem__",
      [wrap](qc::QuantumComputation& circ, DiffType i,
             const qc::Operation& op) {
        i = wrap(i, circ.getNops());
        circ.at(static_cast<SizeType>(i)) = op.clone();
      },
      "index"_a, "value"_a, R"pb(Set the operation at the given index.

Args:
    index: The index of the operation to set.
    value: The operation to set at the given index.)pb");

  qc.def(
      "__setitem__",
      [](qc::QuantumComputation& circ, const nb::slice& slice,
         const std::vector<qc::Operation*>& ops) {
        auto [start, stop, step, sliceLength] = slice.compute(circ.getNops());
        if (sliceLength != ops.size()) {
          throw std::runtime_error(
              "Length of slice and number of operations do not match.");
        }
        for (std::size_t i = 0; i < sliceLength; ++i) {
          assert(ops[i] != nullptr && "ops must not contain nullptr");
          circ.at(static_cast<SizeType>(start)) = ops[i]->clone();
          start += step;
        }
      },
      nb::sig("def __setitem__(self, index: slice, value: "
              "collections.abc.Iterable[mqt.core.ir.operations.Operation]) -> "
              "None"),
      R"pb(Set the operations in the given slice.

Args:
    index: The slice of operations to set.
    value: The operations to set in the given slice.)pb");

  qc.def(
      "__delitem__",
      [wrap](qc::QuantumComputation& circ, DiffType i) {
        i = wrap(i, circ.getNops());
        circ.erase(circ.begin() + i);
      },
      "index"_a, R"pb(Delete the operation at the given index.

Args:
    index: The index of the operation to delete.)pb");

  qc.def(
      "__delitem__",
      [](qc::QuantumComputation& circ, const nb::slice& slice) {
        auto [start, stop, step, sliceLength] = slice.compute(circ.getNops());
        // Delete in reverse order to not invalidate indices
        std::vector<DiffType> indices;
        indices.reserve(sliceLength);
        for (std::size_t i = 0; i < sliceLength; ++i) {
          indices.emplace_back(static_cast<DiffType>(start) +
                               (static_cast<DiffType>(i) * step));
        }
        std::ranges::sort(indices, std::greater<>());
        for (const auto idx : indices) {
          circ.erase(circ.begin() + idx);
        }
      },
      "index"_a, R"pb(Delete the operations in the given slice.

Args:
    index: The slice of operations to delete.)pb");

  qc.def("__len__", &qc::QuantumComputation::getNops,
         "Return the number of operations in the quantum computation.");

  qc.def(
      "insert",
      [](qc::QuantumComputation& circ, std::size_t idx,
         const qc::Operation& op) {
        circ.insert(circ.begin() + static_cast<int64_t>(idx), op.clone());
      },
      "index"_a, "value"_a, R"pb(Insert an operation at the given index.

Args:
    index: The index to insert the operation at.
    value: The operation to insert.)pb");

  qc.def(
      "append",
      [](qc::QuantumComputation& circ, const qc::Operation& op) {
        circ.emplace_back(op.clone());
      },
      "value"_a, R"pb(Append an operation to the end of the quantum computation.

Args:
    value: The operation to append.)pb");

  qc.def("reverse", &qc::QuantumComputation::reverse,
         "Reverse the order of the operations in the quantum computation "
         "(in-place).");

  qc.def("clear", nb::overload_cast<>(&qc::QuantumComputation::reset),
         "Clear the quantum computation of all operations.");

  ///---------------------------------------------------------------------------
  ///                         \n (Qu)Bit Registers \n
  ///---------------------------------------------------------------------------

  qc.def("add_qubit_register", &qc::QuantumComputation::addQubitRegister, "n"_a,
         "name"_a = "q", R"pb(Add a qubit register to the quantum computation.

Args:
    n: The number of qubits in the qubit register.
    name: The name of the qubit register.

Returns:
    The qubit register added to the quantum computation.)pb");

  qc.def("add_classical_register",
         &qc::QuantumComputation::addClassicalRegister, "n"_a, "name"_a = "c",
         R"pb(Add a classical register to the quantum computation.

Args:
    n: The number of bits in the classical register.
    name: The name of the classical register.

Returns:
    The classical register added to the quantum computation.)pb");

  qc.def("add_ancillary_register",
         &qc::QuantumComputation::addAncillaryRegister, "n"_a, "name"_a = "anc",
         R"pb(Add an ancillary register to the quantum computation.

Args:
    n: The number of qubits in the ancillary register.
    name: The name of the ancillary register.

Returns:
    The ancillary register added to the quantum computation.)pb");

  qc.def("unify_quantum_registers",
         &qc::QuantumComputation::unifyQuantumRegisters, "name"_a = "q",
         R"pb(Unify all quantum registers in the quantum computation.

Args:
    name: The name of the unified quantum register.

Returns:
    The unified quantum register.)pb");

  qc.def_prop_ro("qregs", &qc::QuantumComputation::getQuantumRegisters,
                 "The quantum registers in the quantum computation.");

  qc.def_prop_ro("cregs", &qc::QuantumComputation::getClassicalRegisters,
                 "The classical registers in the quantum computation.");

  qc.def_prop_ro("ancregs", &qc::QuantumComputation::getAncillaRegisters,
                 "The ancillary registers in the quantum computation.");

  ///---------------------------------------------------------------------------
  ///               \n Input Layout and Output Permutation \n
  ///---------------------------------------------------------------------------

  qc.def_rw("initial_layout", &qc::QuantumComputation::initialLayout,
            R"pb(The initial layout of the qubits in the quantum computation.

This is a permutation of the qubits in the quantum computation.
It is mainly used to track the mapping of circuit qubits to device qubits during quantum circuit compilation.
The keys are the device qubits (in which a compiled circuit is expressed in), and the values are the circuit qubits (in which the original quantum circuit is expressed in).

Any operations in the quantum circuit are expected to be expressed in terms of the keys of the initial layout.

Examples:
    - If no initial layout is explicitly specified (which is the default), the initial layout is assumed to be the identity permutation.
    - Assume a three-qubit circuit has been compiled to a four qubit device and circuit qubit 0 is mapped to device qubit 1, circuit qubit 1 is mapped to device qubit 2, and circuit qubit 2 is mapped to device qubit 3.
      Then the initial layout is {1: 0, 2: 1, 3: 2}.)pb");

  qc.def_rw(
      "output_permutation", &qc::QuantumComputation::outputPermutation,
      R"pb(The output permutation of the qubits in the quantum computation.

This is a permutation of the qubits in the quantum computation.
It is mainly used to track where individual qubits end up at the end of the quantum computation, for example after a circuit has been compiled to a specific device and SWAP gates have been inserted, which permute the qubits.
Similar to the initial layout, the keys are the qubits in the circuit and the values are the qubits in the "original" circuit.

Examples:
    - If no output permutation is explicitly specified and the circuit does not contain measurements at the end, the output permutation is assumed to be the identity permutation.
    - If the circuit contains measurements at the end, these measurements are used to infer the output permutation.
      Assume a three-qubit circuit has been compiled to a four qubit device and, at the end of the circuit, circuit qubit 0 is measured into classical bit 2, circuit qubit 1 is measured into classical bit 1, and circuit qubit 3 is measured into classical bit 0.
      Then the output permutation is {0: 2, 1: 1, 3: 0}.)pb");

  qc.def("initialize_io_mapping", &qc::QuantumComputation::initializeIOMapping,
         R"pb(Initialize the I/O mapping of the quantum computation.

If no initial layout is explicitly specified, the initial layout is assumed to be the identity permutation.
If the circuit contains measurements at the end, these measurements are used to infer the output permutation.)pb");

  ///---------------------------------------------------------------------------
  ///                  \n Ancillary and Garbage Handling \n
  ///---------------------------------------------------------------------------

  qc.def_prop_ro(
      "ancillary", nb::overload_cast<>(&qc::QuantumComputation::getAncillary),
      "A list of booleans indicating whether each qubit is ancillary.");

  qc.def("set_circuit_qubit_ancillary",
         &qc::QuantumComputation::setLogicalQubitAncillary, "q"_a,
         R"pb(Set a circuit (i.e., logical) qubit to be ancillary.

Args:
    q: The index of the circuit qubit to set as ancillary.)pb");

  qc.def("set_circuit_qubits_ancillary",
         &qc::QuantumComputation::setLogicalQubitsAncillary, "q_min"_a,
         "q_max"_a,
         R"pb(Set a range of circuit (i.e., logical) qubits to be ancillary.

Args:
    q_min: The minimum index of the circuit qubits to set as ancillary.
    q_max: The maximum index of the circuit qubits to set as ancillary.)pb");

  qc.def("is_circuit_qubit_ancillary",
         &qc::QuantumComputation::logicalQubitIsAncillary, "q"_a,
         R"pb(Check if a circuit (i.e., logical) qubit is ancillary.

Args:
    q: The index of the circuit qubit to check.

Returns:
    True if the circuit qubit is ancillary, False otherwise.)pb");

  qc.def_prop_ro(
      "garbage", nb::overload_cast<>(&qc::QuantumComputation::getGarbage),
      "A list of booleans indicating whether each qubit is garbage.");

  qc.def("set_circuit_qubit_garbage",
         &qc::QuantumComputation::setLogicalQubitGarbage, "q"_a,
         R"pb(Set a circuit (i.e., logical) qubit to be garbage.

Args:
    q: The index of the circuit qubit to set as garbage.)pb");

  qc.def("set_circuit_qubits_garbage",
         &qc::QuantumComputation::setLogicalQubitsGarbage, "q_min"_a, "q_max"_a,
         R"pb(Set a range of circuit (i.e., logical) qubits to be garbage.

Args:
    q_min: The minimum index of the circuit qubits to set as garbage.
    q_max: The maximum index of the circuit qubits to set as garbage.)pb");

  qc.def("is_circuit_qubit_garbage",
         &qc::QuantumComputation::logicalQubitIsGarbage, "q"_a,
         R"pb(Check if a circuit (i.e., logical) qubit is garbage.

Args:
    q: The index of the circuit qubit to check.

Returns:
    True if the circuit qubit is garbage, False otherwise.)pb");

  ///---------------------------------------------------------------------------
  ///                    \n Symbolic Circuit Handling \n
  ///---------------------------------------------------------------------------

  qc.def_prop_ro("variables", &qc::QuantumComputation::getVariables,
                 "The set of variables in the quantum computation.");

  qc.def("add_variable", &qc::QuantumComputation::addVariable, "var"_a,
         R"pb(Add a variable to the quantum computation.

Args:
    var: The variable to add.)pb");

  qc.def(
      "add_variables",
      [](qc::QuantumComputation& circ,
         const std::vector<qc::SymbolOrNumber>& vars) {
        for (const auto& var : vars) {
          circ.addVariable(var);
        }
      },
      "vars_"_a, R"pb(Add multiple variables to the quantum computation.

Args:
    vars_: The variables to add.)pb");

  qc.def("is_variable_free", &qc::QuantumComputation::isVariableFree,
         R"pb(Check if the quantum computation is free of variables.

Returns:
    True if the quantum computation is free of variables, False otherwise.)pb");

  qc.def(
      "instantiate", &qc::QuantumComputation::instantiate, "assignment"_a,
      R"pb(Instantiate the quantum computation with the given variable assignment.

Args:
    assignment: The variable assignment to instantiate the quantum computation with.

Returns:
    The instantiated quantum computation.)pb");

  qc.def(
      "instantiate_inplace", &qc::QuantumComputation::instantiateInplace,
      "assignment"_a,
      R"pb(Instantiate the quantum computation with the given variable assignment in-place.

Args:
    assignment: The variable assignment to instantiate the quantum computation with.)pb");

  ///---------------------------------------------------------------------------
  ///                       \n Output Handling \n
  ///---------------------------------------------------------------------------

  qc.def(
      "qasm2_str",
      [](const qc::QuantumComputation& circ) { return circ.toQASM(false); },
      R"pb(Return the OpenQASM2 representation of the quantum computation as a string.

Note:
    This uses some custom extensions to OpenQASM 2.0 that allow for easier definition of multi-controlled gates.
    These extensions might not be supported by all OpenQASM 2.0 parsers.
    Consider using the :meth:`qasm3_str` method instead, which uses OpenQASM 3.0 that natively supports multi-controlled gates.
    The export also assumes the bigger, non-standard `qelib1.inc` from Qiskit is available.

Returns:
    The OpenQASM2 representation of the quantum computation as a string.)pb");

  qc.def(
      "qasm2",
      [](const qc::QuantumComputation& circ, const std::string& filename) {
        circ.dump(filename, qc::Format::OpenQASM2);
      },
      "filename"_a,
      nb::sig("def qasm2(self, filename: os.PathLike[str] | str) -> None"),
      R"pb(Write the OpenQASM2 representation of the quantum computation to a file.

See Also:
    :meth:`qasm2_str`

Args:
    filename: The filename of the file to write the OpenQASM2 representation to.)pb");

  qc.def(
      "qasm3_str",
      [](const qc::QuantumComputation& circ) { return circ.toQASM(true); },
      R"pb(Return the OpenQASM3 representation of the quantum computation as a string.

Returns:
    The OpenQASM3 representation of the quantum computation as a string.)pb");

  qc.def(
      "qasm3",
      [](const qc::QuantumComputation& circ, const std::string& filename) {
        circ.dump(filename, qc::Format::OpenQASM3);
      },
      "filename"_a,
      nb::sig("def qasm3(self, filename: os.PathLike[str] | str) -> None"),
      R"pb(Write the OpenQASM3 representation of the quantum computation to a file.

See Also:
    :meth:`qasm3_str`

Args:
    filename: The filename of the file to write the OpenQASM3 representation to.)pb");

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

  // I

  qc.def("i", &qc::QuantumComputation::i, "q"_a,
         R"pb(Apply an identity operation.

.. math::
    I = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("ci", &qc::QuantumComputation::ci, "control"_a, "target"_a,
         nb::sig("def ci(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled identity operation.

See Also:
    :meth:`i`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mci", &qc::QuantumComputation::mci, "controls"_a, "target"_a,
         nb::sig("def mci(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled identity operation.

See Also:
    :meth:`i`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // X

  qc.def("x", &qc::QuantumComputation::x, "q"_a,
         R"pb(Apply a Pauli-X gate.

.. math::
    X = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cx", &qc::QuantumComputation::cx, "control"_a, "target"_a,
         nb::sig("def cx(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled Pauli-X (i.e., CNOT or CX) gate.

See Also:
    :meth:`x`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcx", &qc::QuantumComputation::mcx, "controls"_a, "target"_a,
         nb::sig("def mcx(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled Pauli-X (i.e., Toffoli or MCX) gate.

See Also:
    :meth:`x`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // Y

  qc.def("y", &qc::QuantumComputation::y, "q"_a,
         R"pb(Apply a Pauli-Y gate.

.. math::
    Y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cy", &qc::QuantumComputation::cy, "control"_a, "target"_a,
         nb::sig("def cy(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled Pauli-Y gate.

See Also:
    :meth:`y`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcy", &qc::QuantumComputation::mcy, "controls"_a, "target"_a,
         nb::sig("def mcy(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled Pauli-Y gate.

See Also:
    :meth:`y`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // Z

  qc.def("z", &qc::QuantumComputation::z, "q"_a,
         R"pb(Apply a Pauli-Z gate.

.. math::
    Z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cz", &qc::QuantumComputation::cz, "control"_a, "target"_a,
         nb::sig("def cz(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled Pauli-Z gate.

See Also:
    :meth:`z`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcz", &qc::QuantumComputation::mcz, "controls"_a, "target"_a,
         nb::sig("def mcz(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled Pauli-Z gate.

See Also:
    :meth:`z`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // H

  qc.def("h", &qc::QuantumComputation::h, "q"_a,
         R"pb(Apply a Hadamard gate.

.. math::
    H = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & 1 \\ 1 & -1 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("ch", &qc::QuantumComputation::ch, "control"_a, "target"_a,
         nb::sig("def ch(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled Hadamard gate.

See Also:
    :meth:`h`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mch", &qc::QuantumComputation::mch, "controls"_a, "target"_a,
         nb::sig("def mch(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled Hadamard gate.

See Also:
    :meth:`h`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // S

  qc.def("s", &qc::QuantumComputation::s, "q"_a,
         R"pb(Apply an S (i.e., phase) gate.

.. math::
    S = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cs", &qc::QuantumComputation::cs, "control"_a, "target"_a,
         nb::sig("def cs(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled S gate.

See Also:
    :meth:`s`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcs", &qc::QuantumComputation::mcs, "controls"_a, "target"_a,
         nb::sig("def mcs(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled S gate.

See Also:
    :meth:`s`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // Sdg

  qc.def("sdg", &qc::QuantumComputation::sdg, "q"_a,
         R"pb(Apply an :math:`S^\dagger` gate.

.. math::
    S^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & -i \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("csdg", &qc::QuantumComputation::csdg, "control"_a, "target"_a,
         nb::sig("def csdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`S^\dagger` gate.

See Also:
    :meth:`sdg`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcsdg", &qc::QuantumComputation::mcsdg, "controls"_a, "target"_a,
         nb::sig("def mcsdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled :math:`S^\dagger` gate.

See Also:
    :meth:`sdg`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // T

  qc.def("t", &qc::QuantumComputation::t, "q"_a,
         R"pb(Apply a T gate.

.. math::
    T = \begin{pmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("ct", &qc::QuantumComputation::ct, "control"_a, "target"_a,
         nb::sig("def ct(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled T gate.

See Also:
    :meth:`t`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mct", &qc::QuantumComputation::mct, "controls"_a, "target"_a,
         nb::sig("def mct(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled T gate.

See Also:
    :meth:`t`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // Tdg

  qc.def("tdg", &qc::QuantumComputation::tdg, "q"_a,
         R"pb(Apply a :math:`T^\dagger` gate.

.. math::
    T^\dagger = \begin{pmatrix} 1 & 0 \\ 0 & e^{-i \pi / 4} \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("ctdg", &qc::QuantumComputation::ctdg, "control"_a, "target"_a,
         nb::sig("def ctdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`T^\dagger` gate.

See Also:
    :meth:`tdg`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mctdg", &qc::QuantumComputation::mctdg, "controls"_a, "target"_a,
         nb::sig("def mctdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled :math:`T^\dagger` gate.

See Also:
    :meth:`tdg`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // V

  qc.def("v", &qc::QuantumComputation::v, "q"_a,
         R"pb(Apply a V gate.

.. math::
    V = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -i \\ -i & 1 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cv", &qc::QuantumComputation::cv, "control"_a, "target"_a,
         nb::sig("def cv(self, control: mqt.core.ir.operations.Control | int, "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a controlled V gate.

See Also:
    :meth:`v`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcv", &qc::QuantumComputation::mcv, "controls"_a, "target"_a,
         nb::sig("def mcv(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled V gate.

See Also:
    :meth:`v`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // Vdg

  qc.def("vdg", &qc::QuantumComputation::vdg, "q"_a,
         R"pb(Apply a :math:`V^\dagger` gate.

.. math::
    V^\dagger = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & i \\ i & 1 \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("cvdg", &qc::QuantumComputation::cvdg, "control"_a, "target"_a,
         nb::sig("def cvdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`V^\dagger` gate.

See Also:
    :meth:`vdg`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcvdg", &qc::QuantumComputation::mcvdg, "controls"_a, "target"_a,
         nb::sig("def mcvdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled :math:`V^\dagger` gate.

See Also:
    :meth:`vdg`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // SX

  qc.def("sx", &qc::QuantumComputation::sx, "q"_a,
         R"pb(Apply a :math:`\sqrt{X}` gate.

.. math::
    \sqrt{X} = \frac{1}{2} \begin{pmatrix} 1 + i & 1 - i \\ 1 - i & 1 + i \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("csx", &qc::QuantumComputation::csx, "control"_a, "target"_a,
         nb::sig("def csx(self, control: mqt.core.ir.operations.Control | int, "
                 "target: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`\sqrt{X}` gate.

See Also:
    :meth:`sx`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcsx", &qc::QuantumComputation::mcsx, "controls"_a, "target"_a,
         nb::sig("def mcsx(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target: int) -> None"),
         R"pb(Apply a multi-controlled :math:`\sqrt{X}` gate.

See Also:
    :meth:`sx`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // SXdg

  qc.def("sxdg", &qc::QuantumComputation::sxdg, "q"_a,
         R"pb(Apply a :math:`\sqrt{X}^\dagger` gate.

.. math::
    \sqrt{X}^{\dagger} = \frac{1}{2} \begin{pmatrix} 1 - i & 1 + i \\ 1 + i & 1 - i \end{pmatrix}

Args:
    q: The target qubit)pb");
  qc.def("csxdg", &qc::QuantumComputation::csxdg, "control"_a, "target"_a,
         nb::sig("def csxdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`\sqrt{X}^\dagger` gate.

See Also:
    :meth:`sxdg`

Args:
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcsxdg", &qc::QuantumComputation::mcsxdg, "controls"_a, "target"_a,
         nb::sig("def mcsxdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target: int) -> None"),
         R"pb(Apply a multi-controlled :math:`\sqrt{X}^\dagger` gate.

See Also:
    :meth:`sxdg`

Args:
    controls: The control qubits
    target: The target qubit)pb");

  // RX

  qc.def("rx", &qc::QuantumComputation::rx, "theta"_a, "q"_a,
         R"pb(Apply an :math:`R_x(\theta)` gate.

.. math::
    R_x(\theta) = e^{-i \theta X / 2} = \cos(\theta / 2) I - i \sin(\theta / 2) X
                = \begin{pmatrix} \cos(\theta / 2) & -i \sin(\theta / 2) \\ -i \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    q: The target qubit)pb");
  qc.def("crx", &qc::QuantumComputation::crx, "theta"_a, "control"_a,
         "target"_a,
         nb::sig("def crx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target: int) -> None"),
         R"pb(Apply a controlled :math:`R_x(\theta)` gate.

See Also:
    :meth:`rx`

Args:
    theta: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcrx", &qc::QuantumComputation::mcrx, "theta"_a, "controls"_a,
         "target"_a,
         nb::sig("def mcrx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a multi-controlled :math:`R_x(\theta)` gate.

See Also:
    :meth:`rx`

Args:
    theta: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // RY

  qc.def("ry", &qc::QuantumComputation::ry, "theta"_a, "q"_a,
         R"pb(Apply an :math:`R_y(\theta)` gate.

.. math::
    R_y(\theta) = e^{-i \theta Y / 2} = \cos(\theta / 2) I - i \sin(\theta / 2) Y
                = \begin{pmatrix} \cos(\theta / 2) & -\sin(\theta / 2) \\ \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    q: The target qubit)pb");
  qc.def("cry", &qc::QuantumComputation::cry, "theta"_a, "control"_a,
         "target"_a,
         nb::sig("def cry(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target: int) -> None"),
         R"pb(Apply a controlled :math:`R_y(\theta)` gate.

See Also:
    :meth:`ry`

Args:
    theta: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcry", &qc::QuantumComputation::mcry, "theta"_a, "controls"_a,
         "target"_a,
         nb::sig("def mcry(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a multi-controlled :math:`R_y(\theta)` gate.

See Also:
    :meth:`ry`

Args:
    theta: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // RZ

  qc.def("rz", &qc::QuantumComputation::rz, "theta"_a, "q"_a,
         R"pb(Apply an :math:`R_z(\theta)` gate.

.. math::
    R_z(\theta) = e^{-i \theta Z / 2} = \begin{pmatrix} e^{-i \theta / 2} & 0 \\ 0 & e^{i \theta / 2} \end{pmatrix}

Args:
    theta: The rotation angle
    q: The target qubit)pb");
  qc.def("crz", &qc::QuantumComputation::crz, "theta"_a, "control"_a,
         "target"_a,
         nb::sig("def crz(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target: int) -> None"),
         R"pb(Apply a controlled :math:`R_z(\theta)` gate.

See Also:
    :meth:`rz`

Args:
    theta: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcrz", &qc::QuantumComputation::mcrz, "theta"_a, "controls"_a,
         "target"_a,
         nb::sig("def mcrz(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a multi-controlled :math:`R_z(\theta)` gate.

See Also:
    :meth:`rz`

Args:
    theta: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // P

  qc.def("p", &qc::QuantumComputation::p, "theta"_a, "q"_a,
         R"pb(Apply a phase gate.

.. math::
    P(\theta) = \begin{pmatrix} 1 & 0 \\ 0 & e^{i \theta} \end{pmatrix}

Args:
    theta: The rotation angle
    q: The target qubit)pb");
  qc.def("cp", &qc::QuantumComputation::cp, "theta"_a, "control"_a, "target"_a,
         nb::sig("def cp(self, theta: mqt.core.ir.symbolic.Expression | float, "
                 "control: "
                 "mqt.core.ir.operations.Control | int, target: int) -> None"),
         R"pb(Apply a controlled phase gate.

See Also:
    :meth:`p`

Args:
    theta: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcp", &qc::QuantumComputation::mcp, "theta"_a, "controls"_a,
         "target"_a,
         nb::sig("def mcp(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a multi-controlled phase gate.

See Also:
    :meth:`p`

Args:
    theta: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // U2

  qc.def("u2", &qc::QuantumComputation::u2, "phi"_a, "lambda_"_a, "q"_a,
         R"pb(Apply a :math:`U_2(\phi, \lambda)` gate.

.. math::
    U_2(\phi, \lambda) = \frac{1}{\sqrt{2}} \begin{pmatrix} 1 & -e^{i \lambda} \\ e^{i \phi} & e^{i (\phi + \lambda)} \end{pmatrix}

Args:
    phi: The rotation angle
    lambda_: The rotation angle
    q: The target qubit)pb");
  qc.def("cu2", &qc::QuantumComputation::cu2, "phi"_a, "lambda_"_a, "control"_a,
         "target"_a,
         nb::sig("def cu2(self, phi: mqt.core.ir.symbolic.Expression | float, "
                 "lambda_: "
                 "mqt.core.ir.symbolic.Expression | float, control: "
                 "mqt.core.ir.operations.Control | "
                 "int, target: int) -> None"),
         R"pb(Apply a controlled :math:`U_2(\phi, \lambda)` gate.

See Also:
    :meth:`u2`

Args:
    phi: The rotation angle
    lambda_: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def("mcu2", &qc::QuantumComputation::mcu2, "phi"_a, "lambda_"_a,
         "controls"_a, "target"_a,
         nb::sig("def mcu2(self, phi: mqt.core.ir.symbolic.Expression | float, "
                 "lambda_: "
                 "mqt.core.ir.symbolic.Expression | float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target: int) "
                 "-> None"),
         R"pb(Apply a multi-controlled :math:`U_2(\phi, \lambda)` gate.

See Also:
    :meth:`u2`

Args:
    phi: The rotation angle
    lambda_: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // R

  qc.def("r", &qc::QuantumComputation::r, "theta"_a, "phi"_a, "q"_a,
         R"pb(Apply an :math:`R(\theta, \phi)` gate.

.. math::
    R(\theta, \phi) = e^{-i \frac{\theta}{2} (\cos(\phi) X + \sin(\phi) Y)}
                    = \begin{pmatrix} \cos(\theta / 2) & -i e^{-i \phi} \sin(\theta / 2) \\ -i e^{i \phi} \sin(\theta / 2) & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    phi: The rotation angle
    q: The target qubit)pb");
  qc.def(
      "cr", &qc::QuantumComputation::cr, "theta"_a, "phi"_a, "control"_a,
      "target"_a,
      nb::sig(
          "def cr(self, theta: mqt.core.ir.symbolic.Expression | float, phi: "
          "mqt.core.ir.symbolic.Expression "
          "| float, control: mqt.core.ir.operations.Control | int, target: "
          "int) -> None"),
      R"pb(Apply a controlled :math:`R(\theta, \phi)` gate.

See Also:
    :meth:`r`

Args:
    theta: The rotation angle
    phi: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def(
      "mcr", &qc::QuantumComputation::mcr, "theta"_a, "phi"_a, "controls"_a,
      "target"_a,
      nb::sig(
          "def mcr(self, theta: mqt.core.ir.symbolic.Expression | float, phi: "
          "mqt.core.ir.symbolic.Expression | float, controls: "
          "collections.abc.Set[mqt.core.ir.operations.Control | int], target: "
          "int) "
          "-> None"),
      R"pb(Apply a multi-controlled :math:`R(\theta, \phi)` gate.

See Also:
    :meth:`r`

Args:
    theta: The rotation angle
    phi: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // U

  qc.def("u", &qc::QuantumComputation::u, "theta"_a, "phi"_a, "lambda_"_a,
         "q"_a,
         R"pb(Apply a :math:`U(\theta, \phi, \lambda)` gate.

.. math::
    U(\theta, \phi, \lambda) = \begin{pmatrix} \cos(\theta / 2) & -e^{i \lambda} \sin(\theta / 2) \\ e^{i \phi} \sin(\theta / 2) & e^{i (\phi + \lambda)}\cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    phi: The rotation angle
    lambda_: The rotation angle
    q: The target qubit)pb");
  qc.def(
      "cu", &qc::QuantumComputation::cu, "theta"_a, "phi"_a, "lambda_"_a,
      "control"_a, "target"_a,
      nb::sig(
          "def cu(self, theta: mqt.core.ir.symbolic.Expression | float, phi: "
          "mqt.core.ir.symbolic.Expression | float, lambda_: "
          "mqt.core.ir.symbolic.Expression | "
          "float, control: mqt.core.ir.operations.Control | int, target: int) "
          "-> None"),
      R"pb(Apply a controlled :math:`U(\theta, \phi, \lambda)` gate.

See Also:
    :meth:`u`

Args:
    theta: The rotation angle
    phi: The rotation angle
    lambda_: The rotation angle
    control: The control qubit
    target: The target qubit)pb");
  qc.def(
      "mcu", &qc::QuantumComputation::mcu, "theta"_a, "phi"_a, "lambda_"_a,
      "controls"_a, "target"_a,
      nb::sig(
          "def mcu(self, theta: mqt.core.ir.symbolic.Expression | float, phi: "
          "mqt.core.ir.symbolic.Expression | float, lambda_: "
          "mqt.core.ir.symbolic.Expression | "
          "float, controls: collections.abc.Set[mqt.core.ir.operations.Control "
          "| "
          "int], target: int) -> None"),
      R"pb(Apply a multi-controlled :math:`U(\theta, \phi, \lambda)` gate.

See Also:
    :meth:`u`

Args:
    theta: The rotation angle
    phi: The rotation angle
    lambda_: The rotation angle
    controls: The control qubits
    target: The target qubit)pb");

  // SWAP

  qc.def("swap", &qc::QuantumComputation::swap, "target1"_a, "target2"_a,
         R"pb(Apply a SWAP gate.

.. math::
    \text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cswap", &qc::QuantumComputation::cswap, "control"_a, "target1"_a,
         "target2"_a,
         nb::sig("def cswap(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a controlled SWAP gate.

See Also:
    :meth:`swap`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcswap", &qc::QuantumComputation::mcswap, "controls"_a, "target1"_a,
         "target2"_a,
         nb::sig("def mcswap(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled SWAP gate.

See Also:
    :meth:`swap`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // DCX

  qc.def("dcx", &qc::QuantumComputation::dcx, "target1"_a, "target2"_a,
         R"pb(Apply a DCX (i.e., double CNOT) gate.

.. math::
    DCX = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cdcx", &qc::QuantumComputation::cdcx, "control"_a, "target1"_a,
         "target2"_a,
         nb::sig("def cdcx(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a controlled DCX gate.

See Also:
    :meth:`dcx`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcdcx", &qc::QuantumComputation::mcdcx, "controls"_a, "target1"_a,
         "target2"_a,
         nb::sig("def mcdcx(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled DCX gate.

See Also:
    :meth:`dcx`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // ECR

  qc.def("ecr", &qc::QuantumComputation::ecr, "target1"_a, "target2"_a,
         R"pb(Apply a ECR (echoed cross-resonance) gate.

.. math::
    ECR = \frac{1}{\sqrt{2}} \begin{pmatrix} 0 & 0 & 1 & i \\ 0 & 0 & i & 1 \\ 1 & -i & 0 & 0 \\ -i & 1 & 0 & 0 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cecr", &qc::QuantumComputation::cecr, "control"_a, "target1"_a,
         "target2"_a,
         nb::sig("def cecr(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a controlled ECR gate.

See Also:
    :meth:`ecr`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcecr", &qc::QuantumComputation::mcecr, "controls"_a, "target1"_a,
         "target2"_a,
         nb::sig("def mcecr(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | "
                 "int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled ECR gate.

See Also:
    :meth:`ecr`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // iSWAP

  qc.def("iswap", &qc::QuantumComputation::iswap, "target1"_a, "target2"_a,
         R"pb(Apply a :math:`i\text{SWAP}` gate.

.. math::
    i\text{SWAP} = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & i & 0 \\ 0 & i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("ciswap", &qc::QuantumComputation::ciswap, "control"_a, "target1"_a,
         "target2"_a,
         nb::sig("def ciswap(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a controlled :math:`i\text{SWAP}` gate.

See Also:
    :meth:`iswap`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mciswap", &qc::QuantumComputation::mciswap, "controls"_a, "target1"_a,
         "target2"_a,
         nb::sig("def mciswap(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`i\text{SWAP}` gate.

See Also:
    :meth:`iswap`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // iSWAPdg

  qc.def("iswapdg", &qc::QuantumComputation::iswapdg, "target1"_a, "target2"_a,
         R"pb(Apply a :math:`i\text{SWAP}^\dagger` gate.

.. math::
    i\text{SWAP}^\dagger = \begin{pmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & -i & 0 \\ 0 & -i & 0 & 0 \\ 0 & 0 & 0 & 1 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("ciswapdg", &qc::QuantumComputation::ciswapdg, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def ciswapdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: "
                 "int, target2: int) -> None"),
         R"pb(Apply a controlled :math:`i\text{SWAP}^\dagger` gate.

See Also:
    :meth:`iswapdg`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mciswapdg", &qc::QuantumComputation::mciswapdg, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mciswapdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`i\text{SWAP}^\dagger` gate.

See Also:
    :meth:`iswapdg`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // Peres

  qc.def("peres", &qc::QuantumComputation::peres, "target1"_a, "target2"_a,
         R"pb(Apply a Peres gate.

.. math::
    \text{Peres} = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}

Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cperes", &qc::QuantumComputation::cperes, "control"_a, "target1"_a,
         "target2"_a,
         nb::sig("def cperes(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a controlled Peres gate.

See Also:
    :meth:`peres`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcperes", &qc::QuantumComputation::mcperes, "controls"_a, "target1"_a,
         "target2"_a,
         nb::sig("def mcperes(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled Peres gate.

See Also:
    :meth:`peres`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // Peresdg

  qc.def("peresdg", &qc::QuantumComputation::peresdg, "target1"_a, "target2"_a,
         R"pb(Apply a :math:`\text{Peres}^\dagger` gate.

.. math::
    \text{Peres}^\dagger = \begin{pmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{pmatrix}
Args:
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cperesdg", &qc::QuantumComputation::cperesdg, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def cperesdg(self, control: mqt.core.ir.operations.Control | "
                 "int, target1: "
                 "int, target2: int) -> None"),
         R"pb(Apply a controlled :math:`\text{Peres}^\dagger` gate.

See Also:
    :meth:`peresdg`

Args:
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcperesdg", &qc::QuantumComputation::mcperesdg, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mcperesdg(self, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control "
                 "| int], target1: int, target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`\text{Peres}^\dagger` gate.

See Also:
    :meth:`peresdg`

Args:
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // RXX

  qc.def("rxx", &qc::QuantumComputation::rxx, "theta"_a, "target1"_a,
         "target2"_a,
         R"pb(Apply an :math:`R_{xx}(\theta)` gate.

.. math::
    R_{xx}(\theta) = e^{-i \theta XX / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) X \otimes X
                   = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & -i \sin(\theta / 2) \\
                     0 & \cos(\theta / 2) & -i \sin(\theta / 2) & 0 \\
                     0 & -i \sin(\theta / 2) & \cos(\theta / 2) & 0 \\
                     -i \sin(\theta / 2) & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("crxx", &qc::QuantumComputation::crxx, "theta"_a, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def crxx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target1: int, target2: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`R_{xx}(\theta)` gate.

See Also:
    :meth:`rxx`

Args:
    theta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcrxx", &qc::QuantumComputation::mcrxx, "theta"_a, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mcrxx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{xx}(\theta)` gate.

See Also:
    :meth:`rxx`

Args:
    theta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // RYY

  qc.def("ryy", &qc::QuantumComputation::ryy, "theta"_a, "target1"_a,
         "target2"_a,
         R"pb(Apply an :math:`R_{yy}(\theta)` gate.

.. math::
    R_{yy}(\theta) = e^{-i \theta YY / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) Y \otimes Y
                   = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & i \sin(\theta / 2) \\
                     0 & \cos(\theta / 2) & -i \sin(\theta / 2) & 0 \\
                     0 & -i \sin(\theta / 2) & \cos(\theta / 2) & 0 \\
                     i \sin(\theta / 2) & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cryy", &qc::QuantumComputation::cryy, "theta"_a, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def cryy(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target1: int, target2: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`R_{yy}(\theta)` gate.

See Also:
    :meth:`ryy`

Args:
    theta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcryy", &qc::QuantumComputation::mcryy, "theta"_a, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mcryy(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{yy}(\theta)` gate.

See Also:
    :meth:`ryy`

Args:
    theta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // RZX

  qc.def("rzx", &qc::QuantumComputation::rzx, "theta"_a, "target1"_a,
         "target2"_a,
         R"pb(Apply an :math:`R_{zx}(\theta)` gate.

.. math::
    R_{zx}(\theta) = e^{-i \theta ZX / 2} = \cos(\theta / 2) I \otimes I - i \sin(\theta / 2) Z \otimes X
                   = \begin{pmatrix} \cos(\theta/2) & -i \sin(\theta/2) & 0 & 0 \\
                     -i \sin(\theta/2) & \cos(\theta/2) & 0 & 0 \\
                     0 & 0 & \cos(\theta/2) & i \sin(\theta/2) \\
                     0 & 0 & i \sin(\theta/2) & \cos(\theta/2) \end{pmatrix}

Args:
    theta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("crzx", &qc::QuantumComputation::crzx, "theta"_a, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def crzx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target1: int, target2: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`R_{zx}(\theta)` gate.

See Also:
    :meth:`rzx`

Args:
    theta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcrzx", &qc::QuantumComputation::mcrzx, "theta"_a, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mcrzx(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{zx}(\theta)` gate.

See Also:
    :meth:`rzx`

Args:
    theta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // RZZ

  qc.def("rzz", &qc::QuantumComputation::rzz, "theta"_a, "target1"_a,
         "target2"_a,
         R"pb(Apply an :math:`R_{zz}(\theta)` gate.

.. math::
    R_{zz}(\theta) = e^{-i \theta ZZ / 2}
                   = \begin{pmatrix} e^{-i \theta / 2} & 0 & 0 & 0 \\
                     0 & e^{i \theta / 2} & 0 & 0 \\
                     0 & 0 & e^{i \theta / 2} & 0 \\
                     0 & 0 & 0 & e^{-i \theta / 2} \end{pmatrix}

Args:
    theta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("crzz", &qc::QuantumComputation::crzz, "theta"_a, "control"_a,
         "target1"_a, "target2"_a,
         nb::sig("def crzz(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, control: "
                 "mqt.core.ir.operations.Control | int, target1: int, target2: "
                 "int) -> None"),
         R"pb(Apply a controlled :math:`R_{zz}(\theta)` gate.

See Also:
    :meth:`rzz`

Args:
    theta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcrzz", &qc::QuantumComputation::mcrzz, "theta"_a, "controls"_a,
         "target1"_a, "target2"_a,
         nb::sig("def mcrzz(self, theta: mqt.core.ir.symbolic.Expression | "
                 "float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{zz}(\theta)` gate.

See Also:
    :meth:`rzz`

Args:
    theta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // XXMinusYY

  qc.def("xx_minus_yy", &qc::QuantumComputation::xx_minus_yy, "theta"_a,
         "beta"_a, "target1"_a, "target2"_a,
         R"pb(Apply an :math:`R_{XX - YY}(\theta, \beta)` gate.

.. math::
    R_{XX - YY}(\theta, \beta) = R_{z_2}(\beta) \cdot e^{-i \frac{\theta}{2} \frac{XX - YY}{2}} \cdot R_{z_2}(-\beta)
                               = \begin{pmatrix} \cos(\theta / 2) & 0 & 0 & -i \sin(\theta / 2) e^{-i \beta} \\
                                 0 & 1 & 0 & 0 \\
                                 0 & 0 & 1 & 0 \\
                                 -i \sin(\theta / 2) e^{i \beta} & 0 & 0 & \cos(\theta / 2) \end{pmatrix}

Args:
    theta: The rotation angle
    beta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cxx_minus_yy", &qc::QuantumComputation::cxx_minus_yy, "theta"_a,
         "beta"_a, "control"_a, "target1"_a, "target2"_a,
         nb::sig("def cxx_minus_yy(self, theta: "
                 "mqt.core.ir.symbolic.Expression | float, beta: "
                 "mqt.core.ir.symbolic.Expression | float, control: "
                 "mqt.core.ir.operations.Control | "
                 "int, target1: int, target2: int) -> None"),
         R"pb(Apply a controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

See Also:
    :meth:`xx_minus_yy`

Args:
    theta: The rotation angle
    beta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcxx_minus_yy", &qc::QuantumComputation::mcxx_minus_yy, "theta"_a,
         "beta"_a, "controls"_a, "target1"_a, "target2"_a,
         nb::sig("def mcxx_minus_yy(self, theta: "
                 "mqt.core.ir.symbolic.Expression | float, beta: "
                 "mqt.core.ir.symbolic.Expression | float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{XX - YY}(\theta, \beta)` gate.

See Also:
    :meth:`xx_minus_yy`

Args:
    theta: The rotation angle
    beta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  // XXPlusYY

  qc.def("xx_plus_yy", &qc::QuantumComputation::xx_plus_yy, "theta"_a, "beta"_a,
         "target1"_a, "target2"_a,
         R"pb(Apply an :math:`R_{XX + YY}(\theta, \beta)` gate.

.. math::
    R_{XX + YY}(\theta, \beta) = R_{z_1}(\beta) \cdot e^{-i \frac{\theta}{2} \frac{XX + YY}{2}} \cdot R_{z_1}(-\beta)
                               = \begin{pmatrix} 1 & 0 & 0 & 0 \\
                                 0 & \cos(\theta / 2) & -i \sin(\theta / 2) e^{-i \beta} & 0 \\
                                 0 & -i \sin(\theta / 2) e^{i \beta} & \cos(\theta / 2) & 0 \\
                                 0 & 0 & 0 & 1 \end{pmatrix}

Args:
    theta: The rotation angle
    beta: The rotation angle
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("cxx_plus_yy", &qc::QuantumComputation::cxx_plus_yy, "theta"_a,
         "beta"_a, "control"_a, "target1"_a, "target2"_a,
         nb::sig("def cxx_plus_yy(self, theta: mqt.core.ir.symbolic.Expression "
                 "| float, beta: "
                 "mqt.core.ir.symbolic.Expression | float, control: "
                 "mqt.core.ir.operations.Control | "
                 "int, target1: int, target2: int) -> None"),
         R"pb(Apply a controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

See Also:
    :meth:`xx_plus_yy`

Args:
    theta: The rotation angle
    beta: The rotation angle
    control: The control qubit
    target1: The first target qubit
    target2: The second target qubit)pb");
  qc.def("mcxx_plus_yy", &qc::QuantumComputation::mcxx_plus_yy, "theta"_a,
         "beta"_a, "controls"_a, "target1"_a, "target2"_a,
         nb::sig("def mcxx_plus_yy(self, theta: "
                 "mqt.core.ir.symbolic.Expression | float, beta: "
                 "mqt.core.ir.symbolic.Expression | float, controls: "
                 "collections.abc.Set[mqt.core.ir.operations.Control | int], "
                 "target1: int, "
                 "target2: int) -> None"),
         R"pb(Apply a multi-controlled :math:`R_{XX + YY}(\theta, \beta)` gate.

See Also:
    :meth:`xx_plus_yy`

Args:
    theta: The rotation angle
    beta: The rotation angle
    controls: The control qubits
    target1: The first target qubit
    target2: The second target qubit)pb");

  qc.def("gphase", &qc::QuantumComputation::gphase, "phase"_a,
         R"pb(Apply a global phase gate.

.. math::
    GPhase(\theta) = (e^{i \theta})

Args:
    phase: The rotation angle)pb");

  qc.def("measure",
         nb::overload_cast<qc::Qubit, std::size_t>(
             &qc::QuantumComputation::measure),
         "qubit"_a, "cbit"_a,
         R"pb(Measure a qubit and store the result in a classical bit.

Args:
    qubit: The qubit to measure
    cbit: The classical bit to store the result)pb");

  qc.def("measure",
         nb::overload_cast<const std::vector<qc::Qubit>&,
                           const std::vector<qc::Bit>&>(
             &qc::QuantumComputation::measure),
         "qubits"_a, "cbits"_a,
         R"pb(Measure multiple qubits and store the results in classical bits.

This method is equivalent to calling :meth:`measure` multiple times.

Args:
    qubits: The qubits to measure
    cbits: The classical bits to store the results)pb");
  qc.def("measure_all", &qc::QuantumComputation::measureAll, nb::kw_only(),
         "add_bits"_a = true,
         R"pb(Measure all qubits and store the results in classical bits.

Details:
    If `add_bits` is `True`, a new classical register (named "`meas`") with the same size as the number of qubits will be added to the circuit and the results will be stored in it.
    If `add_bits` is `False`, the classical register must already exist and have a sufficient number of bits to store the results.

Args:
    add_bits: Whether to explicitly add a classical register)pb");

  qc.def("reset", nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::reset),
         "q"_a, R"pb(Add a reset operation to the circuit.

Args:
    q: The qubit to reset)pb");

  qc.def("reset",
         nb::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::reset),
         "qubits"_a, R"pb(Add a reset operation to the circuit.

Args:
    qubits: The qubits to reset)pb");

  qc.def("barrier", nb::overload_cast<>(&qc::QuantumComputation::barrier),
         "Add a barrier to the circuit.");

  qc.def("barrier",
         nb::overload_cast<qc::Qubit>(&qc::QuantumComputation::barrier), "q"_a,
         R"pb(Add a barrier to the circuit.

Args:
    q: The qubit to add the barrier to)pb");

  qc.def("barrier",
         nb::overload_cast<const std::vector<qc::Qubit>&>(
             &qc::QuantumComputation::barrier),
         "qubits"_a, R"pb(Add a barrier to the circuit.

Args:
    qubits: The qubits to add the barrier to)pb");

  qc.def(
      "if_else",
      [](qc::QuantumComputation& self, qc::Operation* thenOp,
         qc::Operation* elseOp, const qc::ClassicalRegister& controlRegister,
         const std::uint64_t expectedValue = 1U,
         const qc::ComparisonKind kind = qc::ComparisonKind::Eq) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        self.ifElse(std::move(thenPtr), std::move(elsePtr), controlRegister,
                    expectedValue, kind);
      },
      "then_operation"_a, "else_operation"_a, "control_register"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq,
      R"pb(Add an if-else operation to the circuit.

Args:
    then_operation: The operation to apply if the condition is met
    else_operation: The operation to apply if the condition is not met
    control_register: The classical register to check against
    expected_value: The expected value of the control register
    comparison_kind: The kind of comparison to perform)pb");

  qc.def(
      "if_else",
      [](qc::QuantumComputation& self, qc::Operation* thenOp,
         qc::Operation* elseOp, const qc::Bit controlBit,
         const std::uint64_t expectedValue = 1U,
         const qc::ComparisonKind kind = qc::ComparisonKind::Eq) {
        std::unique_ptr<qc::Operation> thenPtr =
            thenOp ? thenOp->clone() : nullptr;
        std::unique_ptr<qc::Operation> elsePtr =
            elseOp ? elseOp->clone() : nullptr;
        self.ifElse(std::move(thenPtr), std::move(elsePtr), controlBit,
                    expectedValue, kind);
      },
      "then_operation"_a, "else_operation"_a, "control_bit"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq,
      R"pb(Add an if-else operation to the circuit.

Args:
    then_operation: The operation to apply if the condition is met
    else_operation: The operation to apply if the condition is not met
    control_bit: The index of the classical bit to check against
    expected_value: The expected value of the control bit
    comparison_kind: The kind of comparison to perform)pb");

  qc.def(
      "if_",
      nb::overload_cast<const qc::OpType, const qc::Qubit,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::if_),
      "op_type"_a, "target"_a, "control_register"_a, "expected_value"_a = 1U,
      "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a.sig("...") = std::vector<qc::fp>{},
      R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    control_register: The classical register to check against
    expected_value: The expected value of the control register
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation)pb");

  qc.def(
      "if_",
      nb::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::if_),
      "op_type"_a, "target"_a, "control"_a, "control_register"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{},
      nb::sig(
          "def if_(self, op_type: mqt.core.ir.operations.OpType, target: "
          "int, control: mqt.core.ir.operations.Control | int, "
          "control_register: mqt.core.ir.registers.ClassicalRegister, "
          "expected_value: int = 1, comparison_kind: "
          "mqt.core.ir.operations.ComparisonKind = ..., params: "
          "collections.abc.Sequence[mqt.core.ir.symbolic.Expression | float] "
          "= ...) -> None"),
      R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    control: The control qubit
    control_register: The classical register to check against
    expected_value: The expected value of the control register
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation.)pb");

  qc.def(
      "if_",
      nb::overload_cast<const qc::OpType, const qc::Qubit, const qc::Controls&,
                        const qc::ClassicalRegister&, const std::uint64_t,
                        const qc::ComparisonKind, const std::vector<qc::fp>&>(
          &qc::QuantumComputation::if_),
      "op_type"_a, "target"_a, "controls"_a, "control_register"_a,
      "expected_value"_a = 1U, "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{},
      nb::sig(
          "def if_(self, op_type: mqt.core.ir.operations.OpType, target: "
          "int, controls: collections.abc.Set[mqt.core.ir.operations.Control | "
          "int], "
          "control_register: mqt.core.ir.registers.ClassicalRegister, "
          "expected_value: int = 1, comparison_kind: "
          "mqt.core.ir.operations.ComparisonKind = ..., params: "
          "collections.abc.Sequence[mqt.core.ir.symbolic.Expression | float] "
          "= ...) -> None"),
      R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    controls: The control qubits
    control_register: The classical register to check against
    expected_value: The expected value of the control register
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation.)pb");

  qc.def("if_",
         nb::overload_cast<const qc::OpType, const qc::Qubit, const qc::Bit,
                           const bool, const qc::ComparisonKind,
                           const std::vector<qc::fp>&>(
             &qc::QuantumComputation::if_),
         "op_type"_a, "target"_a, "control_bit"_a, "expected_value"_a = true,
         "comparison_kind"_a = qc::ComparisonKind::Eq,
         "params"_a.sig("...") = std::vector<qc::fp>{},
         R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    control_bit: The index of the classical bit to check against
    expected_value: The expected value of the control bit
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation.)pb");

  qc.def(
      "if_",
      nb::overload_cast<const qc::OpType, const qc::Qubit, const qc::Control,
                        const qc::Bit, const bool, const qc::ComparisonKind,
                        const std::vector<qc::fp>&>(
          &qc::QuantumComputation::if_),
      "op_type"_a, "target"_a, "control"_a, "control_bit"_a,
      "expected_value"_a = true, "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{},
      nb::sig("def if_(self, op_type: mqt.core.ir.operations.OpType, target: "
              "int, control: mqt.core.ir.operations.Control | int, "
              "control_bit: int, expected_value: bool = True, "
              "comparison_kind: "
              "mqt.core.ir.operations.ComparisonKind = ..., params: "
              "collections.abc.Sequence[mqt.core.ir.symbolic.Expression | "
              "float] = ...) -> None"),
      R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    control: The control qubit
    control_bit: The index of the classical bit to check against
    expected_value: The expected value of the control bit
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation.)pb");

  qc.def(
      "if_",
      nb::overload_cast<const qc::OpType, const qc::Qubit, const qc::Controls&,
                        const qc::Bit, const bool, const qc::ComparisonKind,
                        const std::vector<qc::fp>&>(
          &qc::QuantumComputation::if_),
      "op_type"_a, "target"_a, "controls"_a, "control_bit"_a,
      "expected_value"_a = true, "comparison_kind"_a = qc::ComparisonKind::Eq,
      "params"_a = std::vector<qc::fp>{},
      nb::sig(
          "def if_(self, op_type: mqt.core.ir.operations.OpType, target: "
          "int, controls: collections.abc.Set[mqt.core.ir.operations.Control "
          "| int], control_bit: int, expected_value: bool = True, "
          "comparison_kind: "
          "mqt.core.ir.operations.ComparisonKind = ..., params: "
          "collections.abc.Sequence[mqt.core.ir.symbolic.Expression | "
          "float] = ...) -> None"),
      R"pb(Add an if operation to the circuit.

Args:
    op_type: The operation to apply
    target: The target qubit
    controls: The control qubits
    control_bit: The index of the classical bit to check against
    expected_value: The expected value of the control bit
    comparison_kind: The kind of comparison to perform
    params: The parameters of the operation.)pb");
}

} // namespace mqt
