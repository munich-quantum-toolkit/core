/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "PennyLane.h"

#include "mqt/Compiler/ResultLayout.h"
#include "qdmi/QDMI.hpp"

#include "QDMIDevice.h"

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/variant.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::bindings::pennylane {
namespace nb = nanobind;
using namespace nb::literals;

using PythonJobParameter =
    std::variant<std::string, bool, int, double, nb::bytes>;
using Samples =
    nb::ndarray<nb::numpy, int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

static std::optional<catalyst::JobParameter>
ownJobParameter(const std::optional<PythonJobParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> catalyst::JobParameter {
        if constexpr (std::is_same_v<std::decay_t<decltype(value)>,
                                     nb::bytes>) {
          const auto bytes =
              std::as_bytes(std::span(value.c_str(), value.size()));
          return std::vector<std::byte>(bytes.begin(), bytes.end());
        } else {
          return value;
        }
      },
      *parameter);
}

mlir::QCProgram importCircuit(const nb::object& tape, const nb::object& wires) {
  const auto labels = wires.is_none() ? tape.attr("wires") : wires;
  nb::dict wireMap;
  size_t width = 0;
  for (const auto wire : nb::borrow<nb::iterable>(labels)) {
    if (wireMap.contains(wire)) {
      throw nb::value_error("PennyLane wire labels must be unique");
    }
    wireMap[wire] = nb::int_(width++);
  }
  const auto indices = [&wireMap](const nb::object& operands) {
    std::vector<QubitIdType> result;
    for (const auto wire : nb::borrow<nb::iterable>(operands)) {
      result.push_back(nb::cast<QubitIdType>(wireMap[wire]));
    }
    return result;
  };
  const auto modifiers = nb::module_::import_("pennylane.ops.op_math");
  const auto adjoint = modifiers.attr("Adjoint");
  const auto controlled = modifiers.attr("Controlled");
  catalyst::QDMIDevice recorder(width);
  for (const auto handle : nb::borrow<nb::iterable>(tape.attr("operations"))) {
    auto operation = nb::borrow<nb::object>(handle);
    bool inverse = false;
    std::vector<QubitIdType> controls;
    std::vector<bool> controlValues;
    while (nb::isinstance(operation, adjoint) ||
           nb::isinstance(operation, controlled)) {
      if (nb::isinstance(operation, adjoint)) {
        inverse = !inverse;
      } else {
        const auto outer = indices(operation.attr("control_wires"));
        controls.insert(controls.end(), outer.begin(), outer.end());
        for (const auto value :
             nb::borrow<nb::iterable>(operation.attr("control_values"))) {
          const auto truth = PyObject_IsTrue(value.ptr());
          if (truth < 0) {
            throw nb::python_error();
          }
          controlValues.push_back(truth != 0);
        }
      }
      operation = operation.attr("base");
    }
    const auto name = nb::cast<std::string>(operation.attr("name"));
    auto operands = indices(operation.attr("wires"));
    if (name == "BasisState") {
      if (inverse || !controls.empty()) {
        throw nb::value_error("BasisState cannot be controlled or inverted");
      }
      std::vector<int8_t> bits;
      for (const auto bit :
           nb::borrow<nb::iterable>(operation.attr("parameters")[0])) {
        const auto value = PyFloat_AsDouble(bit.ptr());
        if (PyErr_Occurred() != nullptr) {
          throw nb::python_error();
        }
        if (value != 0 && value != 1) {
          throw nb::value_error("BasisState entries must be zero or one");
        }
        bits.push_back(static_cast<int8_t>(value));
      }
      DataView<int8_t, 1> view(bits);
      recorder.SetBasisState(view, operands);
      continue;
    }
    if (name == "QubitUnitary") {
      const auto matrix =
          nb::cast<nb::ndarray<const std::complex<double>, nb::ndim<2>,
                               nb::c_contig, nb::device::cpu>>(
              operation.attr("parameters")[0]);
      if (matrix.shape(0) != matrix.shape(1)) {
        throw nb::value_error("QubitUnitary matrix must be square");
      }
      recorder.MatrixOperation({matrix.data(), matrix.data() + matrix.size()},
                               operands, inverse, controls, controlValues);
      continue;
    }
    std::vector<double> parameters;
    for (const auto parameter :
         nb::borrow<nb::iterable>(operation.attr("parameters"))) {
      const auto value = PyFloat_AsDouble(parameter.ptr());
      if (PyErr_Occurred() != nullptr) {
        throw nb::python_error();
      }
      parameters.push_back(value);
    }
    recorder.NamedOperation(name, parameters, operands, inverse, controls,
                            controlValues);
  }
  const nb::gil_scoped_release release;
  return recorder.takeProgram();
}

void registerBindings(nb::module_& m) {
  auto pennylane = m.def_submodule(
      "_pennylane", "Runtime support for the PennyLane QDMI device.");
  nb::class_<catalyst::Context>(pennylane, "Context")
      .def(nb::new_([](qdmi::Device device,
                       const std::optional<PythonJobParameter>& custom1,
                       const std::optional<PythonJobParameter>& custom2,
                       const std::optional<PythonJobParameter>& custom3,
                       const std::optional<PythonJobParameter>& custom4,
                       const std::optional<PythonJobParameter>& custom5,
                       std::optional<QDMI_Program_Format> programFormat,
                       std::optional<mlir::TerminalReadout> terminalReadout,
                       bool physicalQubits) {
             catalyst::JobParameters parameters{
                 ownJobParameter(custom1), ownJobParameter(custom2),
                 ownJobParameter(custom3), ownJobParameter(custom4),
                 ownJobParameter(custom5),
             };
             const nb::gil_scoped_release release;
             return catalyst::Context::create(
                 std::move(device), std::move(parameters), programFormat,
                 terminalReadout, physicalQubits);
           }),
           "device"_a, nb::kw_only(), "custom1"_a = nb::none(),
           "custom2"_a = nb::none(), "custom3"_a = nb::none(),
           "custom4"_a = nb::none(), "custom5"_a = nb::none(),
           "program_format"_a = nb::none(), "terminal_readout"_a = nb::none(),
           "physical_qubits"_a = true)
      .def_prop_ro("id", &catalyst::Context::id);

  pennylane.def(
      "samples",
      [](const qdmi::Job& job, size_t numWires, size_t shots,
         size_t programIndex, const std::optional<mlir::ResultLayout>& layout) {
        std::unique_ptr<std::vector<int8_t>> data;
        {
          const nb::gil_scoped_release release;
          data = std::make_unique<std::vector<int8_t>>(catalyst::decodeSamples(
              job, numWires, shots, programIndex, layout));
        }
        const nb::capsule owner(data.get(), [](void* pointer) noexcept {
          delete static_cast<std::vector<int8_t>*>(pointer);
        });
        auto* const samples = data.release();
        return Samples(samples->data(), {shots, numWires}, owner);
      },
      "job"_a, "num_wires"_a, "shots"_a, "program_index"_a = 0,
      "result_layout"_a = nb::none());
}
} // namespace mqt::bindings::pennylane

// Catalyst loads its device factory from the existing Python extension.
extern "C" NB_EXPORT Catalyst::Runtime::QuantumDevice*
MQTQDMIFactory(const char* kwargs) { // NOLINT(readability-identifier-naming)
  return mqt::catalyst::createDevice(kwargs);
}
