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
#include "qdmi/Parameters.hpp"
#include "qdmi/QDMI.hpp"

#include "Array.hpp"
#include "CircuitRecorder.h"
#include "QDMIDevice.h"
#include "QDMIReadout.h"

#include "nanobind/nanobind.h"
#include "nanobind/ndarray.h"
#include "nanobind/stl/optional.h"
#include "nanobind/stl/pair.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/string.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"

#include "llvm/ADT/SmallVector.h"

#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
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

using Samples =
    nb::ndarray<nb::numpy, int8_t, nb::ndim<2>, nb::c_contig, nb::device::cpu>;

// Contexts outlive Python calls, so byte parameters need owned storage.
static std::optional<catalyst::JobParameter>
ownJobParameter(const std::optional<PythonCustomJobParameter>& parameter) {
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
  const auto indices = [&wireMap](const nb::object& wireLabels) {
    std::vector<QubitIdType> result;
    for (const auto wire : nb::borrow<nb::iterable>(wireLabels)) {
      result.push_back(nb::cast<QubitIdType>(wireMap[wire]));
    }
    return result;
  };
  const auto modifiers = nb::module_::import_("pennylane.ops.op_math");
  const auto adjoint = modifiers.attr("Adjoint");
  const auto controlled = modifiers.attr("Controlled");
  mqt::pennylane::CircuitRecorder recorder(width);
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
      llvm::SmallVector<int8_t> bits;
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
      recorder.setBasisState(bits, operands);
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
      recorder.matrixOperation({matrix.data(), matrix.data() + matrix.size()},
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
    recorder.namedOperation(name, parameters, operands, inverse, controls,
                            controlValues);
  }
  const nb::gil_scoped_release release;
  return recorder.takeProgram();
}

void registerBindings(nb::module_& m) {
  auto pennylane = m.def_submodule(
      "_pennylane", "Runtime support for the PennyLane QDMI device.");
  nb::class_<catalyst::Context>(
      pennylane, "Context",
      "Retain a QDMI device, target snapshot, and job parameters.")
      .def(nb::new_([](qdmi::Device device,
                       const std::optional<PythonCustomJobParameter>& custom1,
                       const std::optional<PythonCustomJobParameter>& custom2,
                       const std::optional<PythonCustomJobParameter>& custom3,
                       const std::optional<PythonCustomJobParameter>& custom4,
                       const std::optional<PythonCustomJobParameter>& custom5,
                       std::optional<QDMI_Program_Format> programFormat) {
             catalyst::JobParameters parameters{
                 ownJobParameter(custom1), ownJobParameter(custom2),
                 ownJobParameter(custom3), ownJobParameter(custom4),
                 ownJobParameter(custom5),
             };
             const nb::gil_scoped_release release;
             return catalyst::Context::create(
                 std::move(device), std::move(parameters), programFormat);
           }),
           "device"_a, nb::kw_only(), "custom1"_a = nb::none(),
           "custom2"_a = nb::none(), "custom3"_a = nb::none(),
           "custom4"_a = nb::none(), "custom5"_a = nb::none(),
           "program_format"_a = nb::none(),
           R"doc(Create a context for native compilation and submission.

Args:
    device: Open QDMI device retained for the context lifetime.
    custom1: First provider-specific job parameter.
    custom2: Second provider-specific job parameter.
    custom3: Third provider-specific job parameter.
    custom4: Fourth provider-specific job parameter.
    custom5: Fifth provider-specific job parameter.
    program_format: Preferred program format, or automatic selection.

Byte parameters are copied so later submissions do not borrow Python storage.
)doc")
      .def_prop_ro(
          "id", &catalyst::Context::id,
          "Identifier used by Catalyst to retrieve this retained context.")
      .def_prop_ro(
          "environment", &catalyst::Context::environment,
          "Target environment used to compile programs for this context.")
      .def("submit", &catalyst::Context::submit, "program"_a, "num_shots"_a,
           nb::call_guard<nb::gil_scoped_release>(),
           R"doc(Submit one compiled program with the retained job parameters.

Args:
    program: Program compiled for this context's target environment.
    num_shots: Positive number of execution shots.

Returns:
    The submitted QDMI job, without waiting for completion.
)doc")
      .def(
          "try_submit",
          [](const catalyst::Context& context,
             const std::vector<const mlir::CompiledProgram*>& programs,
             int64_t shots) {
            const nb::gil_scoped_release release;
            return context.trySubmit(programs, shots);
          },
          "programs"_a, "num_shots"_a,
          R"doc(Submit compatible programs as one multi-program job when supported.

Args:
    programs: Programs compiled for this context's target environment.
    num_shots: Positive shot count applied to every program.

Returns:
    The submitted job, or None when individual submissions are required.
)doc");

  pennylane.def(
      "aggregate",
      [](const qdmi::Job& job, size_t width, size_t shots,
         const std::vector<std::pair<std::vector<size_t>, bool>>& measurements,
         size_t programIndex, const std::optional<mlir::ResultLayout>& layout) {
        using Marginal =
            std::variant<std::vector<double>, std::map<std::string, size_t>>;
        auto values = [&] {
          const nb::gil_scoped_release release;
          const auto result =
              runtime::readout(job, width, shots, programIndex, layout, false);
          llvm::SmallVector<Marginal> marginals;
          marginals.reserve(measurements.size());
          for (const auto& [wires, counts] : measurements) {
            if (counts) {
              marginals.emplace_back(runtime::marginalCounts(result, wires));
            } else {
              marginals.emplace_back(
                  runtime::marginalProbabilities(result, wires, shots));
            }
          }
          return marginals;
        }();
        nb::typed<nb::list, nb::object> result;
        for (auto& value : values) {
          if (auto* probabilities = std::get_if<std::vector<double>>(&value)) {
            result.append(ownedArray(std::move(*probabilities)));
          } else {
            nb::typed<nb::dict, nb::str, size_t> counts;
            for (const auto& [bits, count] :
                 std::get<std::map<std::string, size_t>>(value)) {
              counts[nb::str(bits.data(), bits.size())] = nb::int_(count);
            }
            result.append(counts);
          }
        }
        return result;
      },
      "job"_a, "num_wires"_a, "shots"_a, "measurements"_a,
      "program_index"_a = 0, "result_layout"_a = nb::none(),
      R"doc(Compute measurement marginals from validated provider readout.

Args:
    job: Completed provider job whose results are read.
    num_wires: Logical output width after layout projection.
    shots: Positive expected shot count.
    measurements: Pairs of logical wire indices and a counts flag. True
        selects a binary-key count dictionary; false selects a one-dimensional
        NumPy probability array. The first selected wire is most significant.
    program_index: Program within a multi-program job.
    result_layout: Projection from provider output to logical wires.

Returns:
    One marginal per measurement, in the requested order. Counts are preferred;
    unavailable or empty histograms fall back to shots. Malformed results fail
    validation without fallback. Observable eigenvalues are applied by Python.
)doc");

  pennylane.def(
      "samples",
      [](const qdmi::Job& job, size_t numWires, size_t shots,
         size_t programIndex, const std::optional<mlir::ResultLayout>& layout) {
        std::unique_ptr<std::vector<int8_t>> data;
        {
          const nb::gil_scoped_release release;
          data = std::make_unique<std::vector<int8_t>>(runtime::decodeSamples(
              job, numWires, shots, programIndex, layout));
        }
        const nb::capsule owner(data.get(), [](void* pointer) noexcept {
          delete static_cast<std::vector<int8_t>*>(pointer);
        });
        auto* const samples = data.release();
        return Samples(samples->data(), {shots, numWires}, owner);
      },
      "job"_a, "num_wires"_a, "shots"_a, "program_index"_a = 0,
      "result_layout"_a = nb::none(),
      R"doc(Return a shot-by-wire NumPy array in logical wire order.

Args:
    job: Completed provider job whose results are read.
    num_wires: Number of logical output columns after layout projection.
    shots: Positive expected number of rows.
    program_index: Program within a multi-program job.
    result_layout: Projection from provider output to logical wires.

Returns:
    A contiguous int8 array that owns its storage independently of the job.
    Ordered SHOTS are preferred; unavailable or empty shots fall back to
    histogram expansion, which preserves frequencies but has no shot order.
    Malformed results fail validation without fallback.
)doc");
}
} // namespace mqt::bindings::pennylane

// Catalyst loads its device factory from the existing Python extension.
extern "C" NB_EXPORT Catalyst::Runtime::QuantumDevice*
MQTQDMIFactory(const char* kwargs) { // NOLINT(readability-identifier-naming)
  return mqt::catalyst::createDevice(kwargs);
}
