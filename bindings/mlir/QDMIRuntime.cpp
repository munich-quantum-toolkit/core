/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QDMIRuntime.h"

#include "Result.hpp"

#include "nanobind/stl/optional.h"
#include "nanobind/stl/shared_ptr.h"
#include "nanobind/stl/variant.h"
#include "nanobind/stl/vector.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace mqt::bindings {
namespace nb = nanobind;

// Contexts outlive Python calls, so byte parameters need owned storage.
std::optional<runtime::JobParameter>
ownJobParameter(const std::optional<PythonCustomJobParameter>& parameter) {
  if (!parameter) {
    return std::nullopt;
  }
  return std::visit(
      [](const auto& value) -> runtime::JobParameter {
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

void registerQDMIRuntime(nb::module_& module) {
  using namespace nb::literals;
  auto runtimeModule =
      module.def_submodule("_qdmi", "Shared native QDMI execution support.");
  nb::class_<runtime::Context>(
      runtimeModule, "Context",
      "Retain a QDMI device, target snapshot, and default job parameters.")
      .def(nb::new_([](qdmi::Device device) {
             const nb::gil_scoped_release release;
             auto environment =
                 unwrap(mlir::targetEnvironmentFromDevice(device));
             return std::make_shared<runtime::Context>(std::move(device),
                                                       std::move(environment));
           }),
           "device"_a, "Create an execution context from an open device.")
      .def_prop_ro(
          "environment", &runtime::Context::environment,
          "Target environment used to compile programs for this context.")
      .def(
          "prepare",
          [](const runtime::Context& context, const nb::tuple& programs) {
            struct Group {
              llvm::SmallVector<const mlir::CompiledProgram*> programs;
              bool retained = true;
            };
            llvm::SmallMapVector<QDMI_Program_Format, Group, 4> groups;
            for (const auto program : programs) {
              const auto& compiled =
                  nb::cast<const mlir::CompiledProgram&>(program);
              auto& group = groups[compiled.programFormat()];
              group.programs.push_back(&compiled);
              const auto environment =
                  nb::getattr(program, "_environment", nb::none());
              group.retained &=
                  nb::isinstance<mlir::TargetEnvironment>(environment) &&
                  nb::cast<const mlir::TargetEnvironment*>(environment) ==
                      &context.environment();
            }
            nb::dict destinations;
            for (const auto& [format, group] : groups) {
              auto destination = [&] {
                const nb::gil_scoped_release release;
                return unwrap(context.prepare(group.programs, group.retained));
              }();
              destinations[nb::cast(format)] = nb::cast(std::move(destination));
            }
            return destinations;
          },
          "programs"_a,
          nb::sig("def prepare(self, programs: "
                  "tuple[mqt.core.mlir.CompiledProgram, ...]) -> "
                  "dict[mqt.core.qdmi.ProgramFormat, "
                  "mqt.core.mlir.TargetEnvironment]"),
          R"doc(Validate all format groups before submitting any program.

Programs compiled against this context reuse its target snapshot. Other groups
obtain a fresh destination snapshot. Returns destinations indexed by format.
)doc")
      .def(
          "submit",
          [](const runtime::Context& context,
             const mlir::CompiledProgram& program, int64_t shots) {
            const nb::gil_scoped_release release;
            return unwrap(context.submit(program, shots));
          },
          "program"_a, "num_shots"_a,
          R"doc(Submit one compiled program with the retained job parameters.

Args:
    program: Program compiled for this context's target environment.
    num_shots: Nonnegative number of execution shots.

Returns:
    The submitted QDMI job, without waiting for completion.
)doc")
      .def(
          "try_submit",
          [](const runtime::Context& context,
             const std::vector<const mlir::CompiledProgram*>& programs,
             int64_t shots,
             const std::optional<PythonCustomJobParameter>& custom1,
             const std::optional<PythonCustomJobParameter>& custom2,
             const std::optional<PythonCustomJobParameter>& custom3,
             const std::optional<PythonCustomJobParameter>& custom4,
             const std::optional<PythonCustomJobParameter>& custom5,
             const mlir::TargetEnvironment* destination) {
            const runtime::BorrowedJobParameters parameters{
                toCustomJobParameter(custom1), toCustomJobParameter(custom2),
                toCustomJobParameter(custom3), toCustomJobParameter(custom4),
                toCustomJobParameter(custom5),
            };
            const nb::gil_scoped_release release;
            return unwrap(context.trySubmit(programs, shots,
                                            custom1 || custom2 || custom3 ||
                                                    custom4 || custom5
                                                ? &parameters
                                                : nullptr,
                                            destination));
          },
          "programs"_a, "num_shots"_a, nb::kw_only(),
          "custom1"_a.noconvert() = nb::none(),
          "custom2"_a.noconvert() = nb::none(),
          "custom3"_a.noconvert() = nb::none(),
          "custom4"_a.noconvert() = nb::none(),
          "custom5"_a.noconvert() = nb::none(), "destination"_a = nb::none(),
          R"doc(Submit a compatible group, returning None if the provider declines it.

Args:
    programs: Programs with one payload format.
    num_shots: Nonnegative shot count applied to every program.
    custom1: First provider-specific job parameter.
    custom2: Second provider-specific job parameter.
    custom3: Third provider-specific job parameter.
    custom4: Fourth provider-specific job parameter.
    custom5: Fifth provider-specific job parameter.
    destination: Validated destination from prepare, or this context's snapshot.

A supplied custom parameter replaces the retained parameter set for this call.
The context itself is unchanged. With no supplied custom parameters, use the
retained defaults.
)doc");
}

} // namespace mqt::bindings
