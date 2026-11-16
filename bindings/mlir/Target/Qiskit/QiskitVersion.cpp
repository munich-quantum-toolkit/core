/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "QiskitVersion.h"

#include "Qiskit.h"
#include "Result.hpp"

// Keep the translation interface visible where the factory is instantiated.
#include "QiskitTranslation.h" // IWYU pragma: keep

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mqt::bindings::qiskit {
namespace nb = nanobind;

nb::object nativeRGate(const std::string_view name, nb::handle phase) {
  const auto alias = mlir::nativeRAlias(name);
  if (!alias) {
    return nb::none();
  }
  const auto circuit = nb::module_::import_("qiskit.circuit");
  nb::list parameters;
  parameters.append(phase);
  auto gate =
      circuit.attr("Gate")(nb::str(name.data(), name.size()), 1, parameters);
  auto definition = circuit.attr("QuantumCircuit")(1);
  definition.attr("r")(alias->angle, phase, 0);
  definition.attr("global_phase") = alias->globalPhase;
  gate.attr("definition") = definition;
  return gate;
}

namespace {
using Factory = std::unique_ptr<VersionedTranslation> (*)();

struct Selection {
  Factory factory = nullptr;
  uint32_t minimum = 0;
};
} // namespace

[[nodiscard]] static Selection selectVersion(uint32_t version) {
  Selection selected;
  if ((version >> 24U) == 2U) {
#define MQT_QISKIT_VERSION(minor, patch, release, range)                       \
  {                                                                            \
    constexpr auto min =                                                       \
        (2U << 24U) | (minor##U << 16U) | (patch##U << 8U) | 0xf0U;            \
    if (version >= min && min > selected.minimum) {                            \
      selected = {createQiskit2_##minor, min};                                 \
    }                                                                          \
  }
#include "SupportedVersions.inc"
#undef MQT_QISKIT_VERSION
  }
  return selected;
}

bool supportsVersion(llvm::StringRef version) {
  uint32_t major = 0;
  uint32_t minor = 0;
  uint32_t patch = 0;
  if (version.consumeInteger(10, major) || !version.consume_front(".") ||
      version.consumeInteger(10, minor)) {
    return false;
  }
  if (version.consume_front(".") && version.consumeInteger(10, patch)) {
    return false;
  }
  if (major > 255 || minor > 255 || patch > 255) {
    return false;
  }
  const bool prerelease =
      version.consume_front("a") || version.consume_front("b") ||
      version.consume_front("rc") || version.consume_front(".dev");
  if (prerelease || version.consume_front(".post")) {
    unsigned serial = 0;
    if (version.consumeInteger(10, serial)) {
      return false;
    }
  }
  if (version.consume_front("+")) {
    if (version.empty() || version.find_first_not_of(
                               "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTU"
                               "VWXYZ0123456789.-") != llvm::StringRef::npos) {
      return false;
    }
  } else if (!version.empty()) {
    return false;
  }
  const auto stage = prerelease ? 0U : 0xf0U;
  return selectVersion((major << 24U) | (minor << 16U) | (patch << 8U) | stage)
             .factory != nullptr;
}

std::unique_ptr<VersionedTranslation> selectTranslation() {
  const auto qiskit = nb::module_::import_("qiskit");
  const auto version = nb::cast<uint32_t>(
      nb::module_::import_("qiskit._accelerate.capi").attr("API_VERSION"));
  const auto [factory, selectedMinimum] = selectVersion(version);
  if (factory != nullptr) {
    static bool warned = false;
    if (!warned && (version >> 16U) > (selectedMinimum >> 16U)) {
      const auto warning =
          nb::module_::import_("mqt.core").attr("UntestedVersionWarning");
      const auto adapter =
          "2." + std::to_string((selectedMinimum >> 16U) & 0xffU);
      const auto message =
          "Qiskit " + nb::cast<std::string>(qiskit.attr("__version__")) +
          " is newer than the newest tested version " + adapter +
          "; using the " + adapter +
          " adapter. Report problems at "
          "https://github.com/munich-quantum-toolkit/core/issues.";
      if (PyErr_WarnEx(warning.ptr(), message.c_str(), 1) < 0) {
        throw nb::python_error();
      }
      warned = true;
    }
    return factory();
  }
  throw std::runtime_error(
      "Qiskit circuit translation does not support installed version '" +
      nb::cast<std::string>(qiskit.attr("__version__")) +
      "'; supported versions: >=2.5.0,<3.0.0");
}

mlir::CompilerTarget importTarget(nb::handle target, nb::handle operationNames,
                                  const std::optional<std::string>& name) {
  return selectTranslation()->importTarget(target, operationNames, name, false);
}

mlir::CompilerTarget::NativeOperations
importNativeOperations(nb::handle target, nb::handle operationNames) {
  return selectTranslation()->importNativeOperations(target, operationNames,
                                                     false, std::nullopt);
}

mlir::TargetEnvironment
importTargetEnvironment(nb::handle source, std::optional<size_t> nativeWidth,
                        bool strict) {
  using Target = mlir::CompilerTarget;
  if (nativeWidth == 0U) {
    throw nb::value_error("native_num_qubits must be positive");
  }
  const auto translation = selectTranslation();
  auto target =
      nativeWidth
          ? unwrap(Target::forGateSet(translation->importNativeOperations(
                source, nb::none(), strict, nativeWidth)))
          : translation->importTarget(source, nb::none(), std::nullopt, strict);
  return {target, translation->payloadSpecification(source)};
}

mlir::PayloadSpecification payloadSpecification(nb::handle target) {
  return selectTranslation()->payloadSpecification(target);
}

} // namespace mqt::bindings::qiskit
