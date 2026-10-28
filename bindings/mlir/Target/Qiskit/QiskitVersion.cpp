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

// Keep the translation interface visible where the factory is instantiated.
#include "QiskitTranslation.h" // IWYU pragma: keep

#include "nanobind/nanobind.h"
#include "nanobind/stl/string.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace mqt::bindings::qiskit {
namespace nb = nanobind;

std::unique_ptr<VersionedTranslation> selectTranslation() {
  const auto qiskit = nb::module_::import_("qiskit");
  const auto version = nb::cast<uint32_t>(
      nb::module_::import_("qiskit._accelerate.capi").attr("API_VERSION"));
  using Factory = std::unique_ptr<VersionedTranslation> (*)();
  Factory factory = nullptr;
  uint32_t selectedMinimum = 0;
  if ((version >> 24U) == 2U) {
#define MQT_QISKIT_VERSION(minor, patch, minimum, range)                       \
  {                                                                            \
    constexpr auto min =                                                       \
        (2U << 24U) | (minor##U << 16U) | (patch##U << 8U) | 0xf0U;            \
    if (version >= min && min > selectedMinimum) {                             \
      factory = createQiskit2_##minor;                                         \
      selectedMinimum = min;                                                   \
    }                                                                          \
  }
#include "SupportedVersions.inc"
#undef MQT_QISKIT_VERSION
  }
  if (factory != nullptr) {
    return factory();
  }
  throw std::runtime_error(
      "Qiskit circuit translation does not support installed version '" +
      nb::cast<std::string>(qiskit.attr("__version__")) +
      "'; supported versions: >=2.5.0,<3.0.0");
}

mlir::CompilerTarget importTarget(nb::handle target, nb::handle operationNames,
                                  const std::optional<std::string>& name) {
  return selectTranslation()->importTarget(target, operationNames, name);
}

mlir::CompilerTarget::NativeOperations
importNativeOperations(nb::handle target, nb::handle operationNames) {
  return selectTranslation()->importNativeOperations(target, operationNames);
}

mlir::PayloadSpecification payloadSpecification(nb::handle target) {
  return selectTranslation()->payloadSpecification(target);
}

} // namespace mqt::bindings::qiskit
