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

// Keep the translation interface visible where the factory is instantiated.
#include "QiskitTranslation.h" // IWYU pragma: keep

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <cstddef>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mqt::bindings::qiskit {
namespace nb = nanobind;

namespace {
struct InstalledVersion {
  unsigned int major = 0;
  unsigned int minor = 0;
  unsigned int patch = 0;
  std::string text;
};
} // namespace

[[nodiscard]] static unsigned int parseComponent(std::string_view text,
                                                 size_t& offset) {
  const auto start = offset;
  unsigned int value = 0;
  while (offset < text.size() && text[offset] >= '0' && text[offset] <= '9') {
    const auto digit = static_cast<unsigned int>(text[offset] - '0');
    if (value > (std::numeric_limits<unsigned int>::max() - digit) / 10U) {
      throw std::runtime_error("invalid Qiskit version '" + std::string(text) +
                               "'");
    }
    value = (value * 10U) + digit;
    ++offset;
  }
  if (offset == start) {
    throw std::runtime_error("invalid Qiskit version '" + std::string(text) +
                             "'");
  }
  return value;
}

static void requireSeparator(const std::string_view text, size_t& offset) {
  if (offset >= text.size() || text[offset] != '.') {
    throw std::runtime_error("invalid Qiskit version '" + std::string(text) +
                             "'");
  }
  ++offset;
}

[[nodiscard]] static std::string supportedVersionRanges() {
  std::string ranges;
#define MQT_QISKIT_VERSION(major, minor, suffix, minimumPatch, minimum, range) \
  ranges += ranges.empty() ? (range) : ", " range;
#include "SupportedVersions.inc"
#undef MQT_QISKIT_VERSION
  return ranges;
}

[[nodiscard]] static constexpr bool matchesVersion(
    const InstalledVersion& version, const unsigned int expectedMajor,
    const unsigned int expectedMinor, const unsigned int minimumPatch) {
  return version.major == expectedMajor && version.minor == expectedMinor &&
         version.patch >= minimumPatch;
}

using TranslationFactory = std::unique_ptr<VersionedTranslation> (*)();

[[nodiscard]] static TranslationFactory
translationFactory(const InstalledVersion& version) {
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
  if (version.text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
    return createCandidateTranslation;
  }
#endif
#define MQT_QISKIT_FACTORY_IMPL(suffix) createQiskit##suffix
#define MQT_QISKIT_VERSION(expectedMajor, expectedMinor, suffix, minimumPatch, \
                           minimum, range)                                     \
  if (matchesVersion(version, expectedMajor##U, expectedMinor##U,              \
                     minimumPatch##U)) {                                       \
    return MQT_QISKIT_FACTORY_IMPL(suffix);                                    \
  }
#include "SupportedVersions.inc"
#undef MQT_QISKIT_VERSION
#undef MQT_QISKIT_FACTORY_IMPL
  return nullptr;
}

[[nodiscard]] static InstalledVersion inspectInstalledVersion() {
  std::string text;
  try {
    text = nb::cast<std::string>(
        nb::module_::import_("qiskit").attr("__version__"));
  } catch (const std::exception& error) {
    throw std::runtime_error(
        "Qiskit circuit translation requires an installed Qiskit package: " +
        std::string(error.what()));
  }

  size_t offset = 0;
  const auto major = parseComponent(text, offset);
  requireSeparator(text, offset);
  const auto minor = parseComponent(text, offset);
  requireSeparator(text, offset);
  const auto patch = parseComponent(text, offset);
  if (offset != text.size()) {
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
    if (text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
      return {.major = major, .minor = minor, .patch = patch, .text = text};
    }
#endif
    throw std::runtime_error(
        "Qiskit circuit translation does not support version '" + text +
        "'; supported versions: " + supportedVersionRanges());
  }
  const InstalledVersion result{
      .major = major, .minor = minor, .patch = patch, .text = text};
  return result;
}

std::unique_ptr<VersionedTranslation> selectTranslation() {
  const auto version = inspectInstalledVersion();
  if (const auto factory = translationFactory(version); factory != nullptr) {
    return factory();
  }
  throw std::runtime_error(
      "Qiskit circuit translation does not support installed version '" +
      version.text + "'; supported versions: " + supportedVersionRanges());
}

} // namespace mqt::bindings::qiskit
