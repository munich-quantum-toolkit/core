/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "Dispatcher.h"

#include "QiskitAdapter.h" // NOLINT(misc-include-cleaner)

// CPython's limited-API umbrella provides these declarations indirectly.
// NOLINTBEGIN(misc-include-cleaner)
#include <Python.h>
// NOLINTEND(misc-include-cleaner)

#include <charconv>
#include <cstddef>
#include <memory> // NOLINT(misc-include-cleaner)
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>

namespace mqt::bindings::qiskit {
namespace {

[[nodiscard]] unsigned int parseComponent(std::string_view text,
                                          std::size_t& offset) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  const char* const begin = text.data() + offset;
  // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  const char* const end = text.data() + text.size();
  unsigned int value = 0;
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc{} || result.ptr == begin) {
    throw std::runtime_error("invalid Qiskit version '" + std::string(text) +
                             "'");
  }
  offset = static_cast<std::size_t>(result.ptr - text.data());
  return value;
}

void requireSeparator(const std::string_view text, std::size_t& offset) {
  if (offset >= text.size() || text[offset] != '.') {
    throw std::runtime_error("invalid Qiskit version '" + std::string(text) +
                             "'");
  }
  ++offset;
}

[[nodiscard]] std::string supportedVersionRanges() {
  std::string ranges;
#define MQT_QISKIT_ADAPTER(major, minor, suffix, minimumPatch, minimum, range) \
  ranges += ranges.empty() ? (range) : ", " range;
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
  return ranges;
}

[[nodiscard]] constexpr bool matchesAdapterVersion(
    const InstalledVersion& version, const unsigned int adapterMajor,
    const unsigned int adapterMinor, const unsigned int minimumPatch) {
  return version.major == adapterMajor && version.minor == adapterMinor &&
         version.patch >= minimumPatch;
}

static_assert(matchesAdapterVersion(
    {.major = 2U, .minor = 6U, .patch = 2U, .text = "2.6.2"}, 2U, 6U, 2U));
static_assert(!matchesAdapterVersion(
    {.major = 2U, .minor = 6U, .patch = 1U, .text = "2.6.1"}, 2U, 6U, 2U));

} // namespace

InstalledVersion inspectInstalledVersion() {
  // NOLINTBEGIN(misc-include-cleaner)
  PyObject* module = PyImport_ImportModule("qiskit");
  if (module == nullptr) {
    PyErr_Clear();
    throw std::runtime_error(
        "the Qiskit compiler bridge requires an installed Qiskit package");
  }
  PyObject* versionObject = PyObject_GetAttrString(module, "__version__");
  Py_DECREF(module);
  if (versionObject == nullptr) {
    PyErr_Clear();
    throw std::runtime_error("installed Qiskit does not expose __version__");
  }
  PyObject* versionBytes =
      PyUnicode_AsEncodedString(versionObject, "utf-8", "strict");
  Py_DECREF(versionObject);
  if (versionBytes == nullptr) {
    PyErr_Clear();
    throw std::runtime_error("installed Qiskit has a non-text __version__");
  }
  const char* versionChars = PyBytes_AsString(versionBytes);
  if (versionChars == nullptr) {
    Py_DECREF(versionBytes);
    PyErr_Clear();
    throw std::runtime_error("installed Qiskit has a non-text __version__");
  }
  const std::string text(versionChars);
  Py_DECREF(versionBytes);

  std::size_t offset = 0;
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
        "Qiskit compiler bridge unavailable for prerelease or non-final "
        "version '" +
        text + "'; supported versions: " + supportedVersionRanges() +
        " (final releases)");
  }
  const InstalledVersion result{
      .major = major, .minor = minor, .patch = patch, .text = text};
  // NOLINTEND(misc-include-cleaner)
  return result;
}

bool hasSupportedAdapter(const InstalledVersion& version) {
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
  if (version.text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
    return true;
  }
#endif
  bool supported = false;
#define MQT_QISKIT_ADAPTER(adapterMajor, adapterMinor, suffix, minimumPatch,   \
                           minimum, range)                                     \
  supported =                                                                  \
      supported || matchesAdapterVersion(version, adapterMajor##U,             \
                                         adapterMinor##U, minimumPatch##U);
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
  return supported;
}

std::unique_ptr<Adapter> selectAdapter() {
  const auto version = inspectInstalledVersion();
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
  if (version.text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
    return createCandidateAdapter();
  }
#endif
#define MQT_QISKIT_CREATE_ADAPTER_IMPL(suffix) createAdapter##suffix()
#define MQT_QISKIT_ADAPTER(adapterMajor, adapterMinor, suffix, minimumPatch,   \
                           minimum, range)                                     \
  if (matchesAdapterVersion(version, adapterMajor##U, adapterMinor##U,         \
                            minimumPatch##U)) {                                \
    return MQT_QISKIT_CREATE_ADAPTER_IMPL(suffix);                             \
  }
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
#undef MQT_QISKIT_CREATE_ADAPTER_IMPL
  if (!hasSupportedAdapter(version)) {
    throw std::runtime_error(
        "Qiskit compiler bridge unavailable for installed version '" +
        version.text + "'; supported versions: " + supportedVersionRanges() +
        " (final releases)");
  }
  throw std::runtime_error(
      "Qiskit compiler bridge adapter registry is inconsistent");
}

} // namespace mqt::bindings::qiskit
