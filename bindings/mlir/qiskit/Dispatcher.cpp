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

#include <Python.h>

#include <charconv>
#include <stdexcept>
#include <string>
#include <string_view>

namespace mqt::bindings::qiskit {
namespace {

[[nodiscard]] unsigned int parseComponent(std::string_view text,
                                          std::size_t& offset) {
  const auto begin = text.data() + offset;
  const auto end = text.data() + text.size();
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
#define MQT_QISKIT_ADAPTER(major, minor, suffix, minimum, range)               \
  ranges += ranges.empty() ? range : ", " range;
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
  return ranges;
}

} // namespace

InstalledVersion inspectInstalledVersion() {
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
  return {.major = major, .minor = minor, .patch = patch, .text = text};
}

bool hasSupportedAdapter(const InstalledVersion& version) {
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
  if (version.text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
    return true;
  }
#endif
#define MQT_QISKIT_ADAPTER(adapterMajor, adapterMinor, suffix, minimum, range) \
  if (version.major == adapterMajor##U && version.minor == adapterMinor##U) {  \
    return true;                                                               \
  }
#include "SupportedAdapters.inc"
#undef MQT_QISKIT_ADAPTER
  return false;
}

std::unique_ptr<Adapter> selectAdapter() {
  const auto version = inspectInstalledVersion();
#ifdef MQT_QISKIT_CAPI_CANDIDATE_VERSION
  if (version.text == MQT_QISKIT_CAPI_CANDIDATE_VERSION) {
    return createCandidateAdapter();
  }
#endif
#define MQT_QISKIT_CREATE_ADAPTER_IMPL(suffix) createAdapter##suffix()
#define MQT_QISKIT_ADAPTER(adapterMajor, adapterMinor, suffix, minimum, range) \
  if (version.major == adapterMajor##U && version.minor == adapterMinor##U) {  \
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
