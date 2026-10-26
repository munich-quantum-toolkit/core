/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "nanobind/nanobind.h"

#ifdef __APPLE__
#include <exception>
#include <new>
#include <stdexcept>
#endif

namespace mqt {
void registerBench(nanobind::module_& mod);
void registerDD(const nanobind::module_& mod);
void registerQDMI(nanobind::module_& mod);
void registerMLIR(nanobind::module_& mod);
} // namespace mqt

NB_MODULE(_core, mod) {
#ifdef __APPLE__
  // Translate in the image that throws until the native APIs return errors.
  nanobind::register_exception_translator([](const std::exception_ptr& error,
                                             void*) {
    const auto report = [](const char* type, const std::exception& failure) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-type-vararg): nanobind API.
      nanobind::chain_error(nanobind::builtins()[type], "%s", failure.what());
    };
    try {
      std::rethrow_exception(error);
    } catch (const std::bad_alloc& failure) {
      report("MemoryError", failure);
    } catch (const std::domain_error& failure) {
      report("ValueError", failure);
    } catch (const std::invalid_argument& failure) {
      report("ValueError", failure);
    } catch (const std::length_error& failure) {
      report("ValueError", failure);
    } catch (const std::out_of_range& failure) {
      report("IndexError", failure);
    } catch (const std::range_error& failure) {
      report("ValueError", failure);
    } catch (const std::overflow_error& failure) {
      report("OverflowError", failure);
    } catch (const std::exception& failure) {
      report("RuntimeError", failure);
    }
  });
#endif
  auto bench = mod.def_submodule("bench");
  const auto dd = mod.def_submodule("dd");
  auto qdmi = mod.def_submodule("qdmi");
  auto mlir = mod.def_submodule("mlir");
  // Set the public scope before nanobind records class and function module
  // names.
  bench.attr("__name__") = "mqt.core.bench";
  dd.attr("__name__") = "mqt.core.dd";
  qdmi.attr("__name__") = "mqt.core.qdmi";
  mlir.attr("__name__") = "mqt.core.mlir";
  mqt::registerBench(bench);
  mqt::registerDD(dd);
  mqt::registerQDMI(qdmi);
  mqt::registerMLIR(mlir);
}
