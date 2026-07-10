/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#pragma once

#include "mlir/Support/PrettyPrinting.h"

#include <gtest/gtest.h>
#include <llvm/ADT/SmallString.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <cstdlib>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace mqt::test {

template <typename BuilderT, typename RetT = void> struct NamedBuilder {
  const char* name = nullptr;
  RetT (*fn)(BuilderT&) = nullptr;

  constexpr NamedBuilder(const char* nameIn, RetT (*fnIn)(BuilderT&)) noexcept
      : name(nameIn), fn(fnIn) {}

  // NOLINTNEXTLINE(*-explicit-constructor)
  constexpr NamedBuilder(std::nullptr_t) noexcept : fn(nullptr) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return fn != nullptr;
  }
};

template <typename BuilderT, typename RetT>
[[nodiscard]] constexpr NamedBuilder<BuilderT, RetT>
namedBuilder(const char* name, RetT (*fn)(BuilderT&)) noexcept {
  return NamedBuilder<BuilderT, RetT>{name, fn};
}

template <typename BuilderT> struct NamedMLIRBuilder {
  using SingleFn = mlir::Value (*)(BuilderT&);
  using MultiFn = mlir::SmallVector<mlir::Value> (*)(BuilderT&);

  const char* name = nullptr;
  std::variant<std::monostate, SingleFn, MultiFn> fn;

  constexpr NamedMLIRBuilder(const char* nameIn, SingleFn fnIn) noexcept
      : name(nameIn), fn(fnIn) {}

  constexpr NamedMLIRBuilder(const char* nameIn, MultiFn fnIn) noexcept
      : name(nameIn), fn(fnIn) {}

  // NOLINTNEXTLINE(*-explicit-constructor)
  constexpr NamedMLIRBuilder(std::nullptr_t) noexcept : fn(std::monostate{}) {}

  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return !std::holds_alternative<std::monostate>(fn);
  }
};

template <typename BuilderT>
[[nodiscard]] constexpr NamedMLIRBuilder<BuilderT>
namedBuilder(const char* name, mlir::Value (*fn)(BuilderT&)) noexcept {
  return NamedMLIRBuilder<BuilderT>{name, fn};
}

template <typename BuilderT>
[[nodiscard]] constexpr NamedMLIRBuilder<BuilderT>
namedBuilder(const char* name,
             mlir::SmallVector<mlir::Value> (*fn)(BuilderT&)) noexcept {
  return NamedMLIRBuilder<BuilderT>{name, fn};
}

template <typename BuilderT, typename... Args>
[[nodiscard]] mlir::OwningOpRef<mlir::ModuleOp>
buildMLIRProgram(mlir::MLIRContext* context,
                 const NamedMLIRBuilder<BuilderT>& builder, Args&&... args) {
  return std::visit(
      [&]<typename T>(T fn) -> mlir::OwningOpRef<mlir::ModuleOp> {
        using FnT = std::decay_t<T>;
        if constexpr (std::is_same_v<FnT, std::monostate>) {
          return {};
        } else if constexpr (requires {
                               BuilderT::build(context, fn,
                                               std::forward<Args>(args)...);
                             }) {
          return BuilderT::build(context, fn, std::forward<Args>(args)...);
        } else {
          return {};
        }
      },
      builder.fn);
}

[[nodiscard]] constexpr const char* displayName(const char* name) noexcept {
  return name != nullptr ? name : "<null>";
}

/**
 * @brief Returns true when IR printing is always enabled via the environment.
 *
 * Set the environment variable @c MQT_MLIR_TEST_PRINT_IR to any non-empty
 * value to unconditionally print IR in every test.
 */
[[nodiscard]] inline bool irPrintingForced() noexcept {
  const char* const env = std::getenv("MQT_MLIR_TEST_PRINT_IR");
  return env != nullptr && *env != '\0';
}

/**
 * @brief RAII helper that defers IR printing until the end of a test.
 *
 * @details
 * Each call to @c record() eagerly renders the given @c ModuleOp to a string
 * and stores it alongside its header. When the @c DeferredPrinter is destroyed
 * (i.e., at the end of the test body) it either:
 *   - flushes all captured IR to @c llvm::errs() when the test has already
 *     recorded a failure (@c ::testing::Test::HasFailure()), or
 *   - flushes unconditionally when the @c MQT_MLIR_TEST_PRINT_IR environment
 *     variable is set to a non-empty value.
 *
 * In all other cases the captured strings are simply discarded, avoiding the
 * significant I/O overhead of box-printing on every passing test.
 *
 * Usage:
 * @code
 *   TEST_P(MyTest, MyCase) {
 *     DeferredPrinter printer;
 *     auto module = build(...);
 *     printer.record(module.get(), "My IR");
 *     EXPECT_TRUE(someCheck(module.get()));
 *   }
 * @endcode
 */
class DeferredPrinter {
public:
  DeferredPrinter() = default;

  // Non-copyable, non-movable — lives exactly as long as the test scope.
  DeferredPrinter(const DeferredPrinter&) = delete;
  DeferredPrinter& operator=(const DeferredPrinter&) = delete;
  DeferredPrinter(DeferredPrinter&&) = delete;
  DeferredPrinter& operator=(DeferredPrinter&&) = delete;

  /**
   * @brief Capture the current state of @p module under the given @p header.
   *
   * The IR is rendered to a string immediately (so later mutations to the
   * module do not affect already-captured snapshots). The output is deferred
   * until the printer is destroyed.
   */
  void record(mlir::ModuleOp module, llvm::StringRef header) {
    llvm::SmallString<4096> irString;
    llvm::raw_svector_ostream irStream(irString);
    module.print(irStream);
    entries_.emplace_back(header.str(), irString.str());
  }

  /**
   * @brief Flush all captured entries to @c llvm::errs() if needed.
   *
   * Triggered by a test failure or the @c MQT_MLIR_TEST_PRINT_IR env var.
   */
  ~DeferredPrinter() {
    if (!irPrintingForced() && !::testing::Test::HasFailure()) {
      return;
    }
    for (const auto& [header, ir] : entries_) {
      mlir::printBoxTop();
      mlir::printBoxLine(header);
      mlir::printBoxMiddle();
      mlir::printBoxText(ir);
      mlir::printBoxBottom();
      llvm::errs().flush();
    }
  }

private:
  std::vector<std::pair<std::string, std::string>> entries_;
};

} // namespace mqt::test

#define MQT_NAMED_BUILDER(fn) ::mqt::test::namedBuilder(#fn, fn)
