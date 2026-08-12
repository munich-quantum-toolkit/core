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

#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/Value.h>

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir::mqt::angle {

constexpr uint32_t MACHINE_WIDTH = 64;
constexpr size_t FLOAT_TO_BITS_OPERATION_ESTIMATE = 159;
constexpr llvm::StringLiteral FINAL_QUANTIZATION_ATTR =
    "mqt.angle.final_precision_bits";

struct QuantizedRadians {
  Value bits;
  uint32_t bitWidth = 0;
};

struct AngleResize {
  Value source;
  uint32_t sourceWidth = 0;
  uint32_t targetWidth = 0;
  llvm::SmallVector<Operation*> operations;
};

/** Return whether @p bitWidth is supported by MQT's OpenQASM angle lowering. */
[[nodiscard]] bool isSupportedWidth(uint32_t bitWidth);

/**
 * Convert a finite binary64 radian value to an OpenQASM angle bit pattern.
 *
 * The conversion reduces modulo 2*pi and rounds to nearest, ties to even.
 * Non-finite inputs and unsupported widths return std::nullopt.
 */
[[nodiscard]] std::optional<uint64_t> quantize(double radians,
                                               uint32_t bitWidth);

/** Convert an OpenQASM angle bit pattern to its binary64 radian bridge. */
[[nodiscard]] double toRadians(uint64_t bits, uint32_t bitWidth);

/** Resize an OpenQASM angle bit pattern with the specified rounding rule. */
[[nodiscard]] uint64_t resize(uint64_t bits, uint32_t sourceWidth,
                              uint32_t targetWidth);

/** Build the canonical runtime float-to-OpenQASM-angle conversion. */
[[nodiscard]] Value buildFloatToBits(OpBuilder& builder, Location loc,
                                     Value radians, uint32_t bitWidth);

/** Build the canonical OpenQASM-angle-to-radians ABI bridge. */
[[nodiscard]] Value buildBitsToRadians(OpBuilder& builder, Location loc,
                                       Value bits);

/** Build the OpenQASM-defined precision conversion between angle bit widths. */
[[nodiscard]] Value buildResize(OpBuilder& builder, Location loc, Value bits,
                                uint32_t targetWidth);

/**
 * Quantize @p radians and materialize the result as an f64 radian value.
 *
 * Compile-time constants become an integer bit-pattern constant followed by
 * the canonical ABI bridge. Dynamic values use the full canonical conversion.
 */
[[nodiscard]] Value buildQuantizedRadians(OpBuilder& builder, Location loc,
                                          Value radians, uint32_t bitWidth);

/** Recognize the canonical angle-bit-pattern to f64 radian bridge. */
[[nodiscard]] std::optional<QuantizedRadians>
matchQuantizedRadians(Value radians);

/** Recognize the canonical runtime f64-to-angle-bit-pattern conversion. */
[[nodiscard]] std::optional<Value> matchFloatToBits(Value bits);

/** Recognize an OpenQASM-defined precision conversion between angle widths. */
[[nodiscard]] std::optional<AngleResize> matchResize(Value bits);

} // namespace mlir::mqt::angle
