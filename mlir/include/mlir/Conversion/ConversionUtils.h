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

#include <mlir/IR/Region.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>
#include <mlir/Transforms/DialectConversion.h>

namespace mlir {

/**
 * @brief Inlines @p source into @p dest and converts the entry block signature.
 *
 * @details Moves all blocks of @p source to the end of @p dest and converts the
 * argument types of the resulting entry block using @p typeConverter. This is
 * the canonical way to migrate a region from one dialect to another during a
 * dialect conversion when the block arguments change type.
 *
 * @param source The region whose blocks are moved out.
 * @param dest The region the blocks are moved into.
 * @param rewriter The conversion rewriter driving the current pass.
 * @param typeConverter The type converter used to convert the entry block
 * signature.
 * @return Whether converting the entry block signature succeeded.
 */
inline LogicalResult moveRegion(Region& source, Region& dest,
                                ConversionPatternRewriter& rewriter,
                                const TypeConverter* typeConverter) {
  rewriter.inlineRegionBefore(source, dest, dest.end());
  auto* block = &dest.front();
  TypeConverter::SignatureConversion sc(block->getNumArguments());
  if (failed(
          typeConverter->convertSignatureArgs(block->getArgumentTypes(), sc))) {
    return failure();
  }
  rewriter.applySignatureConversion(block, sc);
  return success();
}

} // namespace mlir
