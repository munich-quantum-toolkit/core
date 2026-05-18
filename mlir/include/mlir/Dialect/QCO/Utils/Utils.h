#pragma once

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>

namespace mlir::qco {

/**
 * @brief Find the entry point function with entry_point attribute
 *
 * @details
 * Searches for the LLVM function marked with the "entry_point" attribute in
 * the passthrough attributes.
 *
 * @param op The module operation to search in.
 * @returns the entry point function, or nullptr if not found.
 */
func::FuncOp getEntryPoint(ModuleOp op);

} // namespace mlir::qco