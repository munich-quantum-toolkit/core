#pragma once

namespace mlir {
class ModuleOp;
class PassManager;

/**
 * @brief Populate the QIR conversion pipeline on the given pass manager.
 */
void populateQIRConversionPipeline(mlir::PassManager& pm,
                                   bool useAdaptive = false);
} // namespace mlir