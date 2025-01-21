#pragma once

#include "ir/QuantumComputation.hpp"

#include <mlir/Pass/Pass.h>
#include <set>

namespace mlir {

class RewritePatternSet;

namespace mqt {

#define GEN_PASS_DECL
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

void populateToQuantumComputationPatterns(RewritePatternSet& patterns,
                                          qc::QuantumComputation& circuit);
void populateFromQuantumComputationPatterns(RewritePatternSet& patterns,
                                            qc::QuantumComputation& circuit);
void populatePassWithSingleQubitGateRewritePattern(RewritePatternSet& patterns);
void populatePassWithMultiQubitGateRewritePattern(RewritePatternSet& patterns);

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/MQT/Transforms/Passes.h.inc"

} // namespace mqt
} // namespace mlir
