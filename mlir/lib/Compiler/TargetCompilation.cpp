/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Compiler/TargetCompilation.h"

#include "mlir/Compiler/Target.h"
#include "mlir/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Support/Passes.h"

#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>

#include <algorithm>
#include <array>
#include <utility>

namespace mlir {

[[nodiscard]] static const char*
symbolicFusionBasisName(const CompilerTarget::SingleQubitBasis basis) {
  using Basis = CompilerTarget::SingleQubitBasis;
  constexpr std::array names{
      std::pair{Basis::U, "u"},     std::pair{Basis::ZSXX, "zsxx"},
      std::pair{Basis::R, "r"},     std::pair{Basis::XZX, "xzx"},
      std::pair{Basis::XYX, "xyx"}, std::pair{Basis::ZYZ, "zyz"},
      std::pair{Basis::ZXZ, "zxz"},
  };
  // libc++ uses a pointer here, whereas MSVC uses an iterator class.
  // NOLINTNEXTLINE(readability-qualified-auto)
  const auto name = std::ranges::find_if(
      names, [basis](const auto& entry) { return entry.first == basis; });
  return name == names.end() ? nullptr : name->second;
}

void populateTargetCompilationPipeline(OpPassManager& pm,
                                       const CompilerTarget& target) {
  populateQCOCleanupPipeline(pm);
  populateDecomposeMultiControlledPipeline(pm, 3);
  populateDefaultQCOOptimizationPipeline(pm);
  pm.addPass(qco::createFuseTwoQubitGates());
  pm.addPass(qco::createMappingPass(target, qco::MappingPassOptions{}));
  populateQCOCleanupPipeline(pm);
  if (target.hasExplicitOperations()) {
    if (const auto targetBasis = target.synthesisBasis()) {
      if (const char* basis =
              symbolicFusionBasisName(targetBasis->singleQubit)) {
        qco::FuseSingleQubitUnitaryRunsOptions options;
        options.basis = basis;
        options.skipControlledBodies = true;
        pm.addPass(qco::createFuseSingleQubitUnitaryRuns(options));
      }
    }
  }
  pm.addPass(qco::createTargetNativeSynthesis(target));
  pm.addPass(createCSEPass());
  pm.addPass(createRemoveDeadValuesPass());
  pm.addPass(qco::createVerifyTargetConformance(target));
}

} // namespace mlir
