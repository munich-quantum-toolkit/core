/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <utility>

namespace mlir::qir {
#define GEN_PASS_DEF_QIRSETATTRIBUTESANDMETADATA
#include "mlir/Dialect/QIR/Transforms/Passes.h.inc"

namespace {

/// State object for tracking QIR metadata during conversion
struct Metadata {
  /// Number of qubits used in the module
  size_t numQubits{0};
  /// Number of measurement results stored in the module
  size_t numResults{0};
  /// Whether the module uses dynamic qubit management
  bool useDynamicQubit{false};
  /// Whether the module uses dynamic result management
  bool useDynamicResult{false};
  /// Whether the module uses arrays
  bool useArrays{false};
  /// Whether the module uses backward branching (0 = none, 1 = iteration based,
  /// 2 = condition based, 3 = both)
  int backwardsBranching{0};
};

/**
 * @brief Attaches the required attributes to the function marked as
 * entry_point.
 */
struct QIRSetAttributesAndMetadata final
    : impl::QIRSetAttributesAndMetadataBase<QIRSetAttributesAndMetadata> {
  using QIRSetAttributesAndMetadataBase::QIRSetAttributesAndMetadataBase;

protected:
  void runOnOperation() override {
    IRRewriter rewriter(&getContext());
    auto main = getMainFunction(getOperation());
    setMetadata(main, useAdaptive ? getAdaptive(main) : getBase(main),
                rewriter);
  }

private:
  /// Clear and set QIR base profile metadata.
  ///
  /// Adds the required metadata attributes for QIR base profile compliance:
  /// - `entry_point`: Marks the main entry point function
  /// - `output_labeling_schema`: labeled
  /// - `qir_profiles`: base_profile
  /// - `required_num_qubits`: Number of qubits used
  /// - `required_num_results`: Number of measurement results
  /// - `qir_major_version`: 2
  /// - `qir_minor_version`: 1
  /// - `dynamic_qubit_management`: true/false
  /// - `dynamic_result_management`: true/false
  ///
  /// These attributes are required by the QIR specification and inform QIR
  /// consumers about the module's resource requirements and capabilities.
  void setMetadata(LLVM::LLVMFuncOp& main, const Metadata& metadata,
                   IRRewriter& rewriter) {
    auto m = getOperation();
    const auto createFlag = [&](LLVM::ModFlagBehavior behavior, StringRef name,
                                int32_t val) {
      return LLVM::ModuleFlagAttr::get(m->getContext(), behavior,
                                       rewriter.getStringAttr(name),
                                       rewriter.getI32IntegerAttr(val));
    };

    const SmallVector<Attribute> attributes{
        rewriter.getStringAttr("entry_point"),
        rewriter.getStrArrayAttr({"output_labeling_schema", "labeled"}),
        rewriter.getStrArrayAttr({"qir_profiles", useAdaptive
                                                      ? "adaptive_profile"
                                                      : "base_profile"}),
        rewriter.getStrArrayAttr(
            {"required_num_qubits", std::to_string(metadata.numQubits)}),
        rewriter.getStrArrayAttr(
            {"required_num_results", std::to_string(metadata.numResults)})};

    main->setAttr("passthrough", rewriter.getArrayAttr(attributes));

    rewriter.setInsertionPointToEnd(m.getBody());

    SmallVector<Attribute> flags{
        createFlag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2),
        createFlag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 1),
        createFlag(LLVM::ModFlagBehavior::Error, "dynamic_qubit_management",
                   static_cast<int32_t>(metadata.useDynamicQubit)),
        createFlag(LLVM::ModFlagBehavior::Error, "dynamic_result_management",
                   static_cast<int32_t>(metadata.useDynamicResult))};

    if (useAdaptive) {
      flags.emplace_back(createFlag(LLVM::ModFlagBehavior::Error,
                                    "backwards_branching",
                                    metadata.backwardsBranching));
      flags.emplace_back(createFlag(LLVM::ModFlagBehavior::Error, "arrays",
                                    static_cast<int32_t>(metadata.useArrays)));
    }

    removeExistingModuleFlags(m, rewriter);
    LLVM::ModuleFlagsOp::create(rewriter, m.getLoc(),
                                rewriter.getArrayAttr(flags));
  }

  /// Remove existing module flag operations from module.
  /// Note that this might also erase non-QIR module flag operations, but for
  /// now, we assume that there are no others.
  static void removeExistingModuleFlags(ModuleOp m, IRRewriter& rewriter) {
    SmallVector<Operation*> flagOps;
    m->walk([&](LLVM::ModuleFlagsOp op) { flagOps.emplace_back(op); });
    for (Operation* op : llvm::make_early_inc_range(flagOps)) {
      rewriter.eraseOp(op);
    }
  }

  /// Count the number of uniquely indexed qubit pointers.
  /// Assumes that qubits are constant integers that are converted to
  /// an integer pointer and then used in (at least) one quantum instruction.
  static size_t getNumQubits(LLVM::LLVMFuncOp& main) {
    static constexpr StringRef QIS_PREFIX = "__quantum__qis";

    DenseSet<APInt> seen;
    main->walk([&](LLVM::ConstantOp constOp) {
      if (constOp.use_empty()) {
        return;
      }

      const auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr) {
        return;
      }

      if (!intAttr.getType().isInteger()) { // Not a ": index".
        return;
      }

      const auto userIt =
          llvm::find_if(constOp->getUsers(), [](Operation* user) {
            return isa<LLVM::IntToPtrOp>(user);
          });
      if (userIt == constOp->user_end()) {
        return;
      }

      const auto toPtrOp = cast<LLVM::IntToPtrOp>(*userIt);
      const auto callIt =
          llvm::find_if(toPtrOp->getUses(), [](OpOperand& operand) {
            auto callOp = dyn_cast<LLVM::CallOp>(operand.getOwner());
            if (!callOp) {
              return false;
            }

            auto callee = callOp.getCallee();
            if (!callee.has_value()) {
              return false;
            }

            if (*callee == QIR_MEASURE) {

              // The following assumes that the first argument of a
              // measurement call is the qubit. This may (or may not) hold in
              // the future.

              return operand.getOperandNumber() == 0;
            }

            return callee->starts_with(QIS_PREFIX);
          });
      if (callIt == toPtrOp->use_end()) {
        return;
      }

      // The set ensures that we don't insert the same index multiple times.
      seen.insert(intAttr.getValue());
    });

    return seen.size();
  }

  /// Count the number of uniquely indexed result_record_output statements.
  static size_t getNumResults(LLVM::LLVMFuncOp& main) {
    DenseSet<APInt> seen;
    main->walk([&](LLVM::CallOp callOp) {
      if (!callOp.getCallee()) {
        return;
      }

      if (*callOp.getCallee() != QIR_RECORD_OUTPUT) {
        return;
      }

      const auto operand = callOp->getOperand(0);
      auto toPtrOp = dyn_cast<LLVM::IntToPtrOp>(operand.getDefiningOp());
      if (!toPtrOp) {
        return;
      }

      const auto arg = toPtrOp.getArg();
      auto constOp = dyn_cast<LLVM::ConstantOp>(arg.getDefiningOp());
      if (!constOp) {
        return;
      }

      const auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr) {
        return;
      }

      // The set ensures that we don't insert the same index multiple times.
      seen.insert(intAttr.getValue());
    });

    return seen.size();
  }

  /// Determine whether an loop (as a set of blocks) is an iterative loop (true)
  /// or a conditionally terminated loop (false).
  static bool classifyLoop(const SmallPtrSet<Block*, 8>& loop) {
    for (Block* block : loop) {
      Operation* terminator = block->getTerminator();
      assert(terminator != nullptr);

      if (auto condBrOp = dyn_cast<LLVM::CondBrOp>(terminator)) {
        auto condition = condBrOp.getCondition();
        auto callOp = dyn_cast<LLVM::CallOp>(condition.getDefiningOp());

        // If the condition is not produced by a measurement call, we
        // consider it a basic loop.
        if (!callOp || !callOp.getCallee()) {
          return true;
        }

        // If the condition has been produced by a measurement call
        // (e.g. a until-zero-measurement loop), and breaks outside the loop,
        // we found a "conditionally terminating loop".
        if (*callOp.getCallee() == QIR_READ_RESULT &&
            (!loop.contains(condBrOp.getTrueDest()) ||
             !loop.contains(condBrOp.getFalseDest()))) {
          return false;
        }

        // Unseen edge case (so far): The condition of the terminator
        // operation is produced by a function call, which isn't a
        // measurement.
        return true;
      }
    }
  }

  /// Return pair of booleans, indicating whether the entry point uses
  /// iterations = [0] or conditionally terminated loops = [1].
  static std::pair<bool, bool>
  usesBackwardsBranching(LLVM::LLVMFuncOp& main, const DominanceInfo& domInfo) {
    bool useIteration{false};
    bool useCondTerm{false};

    SmallVector<Block*, 8> worklist;

    for (Block& block : main.getBlocks()) {
      for (Block* successor : block.getSuccessors()) {
        if (domInfo.dominates(successor, &block)) { // Back edge.
          Block* header = successor;
          Block* tail = &block;

          SmallPtrSet<Block*, 8> loop{header};
          if (header != tail) {
            worklist.push_back(tail);
          }

          while (!worklist.empty()) {
            Block* curr = worklist.pop_back_val();
            for (Block* pred : curr->getPredecessors()) {
              if (loop.insert(pred).second) {
                worklist.push_back(pred);
              }
            }
          }

          if(classifyLoop(loop)) {
            useIteration |= true;
          } else {
            useCondTerm |= true;
          }


          loop.clear();
        }
      }
    }

    return std::make_pair(useIteration, useCondTerm);
  }

  /// Return triple of booleans, indicating whether the entry point uses
  /// dynamic qubits = [0], dynamic results = [1], or dynamic arrays = [2].
  static std::tuple<bool, bool, bool> usesDynamic(LLVM::LLVMFuncOp& main) {
    bool useDynamicQubit{false};
    bool useDynamicResult{false};
    bool useArrays{false};

    main->walk([&](LLVM::CallOp callOp) {
      if (!callOp.getCallee()) {
        return;
      }

      const auto name = *callOp.getCallee();
      if (name == QIR_QUBIT_ALLOC) {
        useDynamicQubit = true;
      } else if (name == QIR_RESULT_ALLOC) {
        useDynamicResult = true;
      } else if (name == QIR_QUBIT_ARRAY_ALLOC) {
        useDynamicQubit = true;
        useArrays = true;
      } else if (name == QIR_RESULT_ARRAY_ALLOC) {
        useDynamicResult = true;
        useArrays = true;
      }
    });

    return std::make_tuple(useDynamicQubit, useDynamicResult, useArrays);
  }

  /// Return the metadata for a QIR base profile compliant program.
  static Metadata getBase(LLVM::LLVMFuncOp& main) {
    return {.numQubits = getNumQubits(main),
            .numResults = getNumResults(main),
            .useDynamicQubit = false,
            .useDynamicResult = false,
            .useArrays = false,
            .backwardsBranching = 0};
  }

  /// Return the metadata for a QIR base profile compliant program.
  Metadata getAdaptive(LLVM::LLVMFuncOp& main) {
    const auto& domInfo = getAnalysis<DominanceInfo>();
    const auto [useIteration, useCondTerm] =
        usesBackwardsBranching(main, domInfo);
    const auto [useDynamicQubit, useDynamicResult, useArrays] =
        usesDynamic(main);

    Metadata md;
    md.useDynamicQubit = useDynamicQubit;
    md.useDynamicResult = useDynamicResult;
    md.useArrays = useArrays;

    if (!useDynamicQubit) {
      md.numQubits = getNumQubits(main);
    }

    if (!useDynamicResult) {
      md.numResults = getNumResults(main);
    }

    if (useIteration) {
      md.backwardsBranching = useCondTerm ? 3 : 1;
    } else if (useCondTerm) {
      md.backwardsBranching = 2;
    }

    return md;
  }
};
} // namespace
} // namespace mlir::qir
