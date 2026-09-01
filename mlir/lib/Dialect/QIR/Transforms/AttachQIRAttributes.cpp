/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/MQT/IR/MQTDialect.h"
#include "mlir/Dialect/QIR/QIRDefinitions.h"
#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallSet.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Analysis/SliceWalk.h>
#include <mlir/Dialect/LLVMIR/LLVMAttrs.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Dominance.h>
#include <mlir/IR/OpDefinition.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Value.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/LogicalResult.h>

#include <algorithm>
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
  llvm::SmallSet<std::string, 4> integerTypes;
  llvm::SmallSet<std::string, 4> floatingTypes;
  bool usesIRFunctions{false};
  bool usesMultipleTargetBranching{false};
  bool usesMultipleReturnPoints{false};
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
    auto main = getMainFunction(getOperation());
    if (!main) {
      return;
    }

    auto module = getOperation();
    const auto [useDynamicQubit, useDynamicResult, useArrays] =
        usesDynamic(module);
    if (!useAdaptive && (useDynamicQubit || useDynamicResult)) {
      module.emitError()
          << "QIR base profile does not support dynamic resource management";
      signalPassFailure();
      return;
    }

    auto numQubits = getNumQubits(module, !useDynamicQubit);
    auto numResults = getNumResults(module, !useDynamicResult);
    if (failed(numQubits) || failed(numResults)) {
      signalPassFailure();
      return;
    }

    Metadata metadata =
        useAdaptive ? getAdaptive(main, *numQubits, *numResults,
                                  useDynamicQubit, useDynamicResult, useArrays)
                    : getBase(*numQubits, *numResults);
    if (useAdaptive) {
      collectOptionalFeatures(module, main, metadata);
    }
    IRRewriter rewriter(&getContext());
    setMetadata(main, metadata, rewriter);
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
                                Attribute value) {
      return LLVM::ModuleFlagAttr::get(m->getContext(), behavior,
                                       rewriter.getStringAttr(name), value);
    };
    const auto createI32Flag = [&](LLVM::ModFlagBehavior behavior,
                                   StringRef name, int32_t value) {
      return createFlag(behavior, name, rewriter.getI32IntegerAttr(value));
    };
    const auto createBoolFlag = [&](LLVM::ModFlagBehavior behavior,
                                    StringRef name, bool value) {
      return createI32Flag(behavior, name, value ? 1 : 0);
    };

    const SmallVector<Attribute> attributes{
        rewriter.getStringAttr(::qir::ENTRY_POINT_ATTR),
        rewriter.getStrArrayAttr(
            {::qir::OUTPUT_LABELING_SCHEMA_ATTR, ::qir::LABELED_SCHEMA}),
        rewriter.getStrArrayAttr(
            {::qir::QIR_PROFILES_ATTR,
             useAdaptive ? ::qir::ADAPTIVE_PROFILE : ::qir::BASE_PROFILE}),
        rewriter.getStrArrayAttr(
            {"required_num_qubits", std::to_string(metadata.numQubits)}),
        rewriter.getStrArrayAttr(
            {"required_num_results", std::to_string(metadata.numResults)})};

    main->setAttr("passthrough", rewriter.getArrayAttr(attributes));
    mqt::removeEntryPoint(main);

    rewriter.setInsertionPointToEnd(m.getBody());

    SmallVector<Attribute> flags{
        createI32Flag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2),
        createI32Flag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 1),
        createBoolFlag(LLVM::ModFlagBehavior::Error, "dynamic_qubit_management",
                       metadata.useDynamicQubit),
        createBoolFlag(LLVM::ModFlagBehavior::Error,
                       "dynamic_result_management", metadata.useDynamicResult)};

    if (useAdaptive) {
      flags.emplace_back(createI32Flag(LLVM::ModFlagBehavior::Error,
                                       "backwards_branching",
                                       metadata.backwardsBranching));
      flags.emplace_back(createBoolFlag(LLVM::ModFlagBehavior::Error, "arrays",
                                        metadata.useArrays));
      if (metadata.usesIRFunctions) {
        flags.emplace_back(
            createBoolFlag(LLVM::ModFlagBehavior::Error, "ir_functions", true));
      }
      if (metadata.usesMultipleTargetBranching) {
        flags.emplace_back(createBoolFlag(LLVM::ModFlagBehavior::Error,
                                          "multiple_target_branching", true));
      }
      if (metadata.usesMultipleReturnPoints) {
        flags.emplace_back(createBoolFlag(LLVM::ModFlagBehavior::Error,
                                          "multiple_return_points", true));
      }
    }

    removeExistingModuleFlags(m, rewriter);
    const auto setTypes = [&](const StringRef name,
                              const llvm::SmallSet<std::string, 4>& types) {
      if (types.empty()) {
        m->removeAttr(name);
        return;
      }
      SmallVector<StringRef> values(types.begin(), types.end());
      llvm::sort(values);
      m->setAttr(name, rewriter.getStrArrayAttr(values));
    };
    setTypes("qir.int_computations", metadata.integerTypes);
    setTypes("qir.float_computations", metadata.floatingTypes);
    LLVM::ModuleFlagsOp::create(rewriter, m.getLoc(),
                                rewriter.getArrayAttr(flags));
  }

  /// Remove existing module flag operations from module.
  /// Note that this might also erase non-QIR module flag operations, but for
  /// now, we assume that there are no others.
  static void removeExistingModuleFlags(ModuleOp m, IRRewriter& rewriter) {
    SmallVector<Operation*> flagOps;
    m->walk([&](LLVM::ModuleFlagsOp op) { flagOps.emplace_back(op); });
    llvm::for_each(flagOps, [&](Operation* op) { rewriter.eraseOp(op); });
  }

  /// Return one past the greatest indexed qubit pointer.
  /// Assumes that qubits are constant integers that are converted to
  /// an integer pointer and then used in (at least) one quantum instruction.
  enum class PointerProvenance : uint8_t { Unresolved, Cycle, Resolved };

  static FailureOr<PointerProvenance>
  includeStaticPointer(Value pointer, StringRef resource, size_t& capacity,
                       ModuleOp module, SmallPtrSetImpl<Value>& resolving,
                       SmallPtrSetImpl<Value>* aggregates = nullptr) {
    auto toPtrOp = pointer.getDefiningOp<LLVM::IntToPtrOp>();
    if (toPtrOp) {
      auto constOp = toPtrOp.getArg().getDefiningOp<LLVM::ConstantOp>();
      if (!constOp) {
        return toPtrOp.emitError()
               << "statically addressed QIR " << resource
               << " must be converted from an integer constant";
      }
      const auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue());
      if (!intAttr || !intAttr.getType().isInteger()) {
        return constOp.emitError()
               << "QIR " << resource << " index must be an integer constant";
      }
      const auto& index = intAttr.getValue();
      if (index.isNegative() || index.getActiveBits() >= sizeof(size_t) * 8) {
        return constOp.emitError()
               << "QIR " << resource
               << " index must be non-negative and representable as a host "
                  "size";
      }
      capacity =
          std::max(capacity, static_cast<size_t>(index.getZExtValue()) + 1);
      return PointerProvenance::Resolved;
    }
    if (pointer.getDefiningOp<LLVM::ZeroOp>()) {
      capacity = std::max(capacity, size_t{1});
      return PointerProvenance::Resolved;
    }
    if (auto call = pointer.getDefiningOp<LLVM::CallOp>();
        call && call.getCallee() &&
        (*call.getCallee() == QIR_ARRAY_CREATE ||
         *call.getCallee() == QIR_TUPLE_CREATE)) {
      // Generic controlled QIS calls receive aggregate pointers. Their static
      // qubit constituents are counted from the stores that populate them.
      if (aggregates != nullptr) {
        aggregates->insert(pointer);
      }
      return PointerProvenance::Resolved;
    }

    auto blockArgument = dyn_cast<BlockArgument>(pointer);
    if (blockArgument) {
      if (aggregates != nullptr) {
        aggregates->insert(pointer);
      }
      Operation* anchor = blockArgument.getOwner()->getParentOp();
      if (!resolving.insert(pointer).second) {
        return PointerProvenance::Cycle;
      }

      auto function = dyn_cast<LLVM::LLVMFuncOp>(anchor);
      bool sawProvenance = false;
      bool sawCycle = false;
      bool unresolvedProvenance = false;
      LogicalResult status = success();
      if (function && !function.isExternal() &&
          blockArgument.getOwner() == &function.getBody().front()) {
        module.walk([&](Operation* operation) {
          if (failed(status)) {
            return;
          }
          auto call = dyn_cast<LLVM::CallOp>(operation);
          if (!call || !call.getCallee() ||
              *call.getCallee() != function.getSymName()) {
            return;
          }
          auto provenance = includeStaticPointer(
              call.getOperand(blockArgument.getArgNumber()), resource, capacity,
              module, resolving, aggregates);
          if (failed(provenance)) {
            status = failure();
            return;
          }
          sawProvenance |= *provenance == PointerProvenance::Resolved;
          sawCycle |= *provenance == PointerProvenance::Cycle;
          unresolvedProvenance |= *provenance == PointerProvenance::Unresolved;
        });
      } else {
        SmallVector<Value> worklist{pointer};
        SmallPtrSet<Value, 8> visited;
        while (!worklist.empty() && succeeded(status)) {
          Value current = worklist.pop_back_val();
          if (!current) {
            unresolvedProvenance = true;
            continue;
          }
          if (!visited.insert(current).second) {
            sawCycle = true;
            continue;
          }
          if (auto predecessors = getControlFlowPredecessors(current)) {
            unresolvedProvenance |= predecessors->empty();
            worklist.append(*predecessors);
            continue;
          }
          if (auto argument = dyn_cast<BlockArgument>(current)) {
            auto owner =
                dyn_cast<LLVM::LLVMFuncOp>(argument.getOwner()->getParentOp());
            if (!owner || argument.getOwner() != &owner.getBody().front()) {
              unresolvedProvenance = true;
              continue;
            }
          }
          auto provenance = includeStaticPointer(current, resource, capacity,
                                                 module, resolving, aggregates);
          if (failed(provenance)) {
            status = failure();
            continue;
          }
          sawProvenance |= *provenance == PointerProvenance::Resolved;
          sawCycle |= *provenance == PointerProvenance::Cycle;
          unresolvedProvenance |= *provenance == PointerProvenance::Unresolved;
        }
      }
      resolving.erase(pointer);
      if (failed(status)) {
        return failure();
      }
      if (unresolvedProvenance) {
        return PointerProvenance::Unresolved;
      }
      if (sawProvenance) {
        return PointerProvenance::Resolved;
      }
      return sawCycle ? PointerProvenance::Cycle
                      : PointerProvenance::Unresolved;
    }

    return PointerProvenance::Unresolved;
  }

  [[nodiscard]] static Value getQIRResourceAggregate(Value address) {
    if (auto call = address.getDefiningOp<LLVM::CallOp>()) {
      if (call.getCallee() && *call.getCallee() == QIR_ARRAY_ELEMENT) {
        return call.getOperand(0);
      }
      return {};
    }
    auto gep = address.getDefiningOp<LLVM::GEPOp>();
    return gep ? gep.getBase() : Value{};
  }

  static FailureOr<size_t> getNumQubits(ModuleOp scope, bool requireStatic) {
    static constexpr StringRef QIS_PREFIX = "__quantum__qis";

    size_t requiredQubits = 0;
    LogicalResult status = success();
    SmallPtrSet<Value, 8> qubitAggregates;
    SmallVector<std::pair<LLVM::StoreOp, Value>, 8> aggregateStores;
    const auto includePointer = [&](Value pointer) {
      SmallPtrSet<Value, 8> resolving;
      auto provenance = includeStaticPointer(
          pointer, "qubit", requiredQubits, scope, resolving, &qubitAggregates);
      if (failed(provenance)) {
        status = failure();
      } else if (requireStatic && *provenance != PointerProvenance::Resolved) {
        pointer.getParentBlock()->getParentOp()->emitError()
            << "cannot determine the static QIR qubit index from pointer "
               "provenance";
        status = failure();
      }
    };
    scope.walk([&](Operation* operation) {
      if (failed(status)) {
        return;
      }
      if (auto store = dyn_cast<LLVM::StoreOp>(operation);
          store && isa<LLVM::LLVMPointerType>(store.getValue().getType())) {
        if (Value aggregate = getQIRResourceAggregate(store.getAddr())) {
          aggregateStores.emplace_back(store, aggregate);
        }
        return;
      }
      auto callOp = dyn_cast<LLVM::CallOp>(operation);
      if (!callOp || !callOp.getCallee() ||
          !callOp.getCallee()->starts_with(QIS_PREFIX)) {
        return;
      }
      for (OpOperand& operand : callOp->getOpOperands()) {
        if (*callOp.getCallee() == QIR_MEASURE &&
            operand.getOperandNumber() != 0) {
          continue;
        }
        if (!isa<LLVM::LLVMPointerType>(operand.get().getType())) {
          continue;
        }
        includePointer(operand.get());
        if (failed(status)) {
          return;
        }
      }
    });
    if (failed(status)) {
      return failure();
    }

    // Follow only aggregate stores reachable from a qubit-bearing QIS operand.
    // The runtime uses opaque pointers for both qubits and results, so scanning
    // every QIR array or tuple would misclassify unrelated result aggregates.
    SmallPtrSet<Operation*, 8> processedStores;
    bool processedStore = false;
    do {
      processedStore = false;
      for (auto& [store, aggregate] : aggregateStores) {
        if (!qubitAggregates.contains(aggregate) ||
            !processedStores.insert(store.getOperation()).second) {
          continue;
        }
        processedStore = true;
        includePointer(store.getValue());
        if (failed(status)) {
          return failure();
        }
      }
    } while (processedStore);
    return requiredQubits;
  }

  /// Return the capacity required by all statically indexed result pointers.
  static FailureOr<size_t> getNumResults(ModuleOp scope, bool requireStatic) {
    size_t requiredResults = 0;
    LogicalResult status = success();
    const auto includePointer = [&](Value pointer) {
      SmallPtrSet<Value, 8> resolving;
      auto provenance = includeStaticPointer(pointer, "result", requiredResults,
                                             scope, resolving);
      if (failed(provenance)) {
        status = failure();
      } else if (requireStatic && *provenance != PointerProvenance::Resolved) {
        pointer.getParentBlock()->getParentOp()->emitError()
            << "cannot determine the static QIR result index from pointer "
               "provenance";
        status = failure();
      }
    };

    scope.walk([&](Operation* operation) {
      if (failed(status)) {
        return;
      }
      auto callOp = dyn_cast<LLVM::CallOp>(operation);
      if (!callOp) {
        return;
      }
      const auto callee = callOp.getCallee();
      if (!callee) {
        return;
      }
      if (*callee == QIR_MEASURE) {
        includePointer(callOp.getOperand(1));
      } else if (*callee == QIR_RECORD_OUTPUT || *callee == QIR_READ_RESULT) {
        includePointer(callOp.getOperand(0));
      }
    });
    if (failed(status)) {
      return failure();
    }
    return requiredResults;
  }

  /// Determine whether a loop (as a set of blocks) is an iterative loop (true)
  /// or a conditionally terminated loop (false).
  static bool classifyLoop(const SmallPtrSet<Block*, 8>& loop) {
    bool hasConditionalTermination = false;
    for (Block* block : loop) {
      auto condBrOp = dyn_cast_or_null<LLVM::CondBrOp>(block->getTerminator());
      if (!condBrOp || (loop.contains(condBrOp.getTrueDest()) &&
                        loop.contains(condBrOp.getFalseDest()))) {
        continue;
      }
      auto callOp = condBrOp.getCondition().getDefiningOp<LLVM::CallOp>();
      hasConditionalTermination |= callOp && callOp.getCallee() &&
                                   *callOp.getCallee() == QIR_READ_RESULT;
    }
    return !hasConditionalTermination;
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
          if (loop.insert(tail).second) {
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

          if (classifyLoop(loop)) {
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
  static std::tuple<bool, bool, bool> usesDynamic(Operation* scope) {
    bool useDynamicQubit{false};
    bool useDynamicResult{false};
    bool useArrays{false};

    scope->walk([&](Operation* operation) {
      auto callOp = dyn_cast<LLVM::CallOp>(operation);
      if (!callOp) {
        return;
      }
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
      } else if (name == QIR_ARRAY_CREATE || name == QIR_ARRAY_ELEMENT ||
                 name == QIR_ARRAY_RELEASE || name == QIR_ARRAY_RECORD_OUTPUT ||
                 name == QIR_RESULT_ARRAY_RECORD_OUTPUT ||
                 name == QIR_QUBIT_ARRAY_RELEASE ||
                 name == QIR_RESULT_ARRAY_RELEASE) {
        useArrays = true;
      }
    });

    return std::make_tuple(useDynamicQubit, useDynamicResult, useArrays);
  }

  static void collectOptionalFeatures(ModuleOp moduleOp,
                                      LLVM::LLVMFuncOp entryPoint,
                                      Metadata& metadata) {
    const auto recordType = [&](Type type) {
      if (const auto integer = dyn_cast<IntegerType>(type);
          integer && integer.getWidth() > 1) {
        metadata.integerTypes.insert("i" + std::to_string(integer.getWidth()));
      } else if (type.isF16()) {
        metadata.floatingTypes.insert("half");
      } else if (type.isF32()) {
        metadata.floatingTypes.insert("float");
      } else if (type.isF64()) {
        metadata.floatingTypes.insert("double");
      }
    };

    SmallVector<LLVM::LLVMFuncOp> functions;
    moduleOp.walk(
        [&](LLVM::LLVMFuncOp function) { functions.emplace_back(function); });
    for (auto function : functions) {
      if (function.isExternal()) {
        continue;
      }
      metadata.usesIRFunctions |= function != entryPoint;
      if (function != entryPoint) {
        recordType(function.getFunctionType().getReturnType());
      }
      for (Block& block : function.getBody()) {
        llvm::for_each(block.getArgumentTypes(), recordType);
      }
      size_t returnCount = 0;
      function.walk([&](Operation* operation) {
        returnCount += isa<LLVM::ReturnOp>(operation);
        metadata.usesMultipleTargetBranching |= isa<LLVM::SwitchOp>(operation);
        if (operation->hasTrait<OpTrait::ConstantLike>()) {
          return;
        }
        const auto hasScalarResult =
            llvm::any_of(operation->getResultTypes(), [](Type type) {
              return isa<IntegerType>(type) || type.isF16() || type.isF32() ||
                     type.isF64();
            });
        if (hasScalarResult && !isa<LLVM::CallOp>(operation)) {
          llvm::for_each(operation->getOperandTypes(), recordType);
        }
        llvm::for_each(operation->getResultTypes(), recordType);
      });
      metadata.usesMultipleReturnPoints |= returnCount > 1;
    }
  }

  /// Return the metadata for a QIR base profile compliant program.
  static Metadata getBase(size_t numQubits, size_t numResults) {
    return {.numQubits = numQubits,
            .numResults = numResults,
            .useDynamicQubit = false,
            .useDynamicResult = false,
            .useArrays = false,
            .backwardsBranching = 0};
  }

  /// Return the metadata for a QIR adaptive profile compliant program.
  Metadata getAdaptive(LLVM::LLVMFuncOp& main, size_t numQubits,
                       size_t numResults, bool useDynamicQubit,
                       bool useDynamicResult, bool useArrays) {
    const auto& domInfo = getAnalysis<DominanceInfo>();
    const auto [useIteration, useCondTerm] =
        usesBackwardsBranching(main, domInfo);

    Metadata md;
    md.useDynamicQubit = useDynamicQubit;
    md.useDynamicResult = useDynamicResult;
    md.useArrays = useArrays;

    if (!useDynamicQubit) {
      md.numQubits = numQubits;
    }

    if (!useDynamicResult) {
      md.numResults = numResults;
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
