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
#include <iterator>
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

template <typename Callback>
static void walkQIRAttributeOperationsIteratively(Operation* root,
                                                  Callback&& callback) {
  SmallVector<Operation*> worklist{root};
  while (!worklist.empty()) {
    Operation* operation = worklist.pop_back_val();
    callback(operation);
    for (Region& region : operation->getRegions()) {
      for (Block& block : region) {
        for (Operation& nested : block) {
          worklist.push_back(&nested);
        }
      }
    }
  }
}

[[nodiscard]] static bool hasQIREntryPointAttribute(LLVM::LLVMFuncOp function) {
  const auto passthrough = function->getAttrOfType<ArrayAttr>("passthrough");
  return passthrough && llvm::any_of(passthrough, [](Attribute attribute) {
           const auto name = dyn_cast<StringAttr>(attribute);
           return name && name.getValue() == StringRef(::qir::ENTRY_POINT_ATTR);
         });
}

/**
 * @brief Attaches the required attributes to the function marked as
 * entry_point.
 */
struct QIRSetAttributesAndMetadata final
    : impl::QIRSetAttributesAndMetadataBase<QIRSetAttributesAndMetadata> {
  using QIRSetAttributesAndMetadataBase::QIRSetAttributesAndMetadataBase;

protected:
  void runOnOperation() override {
    SmallVector<LLVM::LLVMFuncOp> entryPoints;
    for (auto function : getOperation().getOps<LLVM::LLVMFuncOp>()) {
      if (mqt::isEntryPoint(function) || hasQIREntryPointAttribute(function)) {
        entryPoints.push_back(function);
      }
    }
    if (entryPoints.size() != 1) {
      getOperation().emitError()
          << "QIR metadata attachment requires exactly one entry point, but "
             "found "
          << entryPoints.size();
      signalPassFailure();
      return;
    }

    auto main = entryPoints.front();
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

    const auto isQIRFunctionAttribute = [](Attribute attribute) {
      if (const auto name = dyn_cast<StringAttr>(attribute)) {
        return name.getValue() == StringRef(::qir::ENTRY_POINT_ATTR);
      }
      const auto pair = dyn_cast<ArrayAttr>(attribute);
      const auto key = pair && pair.size() == 2 ? dyn_cast<StringAttr>(pair[0])
                                                : StringAttr{};
      return key &&
             (key.getValue() == StringRef(::qir::OUTPUT_LABELING_SCHEMA_ATTR) ||
              key.getValue() == StringRef(::qir::QIR_PROFILES_ATTR) ||
              key.getValue() == "required_num_qubits" ||
              key.getValue() == "required_num_results");
    };
    SmallVector<Attribute> attributes;
    if (const auto passthrough =
            main->getAttrOfType<ArrayAttr>("passthrough")) {
      llvm::copy_if(passthrough, std::back_inserter(attributes),
                    [&](Attribute attribute) {
                      return !isQIRFunctionAttribute(attribute);
                    });
    }
    attributes.append(
        {rewriter.getStringAttr(::qir::ENTRY_POINT_ATTR),
         rewriter.getStrArrayAttr(
             {::qir::OUTPUT_LABELING_SCHEMA_ATTR, ::qir::LABELED_SCHEMA}),
         rewriter.getStrArrayAttr(
             {::qir::QIR_PROFILES_ATTR,
              useAdaptive ? ::qir::ADAPTIVE_PROFILE : ::qir::BASE_PROFILE}),
         rewriter.getStrArrayAttr(
             {"required_num_qubits", std::to_string(metadata.numQubits)}),
         rewriter.getStrArrayAttr(
             {"required_num_results", std::to_string(metadata.numResults)})});

    main->setAttr("passthrough", rewriter.getArrayAttr(attributes));
    mqt::removeEntryPoint(main);

    rewriter.setInsertionPointToEnd(m.getBody());

    SmallVector<Attribute> flags = collectUnrelatedModuleFlags(m, rewriter);
    flags.emplace_back(
        createI32Flag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2));
    flags.emplace_back(
        createI32Flag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 1));
    flags.emplace_back(createBoolFlag(LLVM::ModFlagBehavior::Error,
                                      "dynamic_qubit_management",
                                      metadata.useDynamicQubit));
    flags.emplace_back(createBoolFlag(LLVM::ModFlagBehavior::Error,
                                      "dynamic_result_management",
                                      metadata.useDynamicResult));

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

  static bool isQIRModuleFlag(StringRef key) {
    return key == "qir_major_version" || key == "qir_minor_version" ||
           key == "dynamic_qubit_management" ||
           key == "dynamic_result_management" || key == "backwards_branching" ||
           key == "arrays" || key == "ir_functions" ||
           key == "multiple_target_branching" ||
           key == "multiple_return_points" || key == "int_computations" ||
           key == "float_computations";
  }

  /// Remove existing top-level QIR module flags and return every unrelated
  /// flag unchanged.
  static SmallVector<Attribute>
  collectUnrelatedModuleFlags(ModuleOp m, IRRewriter& rewriter) {
    SmallVector<Attribute> preserved;
    for (auto flagsOp :
         llvm::make_early_inc_range(m.getOps<LLVM::ModuleFlagsOp>())) {
      for (const auto flag :
           flagsOp.getFlags().getAsRange<LLVM::ModuleFlagAttr>()) {
        if (!isQIRModuleFlag(flag.getKey().getValue())) {
          preserved.emplace_back(flag);
        }
      }
      rewriter.eraseOp(flagsOp);
    }
    return preserved;
  }

  /// Return one past the greatest indexed qubit pointer.
  /// Assumes that qubits are constant integers that are converted to
  /// an integer pointer and then used in (at least) one quantum instruction.
  static LogicalResult
  includeStaticPointer(Value pointer, StringRef resource, size_t& capacity,
                       ModuleOp module, bool requireStatic,
                       SmallPtrSetImpl<Value>& resolving,
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
      const auto index = intAttr.getValue();
      if (index.isNegative() || index.getActiveBits() >= sizeof(size_t) * 8) {
        return constOp.emitError()
               << "QIR " << resource
               << " index must be non-negative and representable as a host "
                  "size";
      }
      capacity =
          std::max(capacity, static_cast<size_t>(index.getZExtValue()) + 1);
      return success();
    }
    if (pointer.getDefiningOp<LLVM::ZeroOp>()) {
      capacity = std::max(capacity, size_t{1});
      return success();
    }
    if (auto call = pointer.getDefiningOp<LLVM::CallOp>();
        call && call.getCallee() &&
        (*call.getCallee() == QIR_ARRAY_CREATE ||
         *call.getCallee() == QIR_TUPLE_CREATE)) {
      // Generic controlled QIS calls receive aggregate pointers. Their static
      // qubit constituents are counted from the stores that populate them.
      if (aggregates) {
        aggregates->insert(pointer);
      }
      return success();
    }

    auto blockArgument = dyn_cast<BlockArgument>(pointer);
    if (blockArgument) {
      if (aggregates) {
        aggregates->insert(pointer);
      }
      Operation* anchor = blockArgument.getOwner()->getParentOp();
      if (!resolving.insert(pointer).second) {
        return anchor->emitError()
               << "cannot determine a static QIR " << resource
               << " index through recursive function arguments";
      }

      auto function = dyn_cast<LLVM::LLVMFuncOp>(anchor);
      bool sawProvenance = false;
      LogicalResult status = success();
      if (function && !function.isExternal() &&
          blockArgument.getOwner() == &function.getBody().front()) {
        walkQIRAttributeOperationsIteratively(
            module, [&](Operation* operation) {
              if (failed(status)) {
                return;
              }
              auto call = dyn_cast<LLVM::CallOp>(operation);
              if (!call || !call.getCallee() ||
                  *call.getCallee() != function.getSymName() ||
                  blockArgument.getArgNumber() >= call.getNumOperands()) {
                return;
              }
              sawProvenance = true;
              status = includeStaticPointer(
                  call.getOperand(blockArgument.getArgNumber()), resource,
                  capacity, module, requireStatic, resolving, aggregates);
            });
      } else {
        SmallVector<Value> worklist{pointer};
        SmallPtrSet<Value, 8> visited;
        bool unresolvedProvenance = false;
        while (!worklist.empty() && succeeded(status)) {
          Value current = worklist.pop_back_val();
          if (!current) {
            unresolvedProvenance = true;
            continue;
          }
          if (!visited.insert(current).second) {
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
          sawProvenance = true;
          status = includeStaticPointer(current, resource, capacity, module,
                                        requireStatic, resolving, aggregates);
        }
        sawProvenance &= !unresolvedProvenance;
      }
      resolving.erase(pointer);
      if (failed(status)) {
        return failure();
      }
      if (sawProvenance) {
        return success();
      }
    }

    if (!requireStatic) {
      return success();
    }
    Operation* anchor = pointer.getDefiningOp();
    if (!anchor) {
      anchor = cast<BlockArgument>(pointer).getOwner()->getParentOp();
    }
    return anchor->emitError() << "cannot determine the static QIR " << resource
                               << " index from pointer provenance";
  }

  [[nodiscard]] static Value getQIRResourceAggregate(Value address) {
    if (auto call = address.getDefiningOp<LLVM::CallOp>()) {
      if (call.getCallee() && *call.getCallee() == QIR_ARRAY_ELEMENT &&
          call.getNumOperands() >= 1) {
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
      status = includeStaticPointer(pointer, "qubit", requiredQubits, scope,
                                    requireStatic, resolving, &qubitAggregates);
    };
    walkQIRAttributeOperationsIteratively(scope, [&](Operation* operation) {
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
      status = includeStaticPointer(pointer, "result", requiredResults, scope,
                                    requireStatic, resolving);
    };

    walkQIRAttributeOperationsIteratively(scope, [&](Operation* operation) {
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
      if (*callee == QIR_MEASURE && callOp.getNumOperands() >= 2) {
        includePointer(callOp.getOperand(1));
      } else if ((*callee == QIR_RECORD_OUTPUT || *callee == QIR_READ_RESULT) &&
                 callOp.getNumOperands() >= 1) {
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

    walkQIRAttributeOperationsIteratively(scope, [&](Operation* operation) {
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
    walkQIRAttributeOperationsIteratively(moduleOp, [&](Operation* operation) {
      if (auto function = dyn_cast<LLVM::LLVMFuncOp>(operation)) {
        functions.emplace_back(function);
      }
    });
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
      walkQIRAttributeOperationsIteratively(
          function, [&](Operation* operation) {
            returnCount += isa<LLVM::ReturnOp>(operation);
            metadata.usesMultipleTargetBranching |=
                isa<LLVM::SwitchOp>(operation);
            if (operation->hasTrait<OpTrait::ConstantLike>()) {
              return;
            }
            const auto hasScalarResult =
                llvm::any_of(operation->getResultTypes(), [](Type type) {
                  return isa<IntegerType>(type) || type.isF16() ||
                         type.isF32() || type.isF64();
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
