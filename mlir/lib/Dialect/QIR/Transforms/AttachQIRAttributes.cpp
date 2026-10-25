/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mqt/Dialect/QIR/QIRDefinitions.h"
#include "mqt/Dialect/QIR/Transforms/Passes.h"
#include "mqt/Dialect/QIR/Utils/QIRUtils.h"

#include "mlir/Analysis/CFGLoopInfo.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BlockSupport.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CFG.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace mlir::qir {
#define GEN_PASS_DEF_QIRSETATTRIBUTESANDMETADATA
#include "mqt/Dialect/QIR/Transforms/Passes.h.inc"

namespace {

/// State object for tracking QIR metadata during conversion
struct Metadata {
  /// Required capacity for static qubit IDs
  uint64_t numQubits{0};
  /// Required capacity for static result IDs
  uint64_t numResults{0};
  /// Whether the module uses dynamic qubit management
  bool useDynamicQubit{false};
  /// Whether the module uses dynamic result management
  bool useDynamicResult{false};
  /// Whether the module uses arrays
  bool useArrays{false};
  /// Whether the module uses backward branching (0 = none, 1 = iteration based,
  /// 2 = condition based, 3 = both)
  uint32_t backwardsBranching{0};
  llvm::SmallSet<std::string, 4> integerTypes;
  llvm::SmallSet<std::string, 4> floatingTypes;
  bool usesIRFunctions{false};
  bool usesMultipleTargetBranching{false};
  bool usesMultipleReturnPoints{false};
};

/// Index only calls to defined helpers; external QIS calls need no caller list.
struct HelperCalls {
  SymbolTableCollection symbols;
  DenseMap<Operation*, SmallVector<LLVM::CallOp>> callers;

  explicit HelperCalls(ModuleOp moduleOp) {
    moduleOp.walk([&](LLVM::CallOp call) {
      if (auto function = lookup(call); function && !function.isExternal()) {
        callers[function].push_back(call);
      }
    });
  }

  LLVM::LLVMFuncOp lookup(LLVM::CallOp call) {
    return call.getCallee()
               ? symbols.lookupSymbolIn<LLVM::LLVMFuncOp>(
                     call->getParentOfType<ModuleOp>(), call.getCalleeAttr())
               : LLVM::LLVMFuncOp{};
  }
};

/// Attaches the required attributes to the function marked as
/// entry_point.
struct QIRSetAttributesAndMetadata final
    : impl::QIRSetAttributesAndMetadataBase<QIRSetAttributesAndMetadata> {
  using QIRSetAttributesAndMetadataBase::QIRSetAttributesAndMetadataBase;

protected:
  void runOnOperation() override {
    auto main = getMainFunction(getOperation());
    if (!main) {
      getOperation().emitError(
          "QIR metadata attachment requires exactly one entry point");
      signalPassFailure();
      return;
    }

    HelperCalls helpers(getOperation());
    Metadata metadata;
    if (useAdaptive) {
      collectAdaptiveFeatures(main, helpers, metadata);
    }
    if (failed(collectResourceCapacities(getOperation(), helpers, metadata))) {
      signalPassFailure();
      return;
    }
    IRRewriter rewriter(&getContext());
    setMetadata(main, metadata, rewriter);
  }

private:
  /// Replace QIR profile metadata while preserving unrelated attributes.
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

    const auto isQIRFunctionMetadata = [](Attribute attribute) {
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
    if (const auto passthrough = main.getPassthroughAttr()) {
      attributes.append(passthrough.begin(), passthrough.end());
    }
    llvm::erase_if(attributes, isQIRFunctionMetadata);
    attributes.append({
        rewriter.getStrArrayAttr(
            {::qir::OUTPUT_LABELING_SCHEMA_ATTR, ::qir::LABELED_SCHEMA}),
        rewriter.getStrArrayAttr({
            ::qir::QIR_PROFILES_ATTR,
            useAdaptive ? ::qir::ADAPTIVE_PROFILE : ::qir::BASE_PROFILE,
        }),
        rewriter.getStrArrayAttr(
            {"required_num_qubits", std::to_string(metadata.numQubits)}),
        rewriter.getStrArrayAttr(
            {"required_num_results", std::to_string(metadata.numResults)}),
    });

    main.setPassthroughAttr(rewriter.getArrayAttr(attributes));

    rewriter.setInsertionPointToEnd(m.getBody());

    SmallVector<Attribute> flags = collectUnrelatedModuleFlags(m, rewriter);
    flags.append({
        createI32Flag(LLVM::ModFlagBehavior::Error, "qir_major_version", 2),
        createI32Flag(LLVM::ModFlagBehavior::Max, "qir_minor_version", 1),
        createBoolFlag(LLVM::ModFlagBehavior::Error, "dynamic_qubit_management",
                       metadata.useDynamicQubit),
        createBoolFlag(LLVM::ModFlagBehavior::Error,
                       "dynamic_result_management", metadata.useDynamicResult),
    });

    if (useAdaptive) {
      flags.emplace_back(
          createI32Flag(LLVM::ModFlagBehavior::Error, "backwards_branching",
                        static_cast<int32_t>(metadata.backwardsBranching)));
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
        return;
      }
      SmallVector<StringRef> values(types.begin(), types.end());
      llvm::sort(values);
      flags.emplace_back(createFlag(LLVM::ModFlagBehavior::Append, name,
                                    rewriter.getStrArrayAttr(values)));
    };
    setTypes("int_computations", metadata.integerTypes);
    setTypes("float_computations", metadata.floatingTypes);
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
  collectUnrelatedModuleFlags(ModuleOp moduleOp, IRRewriter& rewriter) {
    SmallVector<Attribute> preserved;
    for (auto flagsOp :
         llvm::make_early_inc_range(moduleOp.getOps<LLVM::ModuleFlagsOp>())) {
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

  /// Extend the resource capacity to include an ID without overflow.
  static FailureOr<uint64_t> includeResourceId(IntegerAttr index,
                                               uint64_t capacity,
                                               Operation* operation) {
    const auto& value = index.getValue();
    if (value.getActiveBits() > 64 ||
        value.getZExtValue() == std::numeric_limits<uint64_t>::max()) {
      return operation->emitError("static QIR resource ID requires a capacity "
                                  "that does not fit in 64 bits");
    }
    return std::max(capacity, value.getZExtValue() + 1);
  }

  /// Identify scalar qubit operands of the QIS calls emitted by this compiler.
  static bool isQubitOperand(StringRef callee, size_t index) {
    if (callee == QIR_MEASURE || callee == QIR_RESET) {
      return index == 0;
    }
    if (callee == "__quantum__qis__cnot__body") {
      return index < 2;
    }
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  if (callee == QIR_##GETTER) {                                                \
    return index >= (PARAMS) && index < (PARAMS) + (TARGETS);                  \
  }                                                                            \
  if (callee == QIR_C##GETTER) {                                               \
    return index >= (PARAMS) && index < (PARAMS) + (TARGETS) + 1;              \
  }                                                                            \
  if (callee == QIR_CC##GETTER) {                                              \
    return index >= (PARAMS) && index < (PARAMS) + (TARGETS) + 2;              \
  }                                                                            \
  if (callee == QIR_##GETTER##_CTL) {                                          \
    return (PARAMS) == 0 && (TARGETS) == 1 && index == 1;                      \
  }
#include "mqt/Conversion/GateTable.def"
    return false;
  }

  /// Extend capacity for a direct static resource pointer, ignoring dynamic
  /// IDs.
  static FailureOr<uint64_t> includeStaticResource(Value pointer,
                                                   uint64_t capacity) {
    if (pointer.getDefiningOp<LLVM::ZeroOp>()) {
      return std::max(capacity, uint64_t{1});
    }
    auto toPtr = pointer.getDefiningOp<LLVM::IntToPtrOp>();
    auto constant = toPtr ? toPtr.getArg().getDefiningOp<LLVM::ConstantOp>()
                          : LLVM::ConstantOp{};
    auto index =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return index ? includeResourceId(index, capacity, constant)
                 : FailureOr<uint64_t>(capacity);
  }

  /// Visit incoming values and report whether any were found.
  static bool visitIncoming(BlockArgument argument, const HelperCalls& helpers,
                            function_ref<void(Value)> visit) {
    auto* block = argument.getOwner();
    const auto index = argument.getArgNumber();
    bool found = false;
    const auto record = [&](Value value) {
      found = true;
      visit(value);
    };
    if (block->isEntryBlock()) {
      if (auto calls = helpers.callers.find(block->getParentOp());
          calls != helpers.callers.end()) {
        for (auto call : calls->second) {
          record(call.getArgOperands()[index]);
        }
      }
      return found;
    }
    for (auto& edge : block->getUses()) {
      if (auto branch = dyn_cast<BranchOpInterface>(edge.getOwner())) {
        auto operands = branch.getSuccessorOperands(edge.getOperandNumber());
        if (index < operands.size() && !operands.isOperandProduced(index)) {
          record(operands[index]);
        }
      }
    }
    return found;
  }

  /// Count static resource operands, including arguments of retained helpers.
  static FailureOr<uint64_t> getResourceCapacity(SetVector<Value> values,
                                                 HelperCalls& helpers) {
    uint64_t capacity = 0;
    DenseSet<Operation*> visitedCallees;
    const auto enqueue = [&](Value value) { values.insert(value); };
    size_t nextValue = 0;
    while (nextValue < values.size()) {
      auto value = values[nextValue++];
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        visitIncoming(argument, helpers, enqueue);
      } else if (auto select = value.getDefiningOp<LLVM::SelectOp>()) {
        enqueue(select.getTrueValue());
        enqueue(select.getFalseValue());
      } else if (auto call = value.getDefiningOp<LLVM::CallOp>()) {
        if (auto function = helpers.lookup(call);
            function && visitedCallees.insert(function).second) {
          function.walk([&](LLVM::ReturnOp result) {
            llvm::for_each(result.getOperands(), enqueue);
          });
        }
      } else {
        auto next = includeStaticResource(value, capacity);
        if (failed(next)) {
          return failure();
        }
        capacity = *next;
      }
    }
    return capacity;
  }

  /// Count static resources used by QIS calls and compiler-marked stores.
  static LogicalResult collectResourceCapacities(ModuleOp moduleOp,
                                                 HelperCalls& helpers,
                                                 Metadata& metadata) {
    if (metadata.useDynamicQubit && metadata.useDynamicResult) {
      return success();
    }
    SetVector<Value> qubits;
    SetVector<Value> results;
    moduleOp.walk([&](Operation* operation) {
      if (auto store = dyn_cast<LLVM::StoreOp>(operation);
          store && !metadata.useDynamicQubit &&
          store->hasAttr(QIR_QUBIT_STORE_ATTR)) {
        qubits.insert(store.getValue());
      }
      auto call = dyn_cast<LLVM::CallOp>(operation);
      if (!call || !call.getCallee()) {
        return;
      }
      if (!metadata.useDynamicQubit) {
        for (auto [index, operand] : llvm::enumerate(call.getArgOperands())) {
          if (isQubitOperand(*call.getCallee(), index)) {
            qubits.insert(operand);
          }
        }
      }
      if (metadata.useDynamicResult) {
        return;
      }
      const auto callee = call.getCallee();
      if ((callee == QIR_RECORD_OUTPUT || callee == QIR_READ_RESULT) &&
          !call.getArgOperands().empty()) {
        results.insert(call.getArgOperands()[0]);
      } else if (callee == QIR_MEASURE && call.getArgOperands().size() > 1) {
        results.insert(call.getArgOperands()[1]);
      }
    });
    const auto numQubits = getResourceCapacity(std::move(qubits), helpers);
    const auto numResults = getResourceCapacity(std::move(results), helpers);
    if (failed(numQubits) || failed(numResults)) {
      return failure();
    }
    metadata.numQubits = *numQubits;
    metadata.numResults = *numResults;
    return success();
  }

  /// Classify exit dependencies as iterations, feedback, or either (1, 2, 3).
  static uint32_t classifyLoop(const CFGLoop& loop,
                               const HelperCalls& helpers) {
    SetVector<Value> values;
    bool unknown = false;
    for (Block* block : loop.blocks()) {
      auto* terminator = block->getTerminator();
      if (!loop.isLoopLatch(block) &&
          llvm::all_of(terminator->getSuccessors(), [&](Block* successor) {
            return loop.contains(successor);
          })) {
        continue;
      }
      if (auto branch = dyn_cast<LLVM::CondBrOp>(terminator)) {
        values.insert(branch.getCondition());
      } else if (auto branch = dyn_cast<LLVM::SwitchOp>(terminator)) {
        values.insert(branch.getValue());
      } else if (!isa<LLVM::BrOp>(terminator)) {
        unknown = true;
      }
    }
    // shortcut: unknown exits need both flags until dependence is proved.
    unknown |= values.empty();
    const auto enqueue = [&](Value value) { values.insert(value); };
    size_t nextValue = 0;
    while (nextValue < values.size()) {
      auto value = values[nextValue++];
      if (auto argument = dyn_cast<BlockArgument>(value)) {
        unknown |= !visitIncoming(argument, helpers, enqueue);
        continue;
      }
      auto* operation = value.getDefiningOp();
      if (auto call = dyn_cast<LLVM::CallOp>(operation);
          call && call.getCallee() == QIR_READ_RESULT) {
        return 2;
      }
      if (!isMemoryEffectFree(operation) || isa<LLVM::CallOp>(operation)) {
        unknown = true;
        continue;
      }
      llvm::for_each(operation->getOperands(), enqueue);
    }
    return unknown ? 3 : 1;
  }

  /// Collect loop capability bits for one function.
  static uint32_t usesBackwardsBranching(LLVM::LLVMFuncOp function,
                                         const DominanceInfo& domInfo,
                                         const HelperCalls& helpers) {
    if (function.getBody().hasOneBlock()) {
      return 0;
    }
    CFGLoopInfo loops(domInfo.getDomTree(&function.getBody()));
    llvm::ReversePostOrderTraversal<Block*> traversal(
        &function.getBody().front());
    if (llvm::containsIrreducibleCFG<Block*>(traversal, loops)) {
      return 3;
    }
    uint32_t capabilities = 0;
    for (auto* loop : loops.getLoopsInPreorder()) {
      capabilities |= classifyLoop(*loop, helpers);
    }
    return capabilities;
  }

  /// Collect capabilities from all emitted function definitions.
  void collectAdaptiveFeatures(LLVM::LLVMFuncOp entryPoint,
                               const HelperCalls& helpers, Metadata& metadata) {
    const auto recordType = [&](Type type) {
      while (auto array = dyn_cast<LLVM::LLVMArrayType>(type)) {
        type = array.getElementType();
      }
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

    const auto& domInfo = getAnalysis<DominanceInfo>();
    getOperation().walk([&](LLVM::LLVMFuncOp function) {
      if (function.isExternal()) {
        return;
      }
      metadata.usesIRFunctions |= function != entryPoint;
      metadata.backwardsBranching |=
          usesBackwardsBranching(function, domInfo, helpers);
      if (function != entryPoint) {
        recordType(function.getFunctionType().getReturnType());
      }
      for (Block& block : function.getBody()) {
        llvm::for_each(block.getArgumentTypes(), recordType);
      }
      size_t returnCount = 0;
      function.walk([&](Operation* operation) {
        if (auto call = dyn_cast<LLVM::CallOp>(operation)) {
          const auto name = call.getCallee();
          metadata.useDynamicQubit |=
              name == QIR_QUBIT_ALLOC || name == QIR_QUBIT_ARRAY_ALLOC;
          metadata.useDynamicResult |=
              name == QIR_RESULT_ALLOC || name == QIR_RESULT_ARRAY_ALLOC;
          metadata.useArrays |=
              name == QIR_QUBIT_ARRAY_ALLOC || name == QIR_RESULT_ARRAY_ALLOC;
        }
        returnCount += isa<LLVM::ReturnOp>(operation);
        metadata.usesMultipleTargetBranching |= isa<LLVM::SwitchOp>(operation);
        metadata.useArrays |=
            isa<LLVM::AllocaOp, LLVM::LoadOp, LLVM::StoreOp,
                LLVM::ExtractValueOp, LLVM::InsertValueOp>(operation);
        if (auto store = dyn_cast<LLVM::StoreOp>(operation)) {
          recordType(store.getValue().getType());
        }
        if (auto gep = dyn_cast<LLVM::GEPOp>(operation)) {
          const auto dynamic =
              llvm::any_of(gep.getDynamicIndices(), [](Value index) {
                return !matchPattern(index, m_Constant());
              });
          const auto label =
              !dynamic &&
              llvm::all_of(gep.getResult().getUses(), [](OpOperand& use) {
                auto call = dyn_cast<LLVM::CallOp>(use.getOwner());
                return call && call.getCallee() &&
                       call.getCallee()->starts_with("__quantum__rt__") &&
                       call.getCallee()->ends_with("_record_output") &&
                       use.getOperandNumber() + 1 == call.getNumOperands();
              });
          metadata.useArrays |= !label;
        }
        if (operation->hasTrait<OpTrait::ConstantLike>()) {
          return;
        }
        const auto hasScalarResult =
            llvm::any_of(operation->getResultTypes(), [](const Type type) {
              return isa<IntegerType>(type) || type.isF16() || type.isF32() ||
                     type.isF64();
            });
        if (hasScalarResult && !isa<LLVM::CallOp>(operation)) {
          llvm::for_each(operation->getOperandTypes(), recordType);
        }
        llvm::for_each(operation->getResultTypes(), recordType);
      });
      metadata.usesMultipleReturnPoints |= returnCount > 1;
    });
  }
};
} // namespace
} // namespace mlir::qir
