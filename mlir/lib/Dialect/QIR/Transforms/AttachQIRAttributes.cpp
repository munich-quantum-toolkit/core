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

#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
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
  int backwardsBranching{0};
  llvm::SmallSet<std::string, 4> integerTypes;
  llvm::SmallSet<std::string, 4> floatingTypes;
  bool usesIRFunctions{false};
  bool usesMultipleTargetBranching{false};
  bool usesMultipleReturnPoints{false};
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

    Metadata metadata = useAdaptive ? getAdaptive(main) : Metadata{};
    if (!metadata.useDynamicQubit) {
      const auto numQubits = getNumQubits(main);
      if (failed(numQubits)) {
        signalPassFailure();
        return;
      }
      metadata.numQubits = *numQubits;
    }
    if (!metadata.useDynamicResult) {
      const auto numResults = getNumResults(main);
      if (failed(numResults)) {
        signalPassFailure();
        return;
      }
      metadata.numResults = *numResults;
    }
    if (useAdaptive) {
      collectOptionalFeatures(getOperation(), main, metadata);
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
  /// - `required_num_qubits`: Capacity through the highest static qubit ID
  /// - `required_num_results`: Capacity through the highest static result ID
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
  static bool isQubitOperand(StringRef callee, unsigned index) {
    if (callee == QIR_MEASURE || callee == QIR_RESET) {
      return index == 0;
    }
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  if (callee == QIR_##GETTER) {                                                \
    return index >= PARAMS && index < PARAMS + TARGETS;                        \
  }                                                                            \
  if (callee == QIR_C##GETTER) {                                               \
    return index >= PARAMS && index < PARAMS + TARGETS + 1;                    \
  }                                                                            \
  if (callee == QIR_CC##GETTER) {                                              \
    return index >= PARAMS && index < PARAMS + TARGETS + 2;                    \
  }                                                                            \
  if (callee == QIR_##GETTER##_CTL) {                                          \
    return PARAMS == 0 && TARGETS == 1 && index == 1;                          \
  }
#include "mqt/Conversion/GateTable.def"
    return false;
  }

  /// Extend capacity for a direct static resource pointer, ignoring dynamic
  /// IDs.
  static FailureOr<uint64_t> includeStaticResource(Value pointer,
                                                   uint64_t capacity) {
    auto toPtr = pointer.getDefiningOp<LLVM::IntToPtrOp>();
    auto constant = toPtr ? toPtr.getArg().getDefiningOp<LLVM::ConstantOp>()
                          : LLVM::ConstantOp{};
    auto index =
        constant ? dyn_cast<IntegerAttr>(constant.getValue()) : IntegerAttr{};
    return index ? includeResourceId(index, capacity, constant)
                 : FailureOr<uint64_t>(capacity);
  }

  /// Return the capacity required by scalar qubit operands of known QIS calls.
  static FailureOr<uint64_t> getNumQubits(LLVM::LLVMFuncOp& main) {
    FailureOr<uint64_t> capacity = uint64_t{0};
    main.walk([&](LLVM::CallOp call) {
      if (!call.getCallee()) {
        return;
      }
      for (auto [index, operand] : llvm::enumerate(call.getArgOperands())) {
        if (succeeded(capacity) && isQubitOperand(*call.getCallee(), index)) {
          capacity = includeStaticResource(operand, *capacity);
        }
      }
    });
    return capacity;
  }

  /// Return the capacity required to address every static result ID.
  static FailureOr<uint64_t> getNumResults(LLVM::LLVMFuncOp& main) {
    FailureOr<uint64_t> numResults = uint64_t{0};
    main->walk([&](LLVM::CallOp callOp) {
      if (failed(numResults) || !callOp.getCallee()) {
        return;
      }

      const auto callee = *callOp.getCallee();
      if (callee != QIR_RECORD_OUTPUT && callee != QIR_MEASURE &&
          callee != QIR_READ_RESULT) {
        return;
      }
      const auto index = callee == QIR_MEASURE ? 1U : 0U;
      if (callOp.getNumOperands() <= index) {
        return;
      }

      numResults =
          includeStaticResource(callOp.getArgOperands()[index], *numResults);
    });

    return numResults;
  }

  /// Determine whether a loop (as a set of blocks) is an iterative loop (true)
  /// or a conditionally terminated loop (false).
  static bool classifyLoop(const SmallPtrSet<Block*, 8>& loop) {
    for (Block* block : loop) {
      auto branch = dyn_cast<LLVM::CondBrOp>(block->getTerminator());
      if (!branch || (loop.contains(branch.getTrueDest()) &&
                      loop.contains(branch.getFalseDest()))) {
        continue;
      }
      auto call = branch.getCondition().getDefiningOp<LLVM::CallOp>();
      if (call && call.getCallee() == QIR_READ_RESULT) {
        return false;
      }
    }
    return true;
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
            loop.insert(tail);
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

  static void collectOptionalFeatures(ModuleOp moduleOp,
                                      LLVM::LLVMFuncOp entryPoint,
                                      Metadata& metadata) {
    const auto recordType = [&](const Type type) {
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

    moduleOp.walk([&](LLVM::LLVMFuncOp function) {
      if (function.isExternal()) {
        return;
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

  /// Return the dynamic resource and control flow metadata for QIR adaptive.
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
