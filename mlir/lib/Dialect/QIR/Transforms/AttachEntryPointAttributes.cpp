#include "mlir/Dialect/QIR/Transforms/Passes.h"
#include "mlir/Dialect/QIR/Utils/QIRMetadata.h"
#include "mlir/Dialect/QIR/Utils/QIRUtils.h"

#include <llvm/ADT/APInt.h>
#include <llvm/ADT/STLExtras.h>
#include <llvm/ADT/StringRef.h>
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

#include <cstddef>
#include <tuple>

namespace mlir::qir {

#define GEN_PASS_DEF_ATTACHENTRYPOINTATTRIBUTES
#include "mlir/Dialect/QIR/Transforms/Passes.h.inc"

/**
 * @brief Attaches the required attributes to the function marked as
 * entry_point.
 */
struct AttachEntryPointAttributes final
    : impl::AttachEntryPointAttributesBase<AttachEntryPointAttributes> {
  using AttachEntryPointAttributesBase::AttachEntryPointAttributesBase;

protected:
  void runOnOperation() override {
    auto main = getMainFunction(getOperation());
    setQIRAttributes(main, useAdaptive ? getBase(main) : getAdaptive(main));
  }

private:
  /// Count the number of uniquely indexed qubit pointers.
  /// Assumes that qubits are constant integers that are converted to
  /// an integer pointer and then used in (at least) one quantum instruction.
  static size_t getNumQubits(LLVM::LLVMFuncOp& main) {
    static constexpr StringRef PREFIX_LABEL = "@__quantum__qis";

    DenseSet<APInt> seen;
    main->walk([&](LLVM::ConstantOp& constOp) {
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
          llvm::find_if(toPtrOp->getUsers(), [](Operation* user) {
            auto callOp = dyn_cast<LLVM::CallOp>(user);
            if (!callOp) {
              return false;
            }

            const auto funcName = callOp->getName().getStringRef();
            return funcName.starts_with(PREFIX_LABEL);
          });

      if (callIt == toPtrOp->user_end()) {
        return;
      }

      // The set ensures that we don't insert the same index multiple times.
      seen.insert(intAttr.getValue());
    });

    return seen.size();
  }

  /// Count the number of uniquely indexed result_record_output statements.
  static size_t getNumResults(LLVM::LLVMFuncOp& main) {
    static constexpr StringRef REC_FN = "@__quantum__rt__result_record_output";

    DenseSet<APInt> seen;
    main->walk([&](LLVM::CallOp& callOp) {
      if (callOp->getName().getStringRef() != REC_FN) {
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

  /// Return true, if the entry point contains an `LLVM::CondBrOp`.
  static bool hasConditionalBranching(LLVM::LLVMFuncOp& main) {
    bool hasConditional{false};
    main->walk([&](LLVM::CondBrOp&) {
      hasConditional = true;
      return WalkResult::interrupt();
    });
    return hasConditional;
  }

  /// Return true, if the entry point contains an `LLVM::BrOp` for which the
  /// destination dominates the block it terminates.
  bool hasBackwardBranching(LLVM::LLVMFuncOp& main) {
    bool hasBackward{false};
    const auto& domInfo = getAnalysis<DominanceInfo>();
    main->walk([&](LLVM::BrOp& brOp) {
      if (domInfo.dominates(brOp.getDest(), brOp->getBlock())) {
        hasBackward = true;
        return WalkResult::interrupt();
      }
    });
    return hasBackward;
  }

  ///
  static std::tuple<bool, bool, bool>
  hasDynamicQubitAllocation(LLVM::LLVMFuncOp& main) {
    static constexpr StringRef QUBIT_ALLOC = "@__quantum__rt__qubit_allocate";
    static constexpr StringRef RESULT_ALLOC = "@__quantum__rt__result_allocate";
    static constexpr StringRef QUBIT_ARR_ALLOC =
        "@__quantum__rt__qubit_array_allocate";
    static constexpr StringRef RESULT_ARR_ALLOC =
        "@__quantum__rt__result_array_allocate";

    bool useDynamicQubit{false};
    bool useDynamicResult{false};
    bool useArrays{false};

    main->walk([&](LLVM::CallOp& callOp) {
      const auto name = callOp->getName().getStringRef();
      if (name == QUBIT_ALLOC) {
        useDynamicQubit = true;
      } else if (name == RESULT_ALLOC) {
        useDynamicResult = true;
      } else if (name == QUBIT_ARR_ALLOC) {
        useDynamicQubit = true;
        useArrays = true;
      } else if (name == RESULT_ARR_ALLOC) {
        useDynamicResult = true;
        useArrays = true;
      }
    });

    return std::make_tuple(useDynamicQubit, useDynamicResult, useArrays);
  }

  /// Return the metadata for a QIR base profile compliant program.
  static QIRMetadata getBase(LLVM::LLVMFuncOp& main) {
    return {.useAdaptive = false,
            .useDynamicQubit = false,
            .useDynamicResult = false,
            .backwardsBranching = 0,
            .useArrays = false,
            .numQubits = getNumQubits(main),
            .numResults = getNumResults(main)};
  }

  /// Return the metadata for a QIR base profile compliant program.
  QIRMetadata getAdaptive(LLVM::LLVMFuncOp& main) {
    const auto hasConditional = hasConditionalBranching(main);
    const auto hasBackward = hasBackwardBranching(main);
    const auto [useDynamicQubit, useDynamicResult, useArrays] =
        hasDynamicQubitAllocation(main);

    QIRMetadata md;
    md.useAdaptive = true;
    md.useDynamicQubit = useDynamicQubit;
    md.useDynamicResult = useDynamicResult;
    md.useArrays = useArrays;

    if (!useDynamicQubit) {
      md.numQubits = getNumQubits(main);
    }

    if (!useDynamicResult) {
      md.numResults = getNumResults(main);
    }

    if (hasConditional) {
      md.backwardsBranching = hasBackward ? 3 : 2;
    } else if (hasBackward) {
      md.backwardsBranching = 1;
    }

    return md;
  }
};

} // namespace mlir::qir