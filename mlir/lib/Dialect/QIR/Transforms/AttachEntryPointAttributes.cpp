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
protected:
  void runOnOperation() override {
    auto main = getMainFunction(getOperation());
    setQIRAttributes(main, collectMetadata());
  }

private:
  QIRMetadata collectMetadata(LLVM::LLVMFuncOp& main) {
    QIRMetadata metadata;

    if (!useAdaptive) {
      metadata.useAdaptive = false;
      metadata.useDynamicQubit = false;
      metadata.useDynamicResult = false;
      metadata.backwardsBranching = 0;
      metadata.useArrays = false;
      metadata.numQubits = getNumConstantQubits(main);
    }
    
    return metadata;
  }

  static size_t getNumConstantQubits(LLVM::LLVMFuncOp& main) {
    static constexpr StringRef PREFIX_LABEL = "@__quantum__qis";

    size_t numQubits{0};
    DenseSet<APInt> seen; // A set of seen indices.

    std::ignore = main->walk([&](LLVM::ConstantOp& constant) {
      // Must be used and an integer const.
      if (constant.use_empty() || !isa<IntegerAttr>(constant.getValue())) {
        return WalkResult::advance();
      }

      auto intAttr = cast<IntegerAttr>(constant.getValue());

      // Must be not be an index integer.
      if (!intAttr.getType().isInteger()) {
        return WalkResult::advance();
      }

      // We assume that static qubits are constant integers, that are converted
      // to an integer pointer, and then used in one quantum instruction.

      auto userIt = llvm::find_if(constant->getUsers(), [](Operation* user) {
        return isa<LLVM::IntToPtrOp>(user);
      });
      if (userIt == constant->user_end()) {
        return WalkResult::advance();
      }

      auto toPtrOp = cast<LLVM::IntToPtrOp>(*userIt);
      auto callIt = llvm::find_if(toPtrOp->getUsers(), [](Operation* user) {
        auto callOp = dyn_cast<LLVM::CallOp>(user);
        if (!callOp) {
          return false;
        }

        const auto funcName = callOp->getName().getStringRef();
        return funcName.starts_with(PREFIX_LABEL);
      });

      if (callIt == toPtrOp->user_end() || seen.contains(intAttr.getValue())) {
        return WalkResult::advance();
      }

      ++numQubits;
    });
  }
};

} // namespace mlir::qir