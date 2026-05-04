#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <mlir/IR/Operation.h>
#include <mlir/IR/OperationSupport.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::qco {

#define GEN_PASS_DEF_CANCELCONSECUTIVEHERMITIANGATES
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct CancelConsecutiveHermitianGatesPattern final
    : OpInterfaceRewritePattern<UnitaryOpInterface> {
  explicit CancelConsecutiveHermitianGatesPattern(MLIRContext* context)
      : OpInterfaceRewritePattern(context) {}

  /**
   * @returns true, if the i-th output of @p op is the i-th input of @p other.
   */
  static bool haveSameOrientation(UnitaryOpInterface op,
                                  UnitaryOpInterface other) {
    return llvm::all_of(
        llvm::zip_equal(op.getOutputQubits(), other.getInputQubits()),
        [](const auto& pair) {
          const auto& [out, in] = pair;
          return out == in;
        });
  }

  LogicalResult matchAndRewrite(UnitaryOpInterface op,
                                PatternRewriter& rewriter) const override {
    if (op->use_empty()) {
      return failure();
    }

    if (!llvm::all_equal(op->getUsers())) {
      return failure();
    }

    auto other = llvm::dyn_cast<UnitaryOpInterface>(*(op->getUsers().begin()));
    if (other == nullptr) {
      return failure();
    }

    if (!op->hasTrait<HermitianTrait>() || !other->hasTrait<HermitianTrait>()) {
      return failure();
    }

    if (op->getName() != other->getName()) {
      return failure();
    }

    if (op.getNumQubits() != other.getNumQubits()) {
      return failure();
    }

    if (!haveSameOrientation(op, other)) {
      return failure();
    }

    rewriter.replaceOp(other, op.getInputQubits());
    rewriter.eraseOp(op);

    return success();
  }
};
} // namespace

/**
 * @brief Pass that cancels consecutive hermitian gates.
 */
struct CancelConsecutiveHermitianGates final
    : impl::CancelConsecutiveHermitianGatesBase<
          CancelConsecutiveHermitianGates> {
  using CancelConsecutiveHermitianGatesBase<
      CancelConsecutiveHermitianGates>::CancelConsecutiveHermitianGatesBase;

protected:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<CancelConsecutiveHermitianGatesPattern>(patterns.getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace mlir::qco