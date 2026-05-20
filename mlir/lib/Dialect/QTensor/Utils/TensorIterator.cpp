#include "mlir/Dialect/QTensor/Utils/TensorIterator.h"

#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/TypeSwitch.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/Value.h>

#include <iterator>
namespace mlir::qtensor {
TypedValue<RankedTensorType> TensorIterator::tensor() const {
  // A tensor deallocation doesn't have an OpResult.
  if (isa<DeallocOp>(op_)) {
    return nullptr;
  }
  return tensor_;
}

void TensorIterator::forward() {
  // If the iterator is a sentinel already, there is nothing to do.
  if (isSentinel_) {
    return;
  }

  // Find the user-operation of the tensor SSA value.
  assert(tensor_.hasOneUse() && "expected linear typing");
  op_ = *(tensor_.user_begin());

  // A deallocation defines the end of the tensor's life-chain.
  if (isa<DeallocOp, scf::YieldOp>(op_)) {
    isSentinel_ = true;
    return;
  }

  // Find the output from the input tensor SSA value.
  if (!(isa<AllocOp, FromElementsOp>(op_))) {
    TypeSwitch<Operation*>(op_)
        .Case<ExtractOp>([&](ExtractOp op) { tensor_ = op.getOutTensor(); })
        .Case<InsertOp>([&](InsertOp op) { tensor_ = op.getResult(); })
        .Case<scf::ForOp>([&](scf::ForOp op) {
          tensor_ = cast<TypedValue<RankedTensorType>>(
              op.getTiedLoopResult(&*(tensor_.use_begin())));
        })
        .Default([&](Operation* op) {
          report_fatal_error("unknown op in def-use chain: " +
                             op->getName().getStringRef());
        });
  }
}

void TensorIterator::backward() {
  // If the iterator is a sentinel, reactivate the iterator.
  if (isSentinel_) {
    isSentinel_ = false;
    return;
  }

  // For deallocations and scf::YieldOps, tensor_ is an OpOperand.
  // Hence, only get the def-op.
  if (isa<DeallocOp, scf::YieldOp>(op_)) {
    op_ = tensor_.getDefiningOp();
    return;
  }

  // Allocations and FromElements define the start of the tensor's life-chain.
  // Consequently, stop and early exit.
  if (isa<AllocOp, FromElementsOp>(op_)) {
    return;
  }

  // Find the input from the output tensor SSA value.
  TypeSwitch<Operation*>(op_)
      .Case<ExtractOp>([&](ExtractOp op) { tensor_ = op.getTensor(); })
      .Case<InsertOp>([&](InsertOp op) { tensor_ = op.getDest(); })
      .Case<scf::ForOp>([&](scf::ForOp op) {
        if (auto res = dyn_cast<OpResult>(tensor_)) {
          OpOperand* operand = op.getTiedLoopInit(res);
          tensor_ = cast<TypedValue<RankedTensorType>>(operand->get());
          return;
        }

        llvm::report_fatal_error(
            "expected scf.for result for tied init lookup");
      })
      .Default([&](Operation* op) {
        report_fatal_error("unknown op in def-use chain: " +
                           op->getName().getStringRef());
      });

  // Get the operation that produces the tensor value.
  // If the current tensor SSA value is a BlockArgument (no defining op), the
  // operation will be a nullptr.
  op_ = tensor_.getDefiningOp();
}

static_assert(std::bidirectional_iterator<TensorIterator>);
static_assert(std::sentinel_for<std::default_sentinel_t, TensorIterator>,
              "std::default_sentinel_t must be a sentinel for TensorIterator.");
} // namespace mlir::qtensor