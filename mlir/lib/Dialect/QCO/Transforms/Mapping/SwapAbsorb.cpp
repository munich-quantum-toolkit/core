#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QCO/IR/QCOInterfaces.h"
#include "mlir/Dialect/QCO/IR/QCOOps.h"
#include "mlir/Dialect/QCO/Transforms/Passes.h"
#include "mlir/Dialect/QCO/Utils/Drivers.h"
#include "mlir/Dialect/QCO/Utils/WireIterator.h"
#include "mlir/Dialect/QTensor/IR/QTensorOps.h"

#include <llvm/ADT/STLExtras.h>
#include <llvm/Support/Debug.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/IR/Visitors.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Support/WalkResult.h>

namespace mlir::qco {
#define GEN_PASS_DEF_SWAPABSORB
#include "mlir/Dialect/QCO/Transforms/Passes.h.inc"

namespace {
struct SwapAbsorb : impl::SwapAbsorbBase<SwapAbsorb> {
public:
  using SwapAbsorbBase::SwapAbsorbBase;

  void runOnOperation() override {
    ModuleOp anchor = getOperation();
    IRRewriter rewriter(&getContext());
    insertStatics(anchor, rewriter);

    for (auto func : anchor.getOps<func::FuncOp>()) {
      SmallVector<WireIterator> wires;
      for (auto op : func.getOps<StaticOp>()) {
        wires.emplace_back(op.getQubit());
      }

      SmallVector<SWAPOp> readyToAbsorb;
      readyToAbsorb.reserve((wires.size() + 1) / 2);

      std::ignore =
          walkCircuitGraph(wires, WalkDirection::Forward,
                           [&](const ReadyRange& ready, ReleasedOps& released) {
                             for (const auto& [op, indices] : ready) {
                               if (isa<SWAPOp>(op)) {
                                 readyToAbsorb.emplace_back(op);
                               }
                               released.emplace_back(op);
                             }
                             return WalkResult::interrupt();
                           });

      for (auto swapOp : readyToAbsorb) {
        auto in0 = swapOp.getQubit0In();
        auto in1 = swapOp.getQubit1In();

        auto out0 = swapOp.getQubit0Out();
        auto out1 = swapOp.getQubit1Out();

        // TODO: What if single qubit gates are in front of the SWAP input?
        // Tipp: Use the WireIterator.
        StaticOp op0 = cast<StaticOp>(in0.getDefiningOp());
        StaticOp op1 = cast<StaticOp>(in1.getDefiningOp());

        rewriter.replaceAllUsesWith(out0, op1.getQubit());
        rewriter.replaceAllUsesWith(out1, op0.getQubit());
        rewriter.eraseOp(swapOp);
      }
    }
  }

private:
  static void insertStatics(ModuleOp anchor, IRRewriter& rewriter) {
    for (auto func : anchor.getOps<func::FuncOp>()) {
      SmallVector<Operation*> worklist;
      for (Operation& op : func.getOps()) {
        worklist.emplace_back(&op);
      }

      std::size_t n = llvm::range_size(func.getOps<qtensor::ExtractOp>()) - 1;
      for (Operation* op : llvm::reverse(worklist)) {
        rewriter.setInsertionPoint(op);

        if (auto tensorDealloc = dyn_cast<qtensor::DeallocOp>(op)) {
          rewriter.eraseOp(tensorDealloc);
          continue;
        }

        if (auto tensorInsert = dyn_cast<qtensor::InsertOp>(op)) {
          auto q = tensorInsert.getScalar();
          rewriter.create<qco::SinkOp>(rewriter.getUnknownLoc(), q);
          rewriter.eraseOp(tensorInsert);
          continue;
        }

        if (auto tensorExtract = dyn_cast<qtensor::ExtractOp>(op)) {
          auto q = tensorExtract.getResult();
          auto staticOp =
              rewriter.create<qco::StaticOp>(rewriter.getUnknownLoc(), n);
          rewriter.replaceAllUsesWith(q, staticOp.getQubit());
          rewriter.eraseOp(tensorExtract);
          n--;
          continue;
        }

        if (auto tensorAlloc = dyn_cast<qtensor::AllocOp>(op)) {
          rewriter.eraseOp(tensorAlloc);
          continue;
        }
      }
    }
  }
};
} // namespace
} // namespace mlir::qco