#include "mlir/Dialect/MQTOpt/IR/MQTOptDialect.h"
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h"
#include "mlir/Dialect/MQTOpt/Transforms/Transpilation/Architecture.h"

#include <cstddef>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/BuiltinOps.h>

#define DEBUG_TYPE "transpilation-verification"

namespace mqt::ir::opt {

#define GEN_PASS_DEF_TRANSPILATIONVERIFICATIONPASS
#include "mlir/Dialect/MQTOpt/Transforms/Passes.h.inc"

using namespace mlir;

/**
 * @brief This pass verifies that the constraints of a target architecture are
 * met.
 */
struct TranspilationVerificationPass final
    : impl::TranspilationVerificationPassBase<TranspilationVerificationPass> {
  void runOnOperation() override {

    std::size_t nqubits{};
    llvm::DenseMap<Value, std::size_t> qubitToIndex;

    const auto forward = [&](const Value in, const Value out) {
      qubitToIndex[out] = qubitToIndex[in];
      qubitToIndex.erase(in);
    };

    auto arch = transpilation::getArchitecture("MQT-Test");

    auto res = getOperation()->walk([&](Operation* op) {
      // Skip any initialized static qubits.
      if (auto qubit = dyn_cast<QubitOp>(op)) {
        if (nqubits == arch->nqubits()) {
          return WalkResult(qubit->emitOpError()
                            << "requires " << (nqubits + 1)
                            << " qubits but target architecture '"
                            << arch->name() << "' only supports "
                            << arch->nqubits() << " qubits");
        }

        qubitToIndex[qubit.getQubit()] = qubit.getIndex();

        return WalkResult::advance();
      }

      // As of now, we don't support conditionals. Hence, emit an error.
      if (auto cond = dyn_cast<scf::IfOp>(op)) {
        return WalkResult(cond.emitOpError() << "is currently not supported");
      }

      // As of now, we don't support loops with qubit dependencies. Hence, emit
      // an error.
      if (auto loop = dyn_cast<scf::ForOp>(op)) {
        if (loop.getRegionIterArgs().size() == 0) {
          return WalkResult::advance();
        }
        return WalkResult(
            loop.emitOpError()
            << "is currently not supported with qubit dependencies");
      }

      if (auto alloc = dyn_cast<AllocQubitOp>(op)) {
        return WalkResult(alloc->emitOpError()
                          << "is not allowed for transpiled program");
      }

      if (auto dealloc = dyn_cast<DeallocQubitOp>(op)) {
        return WalkResult(dealloc->emitOpError()
                          << "is not allowed for transpiled program");
      }

      if (auto reset = dyn_cast<ResetOp>(op)) {
        forward(reset.getInQubit(), reset.getOutQubit());
        return WalkResult::advance();
      }

      if (auto u = dyn_cast<UnitaryInterface>(op)) {
        const std::size_t nacts = u.getAllInQubits().size();
        if (nacts > 2) {
          return WalkResult(u->emitOpError()
                            << "acts on more than two qubits");
        }

        const Value in0 = u.getAllInQubits()[0];
        const Value out0 = u.getAllOutQubits()[0];

        if (nacts == 1) {
          forward(in0, out0);
          return WalkResult::advance();
        }

        if (nacts == 2) {
          const Value in1 = u.getAllInQubits()[1];
          const Value out1 = u.getAllOutQubits()[1];

          if (!arch->areAdjacent(qubitToIndex[in0], qubitToIndex[in1])) {
            return WalkResult(u->emitOpError()
                              << "is not executable on target architecture '"
                              << arch->name() << "'");
          }

          if (dyn_cast<SWAPOp>(op)) {
            forward(in0, out1);
            forward(in1, out0);
            return WalkResult::advance();
          }

          forward(in0, out0);
          forward(in1, out1);
          return WalkResult::advance();
        }

        return WalkResult::advance();
      }

      if (auto measure = dyn_cast<MeasureOp>(op)) {
        forward(measure.getInQubit(), measure.getOutQubit());
        return WalkResult::advance();
      }

      return WalkResult::advance();
    });

    if (res.wasInterrupted()) {
      signalPassFailure();
    }
  }
};
} // namespace mqt::ir::opt