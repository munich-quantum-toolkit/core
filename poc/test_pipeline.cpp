/*
 * Standalone C++ test for the (py:qasm) -> (mlir:qc) -> (mlir:qco) pipeline.
 *
 * Build:
 *   cmake --build build --target poc-test-pipeline
 * Run:
 *   ./build/poc/poc-test-pipeline
 */

#include "ir/QuantumComputation.hpp" // NOLINT(misc-include-cleaner)
#include "mlir/Conversion/QCToQCO/QCToQCO.h"
#include "mlir/Dialect/QC/IR/QCDialect.h"
#include "mlir/Dialect/QC/Translation/TranslateQuantumComputationToQC.h"
#include "mlir/Dialect/QCO/IR/QCODialect.h"
#include "mlir/Dialect/QTensor/IR/QTensorDialect.h"
#include "mlir/Support/Passes.h"
#include "qasm3/Importer.hpp"

#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/Arith/IR/Arith.h>
#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/MemRef/IR/MemRef.h>
#include <mlir/Dialect/SCF/IR/SCF.h>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Support/LogicalResult.h>

namespace {
const char* const BELL_QASM = R"(
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
)";
} // namespace

int main() {
  auto qc = qasm3::Importer::imports(BELL_QASM);

  mlir::MLIRContext ctx;
  mlir::DialectRegistry registry;
  registry.insert<mlir::qc::QCDialect, mlir::qco::QCODialect,
                  mlir::qtensor::QTensorDialect, mlir::arith::ArithDialect,
                  mlir::func::FuncDialect, mlir::memref::MemRefDialect,
                  mlir::scf::SCFDialect>();
  ctx.appendDialectRegistry(registry);
  ctx.loadAllAvailableDialects();

  auto module = mlir::translateQuantumComputationToQC(&ctx, qc);
  if (!module) {
    llvm::errs() << "translation failed\n";
    return 1;
  }

  llvm::outs() << "=== mlir:qc ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  mlir::PassManager pm(&ctx);
  populateQCCleanupPipeline(pm);
  pm.addPass(mlir::createQCToQCO());
  if (mlir::failed(pm.run(*module))) {
    llvm::errs() << "conversion failed\n";
    return 1;
  }

  llvm::outs() << "=== mlir:qco ===\n";
  module->print(llvm::outs());
  llvm::outs() << "\n";

  return 0;
}
