#include "mqt/Compiler/Programs.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"

#include "mlir/IR/Verifier.h"

#include "llvm/Support/raw_ostream.h"

int main() {
  auto context = mlir::createCompilerContext();
  mlir::qco::QCOProgramBuilder builder(context.get());
  builder.initialize();
  auto [tensor, qubit] = builder.qtensorExtract(builder.qtensorAlloc(1), 0);
  builder.qcoIf(
      true, mlir::ValueRange{tensor, qubit}, [&](mlir::ValueRange args) {
        auto filled = builder.qtensorInsert(args[1], args[0], 0);
        auto [nextTensor, nextQubit] = builder.qtensorExtract(filled, 0);
        return llvm::SmallVector<mlir::Value>{nextTensor, nextQubit};
      });
  auto moduleOp = builder.finalize();
  moduleOp->print(llvm::outs());
  return mlir::failed(mlir::verify(*moduleOp)) ? 1 : 0;
}
