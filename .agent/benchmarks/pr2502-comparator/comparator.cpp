#include "Support/IRVerification.h"
#include "mqt/Dialect/QCO/IR/QCODialect.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include <chrono>
#include <iostream>
#include <string>

bool areModulesEquivalentBaseline(mlir::ModuleOp, mlir::ModuleOp);

int main() {
  mlir::DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect, mlir::qco::QCODialect>();
  mlir::MLIRContext context(registry);
  std::cout << "gates,comparator,sample,milliseconds\n";
  for (int count : {1000, 2000, 4000}) {
    std::string source = "func.func @f(%q0: !qco.qubit) -> !qco.qubit {\n";
    for (int i = 1; i <= count; ++i) {
      source += "%q" + std::to_string(i) + " = qco.h %q" +
                std::to_string(i - 1) + " : !qco.qubit -> !qco.qubit\n";
    }
    source += "return %q" + std::to_string(count) + " : !qco.qubit\n}";
    auto lhs = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    auto rhs = mlir::parseSourceString<mlir::ModuleOp>(source, &context);
    if (!lhs || !rhs || mlir::failed(mlir::verify(*lhs)) ||
        mlir::failed(mlir::verify(*rhs)) ||
        mlir::failed(mlir::qco::verifyLinearity(*lhs)) ||
        mlir::failed(mlir::qco::verifyLinearity(*rhs))) {
      return 1;
    }
    for (int sample = 0; sample < 7; ++sample) {
      for (const auto* mode : {"baseline", "structural", "permutation"}) {
        auto start = std::chrono::steady_clock::now();
        bool equal = std::string(mode) == "baseline"
                         ? areModulesEquivalentBaseline(*lhs, *rhs)
                     : std::string(mode) == "structural"
                         ? areModulesStructurallyEquivalent(*lhs, *rhs)
                         : areModulesEquivalentWithPermutations(*lhs, *rhs);
        auto elapsed = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - start).count();
        if (!equal) {
          return 2;
        }
        std::cout << count << ',' << mode << ',' << sample << ',' << elapsed << '\n';
      }
    }
  }
}
