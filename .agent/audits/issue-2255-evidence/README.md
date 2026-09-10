🤖 *AI text below* 🤖 <!-- rumdl-disable-line MD041 -->

# Evidence for the MLIR contract audit

This folder contains 19 reduced inputs and 23 recorded probe outcomes for the
[audit report](../issue-2255-contracts-2026-09-10.md). It does not implement
fixes or add cases to the regular test suite.

## Recorded run

- Tested commit: `d994fe6833b6b7a9b1bccdeccc64e31c5c09ffd1`.
- Toolchain: AppleClang 21, LLVM/MLIR 23.1.0, arm64, Release with assertions.
- [Recorded outcomes](results-d994fe683.json) retain exit status, stdout,
  stderr, and the five-second timeout flag. Only checkout prefixes and input
  locations were normalized to portable paths; diagnostics were not shortened.
- A zero exit code does not mean the intended contract holds. For example, the
  mapping and QIR probes return success with incorrect output, while the
  verifier probes expose missing checks by accepting invalid signatures.
- The syntax-invalid exploratory inputs are deliberately not included.
- These are diagnostic subprocesses because some baseline cases abort or do not
  finish. Accepted fixes should add direct in-process regression tests.

The existing suite counts from the same baseline were QCO utilities 192,
optimizations 198, QCO IR 579, and compiler 204: 1,173 tests, no failures. Those
selected suites are not the complete repository test matrix.

## Reproduce

Use a separate checkout of the tested commit, then copy this evidence folder
into `.agent/audits/`. Configure the repository's Release preset with the
supported LLVM/MLIR toolchain, Ninja, and compile-command export:

```sh
cmake --preset release -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build build/release --parallel 4 --target mqt-core-mlir-unittests-compiler
python3 .agent/audits/issue-2255-evidence/compile_probe.py
python3 .agent/audits/issue-2255-evidence/run_probes.py
```

The scripts use Python's standard library and the configured compiler-test
compile/link commands. They require a Ninja build on a POSIX host. The
diagnostic source below is extracted into `build/release/issue-2255-probe.cpp`;
the executable and fresh results also stay under `build/release/`. The committed
historical results are never overwritten. Running against a different commit
tests that commit, not the recorded baseline.

The driver and wrapper were reduced for publication to remove unused exploratory
modes. The recorded outcomes came from the original driver with the same
retained mode bodies. The report distinguishes baseline evidence from
publication validation.

## Diagnostic driver

The driver runs ordinary verification and QCO linearity before transformations,
then checks successful output. Passing these checks is not a claim that a
verifier-omission input is semantically valid.

```cpp
#include "mqt/Compiler/Programs.h"
#include "mqt/Compiler/Target.h"
#include "mqt/Compiler/TargetEnvironment.h"
#include "mqt/Conversion/QCOToJeff/QCOToJeff.h"
#include "mqt/Conversion/QCOToQC/QCOToQC.h"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Conversion/QCToQIR/QIRAdaptive/QCToQIRAdaptive.h"
#include "mqt/Conversion/QCToQIR/QIRBase/QCToQIRBase.h"
#include "mqt/Dialect/QCO/IR/QCOOps.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QCO/Transforms/Mapping/Mapping.h"
#include "mqt/Dialect/QCO/Transforms/Passes.h"
#include "mqt/Dialect/QCO/Utils/Sorting.h"
#include "mqt/Dialect/QIR/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>

int main(int argc, char** argv) {
  if (argc != 3) {
    llvm::errs() << "usage: probe MODE INPUT.mlir\n";
    return 64;
  }
  const std::string mode = argv[1];
  auto context = mlir::createCompilerContext();
  if (mode == "qc-unregistered" || mode == "qco-unregistered") {
    context->allowUnregisteredDialects();
  }
  auto moduleOp = mlir::parseSourceFile<mlir::ModuleOp>(argv[2], context.get());
  if (!moduleOp || mlir::failed(mlir::verify(*moduleOp))) {
    llvm::errs() << "INPUT_REJECTED\n";
    return 2;
  }
  if (mlir::failed(mlir::qco::verifyLinearity(*moduleOp))) {
    llvm::errs() << "INPUT_NOT_LINEAR\n";
    return 3;
  }
  llvm::errs() << "INPUT_VERIFIED_AND_LINEAR\n";
  if (mode == "large-beta") {
    for (const auto matrix :
         {mlir::qco::XXPlusYYOp::unitaryMatrix(1.0, 1.0e16),
          mlir::qco::XXMinusYYOp::unitaryMatrix(1.0, 1.0e16)}) {
      const auto product = matrix * matrix.adjoint();
      double error = 0;
      for (size_t row = 0; row < 4; ++row) {
        for (size_t column = 0; column < 4; ++column) {
          error = std::max(error, std::abs(product(row, column) -
                                           (row == column ? 1.0 : 0.0)));
        }
      }
      llvm::outs() << "MAX_UNITARITY_ERROR=" << error << '\n';
    }
    return 0;
  }
  if (mode == "verify") {
    return 0;
  }
  if (mode == "qc-unregistered") {
    return mlir::QCProgram::fromModule(context, std::move(moduleOp)) ? 0 : 4;
  }
  if (mode == "qco-unregistered") {
    return mlir::QCOProgram::fromModule(context, std::move(moduleOp)) ? 0 : 4;
  }
  if (mode == "roundtrip") {
    std::string text;
    llvm::raw_string_ostream stream(text);
    moduleOp->print(stream);
    llvm::outs() << text << '\n';
    auto reparsed =
        mlir::parseSourceString<mlir::ModuleOp>(text, context.get());
    if (!reparsed || mlir::failed(mlir::verify(*reparsed))) {
      llvm::errs() << "ROUNDTRIP_REJECTED\n";
      return 5;
    }
    return 0;
  }
  if (mode == "sort") {
    mlir::IRRewriter rewriter(context.get());
    for (auto function : moduleOp->getOps<mlir::func::FuncOp>()) {
      if (!function.isDeclaration()) {
        mlir::qco::reorderTopologically(function.getBody().front(), rewriter);
      }
    }
  } else {
    mlir::PassManager manager(context.get());
    if (mode == "unroll-full") {
      mlir::qco::QuantumLoopUnrollOptions options;
      options.unrollFactor = -1;
      manager.addNestedPass<mlir::func::FuncOp>(
          mlir::qco::createQuantumLoopUnroll(options));
    } else if (mode == "mapping") {
      auto target = llvm::cantFail(mlir::CompilerTarget::create(
          2, mlir::CompilerTarget::Connectivity::fromCouplings({{0, 1}}),
          mlir::CompilerTarget::NativeOperations::unrestricted()));
      mlir::PayloadFormat format;
      format.id = "test.payload";
      format.version = "1.0.0";
      auto payload =
          llvm::cantFail(mlir::PayloadSpecification::create(std::move(format)));
      mlir::attachTargetEnvironment(*moduleOp,
                                    mlir::TargetEnvironment(target, payload));
      mlir::qco::MappingPassOptions options;
      options.ntrials = 1;
      options.niterations = 1;
      manager.addPass(mlir::qco::createMappingPass(options));
    } else if (mode == "measurement-lifting") {
      manager.addPass(mlir::qco::createMeasurementLifting());
    } else if (mode == "hadamard-lifting") {
      manager.addPass(mlir::qco::createHadamardLifting());
    } else if (mode == "attach-adaptive") {
      manager.addPass(mlir::qir::createQIRSetAttributesAndMetadata({true}));
    } else if (mode == "attach-base") {
      manager.addPass(mlir::qir::createQIRSetAttributesAndMetadata({false}));
    } else if (mode == "qc-qco") {
      manager.addPass(mlir::createQCToQCO());
    } else if (mode == "qco-qc") {
      manager.addPass(mlir::createQCOToQC());
    } else if (mode == "qco-jeff") {
      manager.addPass(mlir::createQCOToJeff());
    } else if (mode == "qc-qir-base") {
      manager.addPass(mlir::createQCToQIRBase());
    } else if (mode == "qc-qir-adaptive") {
      manager.addPass(mlir::createQCToQIRAdaptive());
    } else {
      return 64;
    }
    if (mlir::failed(manager.run(*moduleOp))) {
      llvm::errs() << "PASS_FAILED\n";
      return 6;
    }
  }
  if (mlir::failed(mlir::verify(*moduleOp)) ||
      mlir::failed(mlir::qco::verifyLinearity(*moduleOp))) {
    llvm::errs() << "OUTPUT_INVALID\n";
    return 7;
  }
  llvm::errs() << "OUTPUT_VERIFIED_AND_LINEAR\n";
  moduleOp->print(llvm::outs());
  llvm::outs() << '\n';
}
```
