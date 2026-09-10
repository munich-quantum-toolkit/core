#include "mqt/Compiler/Programs.h"
#include "mqt/Conversion/QCToQCO/QCToQCO.h"
#include "mqt/Dialect/QC/Transforms/Passes.h"
#include "mqt/Dialect/QC/Translation/TranslateQCToOpenQASM3.h"
#include "mqt/Dialect/QCO/Builder/QCOProgramBuilder.h"
#include "mqt/Dialect/QCO/QCOUtils.h"
#include "mqt/Dialect/QTensor/Transforms/Passes.h"
#include "mqt/Support/Passes.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <llvm/Support/raw_ostream.h>
#include <mlir/IR/Verifier.h>
#include <mlir/Pass/PassManager.h>
#include <mlir/Transforms/Passes.h>
#include <optional>
#include <string>
#include <vector>
using Clock = std::chrono::steady_clock;
static double elapsed(Clock::time_point start) {
  return std::chrono::duration<double, std::milli>(Clock::now() - start)
      .count();
}
static size_t operations(mlir::ModuleOp mod) {
  size_t n = 0;
  mod.walk([&](mlir::Operation*) { ++n; });
  return n;
}
int main(int argc, char** argv) {
  if (argc < 5)
    return 1;
  std::string mode = argv[1], format = argv[2];
  if (mode == "builder") {
    auto context = mlir::createCompilerContext();
    mlir::qco::QCOProgramBuilder builder(context.get());
    builder.initialize();
    if (format == "scalar") {
      for (int i = 0; i < std::atoi(argv[3]); ++i) {
        auto q = builder.allocQubit();
        builder.h(q);
      }
    } else {
      for (int i = 0; i < std::atoi(argv[3]); ++i) {
        auto t = builder.qtensorAlloc(4);
        for (int j = 0; j < 4; ++j) {
          auto [next, q] = builder.qtensorExtract(t, j);
          t = next;
          builder.h(q);
        }
      }
    }
    auto mod = builder.finalize();
    if (!mod || mlir::failed(mlir::verify(*mod)) ||
        mlir::failed(mlir::qco::verifyLinearity(*mod)))
      return 20;
    mod->print(llvm::outs());
    llvm::outs() << "\n";
    return 0;
  }
  if (mode == "builder-prep") {
    auto context = mlir::createCompilerContext();
    context->disableMultithreading();
    llvm::outs() << "{\"samples\":[";
    for (int rep = -2; rep < std::atoi(argv[4]); ++rep) {
      mlir::qco::QCOProgramBuilder builder(context.get());
      builder.initialize();
      llvm::SmallVector<mlir::Value> tensors;
      for (int i = 0; i < std::atoi(argv[3]); ++i) {
        auto t = builder.qtensorAlloc(1);
        if (format == "extracted") {
          auto [next, q] = builder.qtensorExtract(t, 0);
          t = next;
          builder.h(q);
        }
        tensors.push_back(t);
      }
      auto start = Clock::now();
      builder.qcoIf(true, mlir::ValueRange(tensors), [](mlir::ValueRange args) {
        return llvm::SmallVector<mlir::Value>(args);
      });
      auto ms = elapsed(start);
      auto mod = builder.finalize();
      if (!mod || mlir::failed(mlir::verify(*mod)) ||
          mlir::failed(mlir::qco::verifyLinearity(*mod)))
        return 24;
      if (rep >= 0) {
        if (rep)
          llvm::outs() << ',';
        llvm::outs() << "{\"ms\":[" << ms << "],\"ops\":" << operations(*mod)
                     << '}';
      }
    }
    llvm::outs() << "]}\n";
    return 0;
  }
  if (mode == "canonicalize-qco") {
    auto originalQCO = mlir::QCOProgram::fromMLIRFile(argv[3]);
    if (!originalQCO)
      return 21;
    auto* context = originalQCO->module().getContext();
    context->disableMultithreading();
    llvm::outs() << "{\"samples\":[";
    for (int i = -2; i < std::atoi(argv[4]); ++i) {
      auto program = originalQCO->copy();
      auto start = Clock::now();
      mlir::PassManager pm(context);
      pm.addPass(mlir::createCanonicalizerPass());
      if (mlir::failed(pm.run(program.module())))
        return 22;
      auto ms = elapsed(start);
      if (mlir::failed(mlir::verify(program.module())) ||
          mlir::failed(mlir::qco::verifyLinearity(program.module())))
        return 23;
      if (argc > 5 && i == 0) {
        std::ofstream file(argv[5]);
        file << program.str();
      }
      if (i >= 0) {
        if (i)
          llvm::outs() << ',';
        llvm::outs() << "{\"ms\":[" << ms
                     << "],\"ops\":" << operations(program.module()) << '}';
      }
    }
    llvm::outs() << "]}\n";
    return 0;
  }

  auto original = mlir::QCProgram::fromMLIRFile(argv[3]);
  if (!original)
    return 2;
  auto* context = original->module().getContext();
  context->disableMultithreading();
  int reps = std::atoi(argv[4]);
  std::optional<mlir::QCOProgram> prepared;
  if (mode != "pipeline" && mode != "qc-shrink") {
    auto qc = original->copy();
    prepared = std::move(qc).intoQCO();
    if (!prepared)
      return 3;
    if (mode == "tail" || mode == "tail-raw" || mode == "tail-once") {
      if (!prepared->cleanup() ||
          !prepared->runPassPipeline("mqt-qco-default") || !prepared->cleanup())
        return 4;
    }
  }
  llvm::outs() << "{\"input_ops\":" << operations(original->module())
               << ",\"qco_ops\":"
               << (prepared ? operations(prepared->module()) : 0)
               << ",\"samples\":[";
  for (int i = -2; i < reps; ++i) {
    std::string emitted;
    std::vector<double> timings;
    size_t outputBytes = 0, outputOps = 0;
    if (mode == "pipeline") {
      auto qc = original->copy();
      auto out = format == "qasm"   ? mlir::ProgramFormat::OpenQASM3
                 : format == "base" ? mlir::ProgramFormat::QIRBase
                                    : mlir::ProgramFormat::QIRAdaptive;
      auto start = Clock::now();
      auto result =
          mlir::runDefaultPipeline(mlir::CompilerInput(std::move(qc)), out);
      timings.push_back(elapsed(start));
      if (!result)
        return 5;
      if (format == "qasm")
        outputBytes = std::get<mlir::OpenQASMProgram>(*result).str().size();
      else {
        start = Clock::now();
        auto text = std::get<mlir::QIRProgram>(*result).llvmIR();
        timings.push_back(elapsed(start));
        if (!text)
          return 6;
        emitted = *text;
        outputBytes = text->size();
      }
    } else if (mode == "qc-shrink") {
      auto qc = original->copy();
      auto start = Clock::now();
      mlir::PassManager pm(context);
      pm.addPass(mlir::qc::createShrinkQubitRegistersPass());
      if (mlir::failed(pm.run(qc.module())))
        return 7;
      timings.push_back(elapsed(start));
      outputOps = operations(qc.module());
    } else {
      auto qco = prepared->copy();
      auto start = Clock::now();
      if (mode == "qco-cleanup") {
        if (!qco.cleanup())
          return 8;
        timings.push_back(elapsed(start));
        outputOps = operations(qco.module());
      } else if (mode == "qco-shrink" || mode == "canonicalize") {
        mlir::PassManager pm(context);
        if (mode == "canonicalize")
          pm.addPass(mlir::createCanonicalizerPass());
        else
          pm.addPass(mlir::qtensor::createShrinkQTensorToFitPass());
        if (mlir::failed(pm.run(qco.module())))
          return 9;
        timings.push_back(elapsed(start));
        outputOps = operations(qco.module());
        if (mlir::failed(mlir::qco::verifyLinearity(qco.module())))
          return 10;
      } else {
        auto qc = std::move(qco).intoQC();
        timings.push_back(elapsed(start));
        if (!qc)
          return 11;
        start = Clock::now();
        if (mode != "tail-once" && !qc->cleanup())
          return 12;
        timings.push_back(elapsed(start));
        start = Clock::now();
        if (format == "qasm") {
          if (mode == "tail-raw") {
            auto text = mlir::qc::translateQCToOpenQASM3(qc->module());
            if (mlir::failed(text))
              return 13;
            emitted = *text;
            outputBytes = text->size();
          } else {
            auto text = qc->toOpenQASM3();
            if (!text)
              return 14;
            emitted = text->str();
            outputBytes = text->str().size();
          }
          timings.push_back(elapsed(start));
        } else {
          auto qir = std::move(*qc).intoQIR(format == "base"
                                                ? mlir::QIRProfile::Base
                                                : mlir::QIRProfile::Adaptive);
          timings.push_back(elapsed(start));
          if (!qir)
            return 15;
          start = Clock::now();
          auto text = qir->llvmIR();
          timings.push_back(elapsed(start));
          if (!text)
            return 16;
          emitted = *text;
          outputBytes = text->size();
        }
      }
    }
    if (argc > 5 && i == 0 && !emitted.empty()) {
      std::ofstream file(argv[5]);
      file << emitted;
    }
    if (i >= 0) {
      if (i)
        llvm::outs() << ',';
      llvm::outs() << "{\"ms\":[";
      for (size_t k = 0; k < timings.size(); ++k) {
        if (k)
          llvm::outs() << ',';
        llvm::outs() << timings[k];
      }
      llvm::outs() << "],\"bytes\":" << outputBytes << ",\"ops\":" << outputOps
                   << '}';
    }
  }
  llvm::outs() << "]}\n";
}
