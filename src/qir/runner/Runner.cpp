/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/runtime/QIR.h"

#include <llvm/ADT/StringExtras.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CoreContainers.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h>
#include <llvm/ExecutionEngine/Orc/ExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/LazyReexports.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/SelfExecutorProcessControl.h>
#include <llvm/ExecutionEngine/Orc/SymbolStringPool.h>
#include <llvm/ExecutionEngine/Orc/TargetProcess/TargetExecutionUtils.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/DynamicLibrary.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <cstdlib>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mqt-core-qir-runner"

static llvm::codegen::RegisterCodeGenFlags CGF;

static llvm::cl::opt<std::string> InputFile(llvm::cl::desc("<input bitcode>"),
                                            llvm::cl::Positional,
                                            llvm::cl::init("-"));

static llvm::cl::list<std::string>
    InputArgv(llvm::cl::ConsumeAfter, llvm::cl::desc("<program arguments>..."));

static llvm::ExitOnError ExitOnError;

static void exitOnLazyCallThroughFailure() { exit(1); }

static llvm::Expected<llvm::orc::ThreadSafeModule>
loadModule(const llvm::StringRef path, llvm::orc::ThreadSafeContext tsCtx) {
  llvm::SMDiagnostic err;
  auto m = tsCtx.withContextDo(
      [&](llvm::LLVMContext* ctx) { return parseIRFile(path, err, *ctx); });
  if (!m) {
    std::string errMsg;
    {
      llvm::raw_string_ostream errMsgStream(errMsg);
      err.print(DEBUG_TYPE, errMsgStream);
    }
    return llvm::make_error<llvm::StringError>(std::move(errMsg),
                                               llvm::inconvertibleErrorCode());
  }

  return llvm::orc::ThreadSafeModule(std::move(m), std::move(tsCtx));
}

static int mingwNoopMain() {
  // Cygwin and MinGW insert calls from the main function to the runtime
  // function __main. The __main function is responsible for setting up main's
  // environment (e.g. running static constructors), however this is not needed
  // when running under lli: the executor process will have run non-JIT ctors,
  // and ORC will take care of running JIT'd ctors. To avoid a missing symbol
  // error we just implement __main as a no-op.
  return 0;
}

// Try to enable debugger support for the given instance.
// This always returns success, but prints a warning if it's not able to enable
// debugger support.
static llvm::Error tryEnableDebugSupport(llvm::orc::LLJIT& jit) {
  if (auto err = enableDebuggerSupport(jit)) {
    [[maybe_unused]] const std::string errMsg = toString(std::move(err));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": " << errMsg << "\n");
  }
  return llvm::Error::success();
}

static std::vector<std::pair<std::string, void*>> manualSymbols;

static int runOrcJIT() {
  // Start setting up the JIT environment.

  // Parse the main module.
  const llvm::orc::ThreadSafeContext tsCtx(
      std::make_unique<llvm::LLVMContext>());
  auto mainModule = ExitOnError(loadModule(InputFile, tsCtx));

  // Get TargetTriple and DataLayout from the main module if they're explicitly
  // set.
  std::optional<llvm::Triple> tt;
  std::optional<llvm::DataLayout> dl;
  mainModule.withModuleDo([&](llvm::Module& m) {
    if (!m.getTargetTriple().empty()) {
      tt = m.getTargetTriple();
    }
    if (!m.getDataLayout().isDefault()) {
      dl = m.getDataLayout();
    }
  });

  llvm::orc::LLLazyJITBuilder builder;

  builder.setJITTargetMachineBuilder(
      tt ? llvm::orc::JITTargetMachineBuilder(*tt)
         : ExitOnError(llvm::orc::JITTargetMachineBuilder::detectHost()));

  tt = builder.getJITTargetMachineBuilder()->getTargetTriple();
  if (dl) {
    builder.setDataLayout(dl);
  }

  if (!llvm::codegen::getMArch().empty()) {
    builder.getJITTargetMachineBuilder()->getTargetTriple().setArchName(
        llvm::codegen::getMArch());
  }

  builder.getJITTargetMachineBuilder()
      ->setCPU(llvm::codegen::getCPUStr())
      .addFeatures(llvm::codegen::getFeatureList())
      .setRelocationModel(llvm::codegen::getExplicitRelocModel())
      .setCodeModel(llvm::codegen::getExplicitCodeModel());

  // Link process symbols.
  builder.setLinkProcessSymbolsByDefault(true);

  auto es = std::make_unique<llvm::orc::ExecutionSession>(
      ExitOnError(llvm::orc::SelfExecutorProcessControl::Create()));
  builder.setLazyCallthroughManager(
      std::make_unique<llvm::orc::LazyCallThroughManager>(
          *es, llvm::orc::ExecutorAddr(), nullptr));
  builder.setExecutionSession(std::move(es));

  builder.setLazyCompileFailureAddr(
      llvm::orc::ExecutorAddr::fromPtr(exitOnLazyCallThroughFailure));

  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  builder.setPrePlatformSetup(tryEnableDebugSupport);

  const auto jit = ExitOnError(builder.create());

  auto& jd = jit->getMainJITDylib();
  llvm::orc::SymbolMap hostSymbols;
  for (const auto& [name, ptr] : manualSymbols) {
    hostSymbols[jit->mangleAndIntern(name)] = {
        llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported};
  }
  ExitOnError(jd.define(llvm::orc::absoluteSymbols(hostSymbols)));

  jit->getMainJITDylib().addGenerator(ExitOnError(
      llvm::orc::DynamicLibrarySearchGenerator::GetForCurrentProcess(
          jit->getDataLayout().getGlobalPrefix())));

  auto* objLayer = &jit->getObjLinkingLayer();
  if (auto* rtDyldObjLayer =
          dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(objLayer)) {
    rtDyldObjLayer->registerJITEventListener(
        *llvm::JITEventListener::createGDBRegistrationListener());
  }

  // If this is a Mingw or Cygwin executor then we need to alias __main to
  // orc_rt_int_void_return_0.
  if (jit->getTargetTriple().isOSCygMing()) {
    auto& workaroundJD = jit->getProcessSymbolsJITDylib()
                             ? *jit->getProcessSymbolsJITDylib()
                             : jit->getMainJITDylib();
    ExitOnError(workaroundJD.define(llvm::orc::absoluteSymbols(
        {{jit->mangleAndIntern("__main"),
          {llvm::orc::ExecutorAddr::fromPtr(mingwNoopMain),
           llvm::JITSymbolFlags::Exported}}})));
  }

  // Regular modules are greedy: They materialize as a whole and trigger
  // materialization for all required symbols recursively. Lazy modules go
  // through partitioning and they replace outgoing calls with reexport stubs
  // that resolve on call-through.
  auto addModule = [&](llvm::orc::JITDylib& jd, llvm::orc::ThreadSafeModule m) {
    return jit->addIRModule(jd, std::move(m));
  };

  // Add the main module.
  ExitOnError(addModule(jit->getMainJITDylib(), std::move(mainModule)));

  // Run any static constructors.
  ExitOnError(jit->initialize(jit->getMainJITDylib()));

  // Resolve and run the main function.
  const auto mainAddr = ExitOnError(jit->lookup("main"));

  // Manual in-process execution with RuntimeDyld.
  using mainFnTy = int(int, char**);
  auto mainFn = mainAddr.toPtr<mainFnTy*>();
  const int result =
      llvm::orc::runAsMain(mainFn, InputArgv, llvm::StringRef(InputFile));

  // Run destructors.
  ExitOnError(jit->deinitialize(jit->getMainJITDylib()));

  return result;
}

#define REGISTER_SYMBOL(name)                                                  \
  llvm::sys::DynamicLibrary::AddSymbol(#name,                                  \
                                       reinterpret_cast<void*>(&(name)));      \
  manualSymbols.emplace_back(#name, reinterpret_cast<void*>(&(name)));

auto main(int argc, char* argv[]) -> int {
  const llvm::InitLLVM session(argc, argv);

  if (const std::span args(argv, argc); args.size() > 1) {
    ExitOnError.setBanner(std::string(args[0]) + ": ");
  }

  // If we have a native target, initialize it to ensure it is linked in and
  // usable by the JIT.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qir interpreter & dynamic compiler\n");

  REGISTER_SYMBOL(__quantum__rt__result_get_zero);
  REGISTER_SYMBOL(__quantum__rt__result_get_one);
  REGISTER_SYMBOL(__quantum__rt__result_equal);
  REGISTER_SYMBOL(__quantum__rt__result_update_reference_count);
  REGISTER_SYMBOL(__quantum__rt__array_create_1d);
  REGISTER_SYMBOL(__quantum__rt__array_get_size_1d);
  REGISTER_SYMBOL(__quantum__rt__array_get_element_ptr_1d);
  REGISTER_SYMBOL(__quantum__rt__array_update_reference_count);
  REGISTER_SYMBOL(__quantum__rt__qubit_allocate);
  REGISTER_SYMBOL(__quantum__rt__qubit_allocate_array);
  REGISTER_SYMBOL(__quantum__rt__qubit_release);
  REGISTER_SYMBOL(__quantum__rt__qubit_release_array);
  REGISTER_SYMBOL(__quantum__qis__x__body);
  REGISTER_SYMBOL(__quantum__qis__y__body);
  REGISTER_SYMBOL(__quantum__qis__z__body);
  REGISTER_SYMBOL(__quantum__qis__h__body);
  REGISTER_SYMBOL(__quantum__qis__s__body);
  REGISTER_SYMBOL(__quantum__qis__sdg__body);
  REGISTER_SYMBOL(__quantum__qis__sx__body);
  REGISTER_SYMBOL(__quantum__qis__sxdg__body);
  REGISTER_SYMBOL(__quantum__qis__sqrtx__body);
  REGISTER_SYMBOL(__quantum__qis__sqrtxdg__body);
  REGISTER_SYMBOL(__quantum__qis__t__body);
  REGISTER_SYMBOL(__quantum__qis__tdg__body);
  REGISTER_SYMBOL(__quantum__qis__r__body);
  REGISTER_SYMBOL(__quantum__qis__prx__body);
  REGISTER_SYMBOL(__quantum__qis__rx__body);
  REGISTER_SYMBOL(__quantum__qis__ry__body);
  REGISTER_SYMBOL(__quantum__qis__rz__body);
  REGISTER_SYMBOL(__quantum__qis__p__body);
  REGISTER_SYMBOL(__quantum__qis__rxx__body);
  REGISTER_SYMBOL(__quantum__qis__ryy__body);
  REGISTER_SYMBOL(__quantum__qis__rzz__body);
  REGISTER_SYMBOL(__quantum__qis__rzx__body);
  REGISTER_SYMBOL(__quantum__qis__u__body);
  REGISTER_SYMBOL(__quantum__qis__u3__body);
  REGISTER_SYMBOL(__quantum__qis__u2__body);
  REGISTER_SYMBOL(__quantum__qis__u1__body);
  REGISTER_SYMBOL(__quantum__qis__cu1__body);
  REGISTER_SYMBOL(__quantum__qis__cu3__body);
  REGISTER_SYMBOL(__quantum__qis__cnot__body);
  REGISTER_SYMBOL(__quantum__qis__cx__body);
  REGISTER_SYMBOL(__quantum__qis__cy__body);
  REGISTER_SYMBOL(__quantum__qis__cz__body);
  REGISTER_SYMBOL(__quantum__qis__ch__body);
  REGISTER_SYMBOL(__quantum__qis__swap__body);
  REGISTER_SYMBOL(__quantum__qis__cswap__body);
  REGISTER_SYMBOL(__quantum__qis__crz__body);
  REGISTER_SYMBOL(__quantum__qis__cry__body);
  REGISTER_SYMBOL(__quantum__qis__crx__body);
  REGISTER_SYMBOL(__quantum__qis__cp__body);
  REGISTER_SYMBOL(__quantum__qis__ccx__body);
  REGISTER_SYMBOL(__quantum__qis__ccy__body);
  REGISTER_SYMBOL(__quantum__qis__ccz__body);
  REGISTER_SYMBOL(__quantum__qis__m__body);
  REGISTER_SYMBOL(__quantum__qis__measure__body);
  REGISTER_SYMBOL(__quantum__qis__mz__body);
  REGISTER_SYMBOL(__quantum__qis__reset__body);
  REGISTER_SYMBOL(__quantum__rt__initialize);
  REGISTER_SYMBOL(__quantum__rt__read_result);
  REGISTER_SYMBOL(__quantum__rt__result_record_output);

  return runOrcJIT();
}

#undef REGISTER_SYMBOL
