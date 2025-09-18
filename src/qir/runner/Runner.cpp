/*
 * Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
 * Copyright (c) 2025 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "qir/backend/QIR.h"

#include <cstdlib>
#include <llvm/ADT/StringExtras.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h>
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
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>

#define DEBUG_TYPE "mqt-core-qir-runner"

namespace {
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables,readability-identifier-naming)
llvm::codegen::RegisterCodeGenFlags CGF;

const llvm::cl::opt<std::string> INPUT_FILE(llvm::cl::desc("<input bitcode>"),
                                            llvm::cl::Positional,
                                            llvm::cl::init("-"));

const llvm::cl::list<std::string>
    INPUT_ARGV(llvm::cl::ConsumeAfter,
               llvm::cl::desc("<program arguments>..."));

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
llvm::ExitOnError exitOnErr;

void exitOnLazyCallThroughFailure() { exit(1); }

llvm::Expected<llvm::orc::ThreadSafeModule>
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

int mingwNoopMain() {
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
llvm::Error tryEnableDebugSupport(llvm::orc::LLJIT& jit) {
  if (auto err = enableDebuggerSupport(jit)) {
    [[maybe_unused]] const std::string errMsg = toString(std::move(err));
    // NOLINTNEXTLINE(cppcoreguidelines-avoid-do-while)
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": " << errMsg << "\n");
  }
  return llvm::Error::success();
}

int runOrcJIT() {
  // Start setting up the JIT environment.

  // Parse the main module.
  const llvm::orc::ThreadSafeContext tsCtx(
      std::make_unique<llvm::LLVMContext>());
  auto mainModule = exitOnErr(loadModule(INPUT_FILE, tsCtx));

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
         : exitOnErr(llvm::orc::JITTargetMachineBuilder::detectHost()));

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
      exitOnErr(llvm::orc::SelfExecutorProcessControl::Create()));
  builder.setLazyCallthroughManager(
      std::make_unique<llvm::orc::LazyCallThroughManager>(
          *es, llvm::orc::ExecutorAddr(), nullptr));
  builder.setExecutionSession(std::move(es));

  builder.setLazyCompileFailureAddr(
      llvm::orc::ExecutorAddr::fromPtr(exitOnLazyCallThroughFailure));

  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  builder.setPrePlatformSetup(tryEnableDebugSupport);

  const auto jit = exitOnErr(builder.create());

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
    exitOnErr(workaroundJD.define(llvm::orc::absoluteSymbols(
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
  exitOnErr(addModule(jit->getMainJITDylib(), std::move(mainModule)));

  // Run any static constructors.
  exitOnErr(jit->initialize(jit->getMainJITDylib()));

  // Resolve and run the main function.
  const auto mainAddr = exitOnErr(jit->lookup("main"));

  // Manual in-process execution with RuntimeDyld.
  using mainFnTy = int(int, char**);
  auto mainFn = mainAddr.toPtr<mainFnTy*>();
  const int result =
      llvm::orc::runAsMain(mainFn, INPUT_ARGV, llvm::StringRef(INPUT_FILE));

  // Run destructors.
  exitOnErr(jit->deinitialize(jit->getMainJITDylib()));

  return result;
}
} // namespace

auto main(int argc, char* argv[]) -> int {
  const llvm::InitLLVM session(argc, argv);
  const std::span args(argv, argc);

  if (args.size() > 1) {
    exitOnErr.setBanner(std::string(args[0]) + ": ");
  }

  // If we have a native target, initialize it to ensure it is linked in and
  // usable by the JIT.
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  llvm::InitializeNativeTargetAsmParser();

  llvm::cl::ParseCommandLineOptions(argc, argv,
                                    "qir interpreter & dynamic compiler\n");

  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_get_zero",
      reinterpret_cast<void*>(&__quantum__rt__result_get_zero));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_get_one",
      reinterpret_cast<void*>(&__quantum__rt__result_get_one));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_equal",
      reinterpret_cast<void*>(&__quantum__rt__result_equal));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__result_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_create",
      reinterpret_cast<void*>(&__quantum__rt__string_create));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_get_data",
      reinterpret_cast<void*>(&__quantum__rt__string_get_data));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_get_length",
      reinterpret_cast<void*>(&__quantum__rt__string_get_length));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__string_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_concatenate",
      reinterpret_cast<void*>(&__quantum__rt__string_concatenate));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__string_equal",
      reinterpret_cast<void*>(&__quantum__rt__string_equal));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__int_to_string",
      reinterpret_cast<void*>(&__quantum__rt__int_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__double_to_string",
      reinterpret_cast<void*>(&__quantum__rt__double_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bool_to_string",
      reinterpret_cast<void*>(&__quantum__rt__bool_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_to_string",
      reinterpret_cast<void*>(&__quantum__rt__result_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__pauli_to_string",
      reinterpret_cast<void*>(&__quantum__rt__pauli_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_to_string",
      reinterpret_cast<void*>(&__quantum__rt__qubit_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__range_to_string",
      reinterpret_cast<void*>(&__quantum__rt__range_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_to_string",
      reinterpret_cast<void*>(&__quantum__rt__bigint_to_string));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_create_i64",
      reinterpret_cast<void*>(&__quantum__rt__bigint_create_i64));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_create_array",
      reinterpret_cast<void*>(&__quantum__rt__bigint_create_array));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_get_data",
      reinterpret_cast<void*>(&__quantum__rt__bigint_get_data));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_get_length",
      reinterpret_cast<void*>(&__quantum__rt__bigint_get_length));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__bigint_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_negate",
      reinterpret_cast<void*>(&__quantum__rt__bigint_negate));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_add",
      reinterpret_cast<void*>(&__quantum__rt__bigint_add));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_subtract",
      reinterpret_cast<void*>(&__quantum__rt__bigint_subtract));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_multiply",
      reinterpret_cast<void*>(&__quantum__rt__bigint_multiply));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_divide",
      reinterpret_cast<void*>(&__quantum__rt__bigint_divide));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_modulus",
      reinterpret_cast<void*>(&__quantum__rt__bigint_modulus));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_power",
      reinterpret_cast<void*>(&__quantum__rt__bigint_power));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_bitand",
      reinterpret_cast<void*>(&__quantum__rt__bigint_bitand));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_bitor",
      reinterpret_cast<void*>(&__quantum__rt__bigint_bitor));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_bitxor",
      reinterpret_cast<void*>(&__quantum__rt__bigint_bitxor));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_bitnot",
      reinterpret_cast<void*>(&__quantum__rt__bigint_bitnot));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_shiftleft",
      reinterpret_cast<void*>(&__quantum__rt__bigint_shiftleft));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_shiftright",
      reinterpret_cast<void*>(&__quantum__rt__bigint_shiftright));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_equal",
      reinterpret_cast<void*>(&__quantum__rt__bigint_equal));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_greater",
      reinterpret_cast<void*>(&__quantum__rt__bigint_greater));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bigint_greater_eq",
      reinterpret_cast<void*>(&__quantum__rt__bigint_greater_eq));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_create",
      reinterpret_cast<void*>(&__quantum__rt__tuple_create));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_copy",
      reinterpret_cast<void*>(&__quantum__rt__tuple_copy));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__tuple_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_update_alias_count",
      reinterpret_cast<void*>(&__quantum__rt__tuple_update_alias_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_create_1d",
      reinterpret_cast<void*>(&__quantum__rt__array_create_1d));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_copy",
      reinterpret_cast<void*>(&__quantum__rt__array_copy));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_concatenate",
      reinterpret_cast<void*>(&__quantum__rt__array_concatenate));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_slice_1d",
      reinterpret_cast<void*>(&__quantum__rt__array_slice_1d));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_size_1d",
      reinterpret_cast<void*>(&__quantum__rt__array_get_size_1d));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_element_ptr_1d",
      reinterpret_cast<void*>(&__quantum__rt__array_get_element_ptr_1d));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__array_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_update_alias_count",
      reinterpret_cast<void*>(&__quantum__rt__array_update_alias_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_create",
      reinterpret_cast<void*>(&__quantum__rt__array_create));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_dim",
      reinterpret_cast<void*>(&__quantum__rt__array_get_dim));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_size",
      reinterpret_cast<void*>(&__quantum__rt__array_get_size));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_get_element_ptr",
      reinterpret_cast<void*>(&__quantum__rt__array_get_element_ptr));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_slice",
      reinterpret_cast<void*>(&__quantum__rt__array_slice));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_project",
      reinterpret_cast<void*>(&__quantum__rt__array_project));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_create",
      reinterpret_cast<void*>(&__quantum__rt__callable_create));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_copy",
      reinterpret_cast<void*>(&__quantum__rt__callable_copy));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_invoke",
      reinterpret_cast<void*>(&__quantum__rt__callable_invoke));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_make_adjoint",
      reinterpret_cast<void*>(&__quantum__rt__callable_make_adjoint));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_make_controlled",
      reinterpret_cast<void*>(&__quantum__rt__callable_make_controlled));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__callable_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__callable_update_alias_count",
      reinterpret_cast<void*>(&__quantum__rt__callable_update_alias_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__capture_update_reference_count",
      reinterpret_cast<void*>(&__quantum__rt__capture_update_reference_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__capture_update_alias_count",
      reinterpret_cast<void*>(&__quantum__rt__capture_update_alias_count));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__message",
      reinterpret_cast<void*>(&__quantum__rt__message));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__fail", reinterpret_cast<void*>(&__quantum__rt__fail));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_allocate",
      reinterpret_cast<void*>(&__quantum__rt__qubit_allocate));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_allocate_array",
      reinterpret_cast<void*>(&__quantum__rt__qubit_allocate_array));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_release",
      reinterpret_cast<void*>(&__quantum__rt__qubit_release));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__qubit_release_array",
      reinterpret_cast<void*>(&__quantum__rt__qubit_release_array));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__x__body",
      reinterpret_cast<void*>(&__quantum__qis__x__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__y__body",
      reinterpret_cast<void*>(&__quantum__qis__y__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__z__body",
      reinterpret_cast<void*>(&__quantum__qis__z__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__h__body",
      reinterpret_cast<void*>(&__quantum__qis__h__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__s__body",
      reinterpret_cast<void*>(&__quantum__qis__s__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__sdg__body",
      reinterpret_cast<void*>(&__quantum__qis__sdg__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__sx__body",
      reinterpret_cast<void*>(&__quantum__qis__sx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__sxdg__body",
      reinterpret_cast<void*>(&__quantum__qis__sxdg__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__sqrtx__body",
      reinterpret_cast<void*>(&__quantum__qis__sqrtx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__sqrtxdg__body",
      reinterpret_cast<void*>(&__quantum__qis__sqrtxdg__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__t__body",
      reinterpret_cast<void*>(&__quantum__qis__t__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__tdg__body",
      reinterpret_cast<void*>(&__quantum__qis__tdg__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__rx__body",
      reinterpret_cast<void*>(&__quantum__qis__rx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__ry__body",
      reinterpret_cast<void*>(&__quantum__qis__ry__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__rz__body",
      reinterpret_cast<void*>(&__quantum__qis__rz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__p__body",
      reinterpret_cast<void*>(&__quantum__qis__p__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__rxx__body",
      reinterpret_cast<void*>(&__quantum__qis__rxx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__u__body",
      reinterpret_cast<void*>(&__quantum__qis__u__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__u3__body",
      reinterpret_cast<void*>(&__quantum__qis__u3__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__u2__body",
      reinterpret_cast<void*>(&__quantum__qis__u2__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__u1__body",
      reinterpret_cast<void*>(&__quantum__qis__u1__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cu1__body",
      reinterpret_cast<void*>(&__quantum__qis__cu1__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cu3__body",
      reinterpret_cast<void*>(&__quantum__qis__cu3__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cnot__body",
      reinterpret_cast<void*>(&__quantum__qis__cnot__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cx__body",
      reinterpret_cast<void*>(&__quantum__qis__cx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cy__body",
      reinterpret_cast<void*>(&__quantum__qis__cy__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cz__body",
      reinterpret_cast<void*>(&__quantum__qis__cz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__ch__body",
      reinterpret_cast<void*>(&__quantum__qis__ch__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__swap__body",
      reinterpret_cast<void*>(&__quantum__qis__swap__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cswap__body",
      reinterpret_cast<void*>(&__quantum__qis__cswap__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__crz__body",
      reinterpret_cast<void*>(&__quantum__qis__crz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cry__body",
      reinterpret_cast<void*>(&__quantum__qis__cry__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__crx__body",
      reinterpret_cast<void*>(&__quantum__qis__crx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__cp__body",
      reinterpret_cast<void*>(&__quantum__qis__cp__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__rzz__body",
      reinterpret_cast<void*>(&__quantum__qis__rzz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__ccx__body",
      reinterpret_cast<void*>(&__quantum__qis__ccx__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__ccz__body",
      reinterpret_cast<void*>(&__quantum__qis__ccz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__m__body",
      reinterpret_cast<void*>(&__quantum__qis__m__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__measure__body",
      reinterpret_cast<void*>(&__quantum__qis__measure__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__mz__body",
      reinterpret_cast<void*>(&__quantum__qis__mz__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__qis__reset__body",
      reinterpret_cast<void*>(&__quantum__qis__reset__body));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__initialize",
      reinterpret_cast<void*>(&__quantum__rt__initialize));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__read_result",
      reinterpret_cast<void*>(&__quantum__rt__read_result));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__tuple_record_output",
      reinterpret_cast<void*>(&__quantum__rt__tuple_record_output));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__array_record_output",
      reinterpret_cast<void*>(&__quantum__rt__array_record_output));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__result_record_output",
      reinterpret_cast<void*>(&__quantum__rt__result_record_output));
  llvm::sys::DynamicLibrary::AddSymbol(
      "__quantum__rt__bool_record_output",
      reinterpret_cast<void*>(&__quantum__rt__bool_record_output));

  return runOrcJIT();
}
