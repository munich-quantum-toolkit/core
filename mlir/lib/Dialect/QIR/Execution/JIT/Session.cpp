/*
 * Copyright (c) 2023 - 2026 Chair for Design Automation, TUM
 * Copyright (c) 2025 - 2026 Munich Quantum Software Company GmbH
 * All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Licensed under the MIT License
 */

#include "mlir/Dialect/QIR/Execution/JIT/Session.h"

#include "mlir/Dialect/QIR/Execution/JIT/IRRewriter.h"
#include "mlir/Dialect/QIR/Execution/Runtime/QIR.h"
#include "mlir/Dialect/QIR/Execution/Runtime/Runtime.h"
#include "mlir/Dialect/QIR/QIRDefinitions.h"

#include <llvm/ADT/ScopeExit.h>
#include <llvm/ADT/StringRef.h>
#include <llvm/CodeGen/CommandFlags.h>
#include <llvm/ExecutionEngine/JITEventListener.h>
#include <llvm/ExecutionEngine/JITSymbol.h>
#include <llvm/ExecutionEngine/Orc/AbsoluteSymbols.h>
#include <llvm/ExecutionEngine/Orc/Core.h>
#include <llvm/ExecutionEngine/Orc/CoreContainers.h>
#include <llvm/ExecutionEngine/Orc/Debugging/DebuggerSupport.h>
#include <llvm/ExecutionEngine/Orc/JITTargetMachineBuilder.h>
#include <llvm/ExecutionEngine/Orc/LLJIT.h>
#include <llvm/ExecutionEngine/Orc/RTDyldObjectLinkingLayer.h>
#include <llvm/ExecutionEngine/Orc/ThreadSafeModule.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/Casting.h>
#include <llvm/Support/Debug.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/ErrorHandling.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/TargetParser/Triple.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#define DEBUG_TYPE "mqt-core-qir-jit"

namespace qir {

static auto isEntryPoint(const llvm::Function& function) -> bool {
  return function.hasFnAttribute(ENTRY_POINT_ATTR);
}

static auto selectEntryPoint(llvm::Module& module) -> llvm::Function& {
  llvm::Function* selected = nullptr;
  for (auto& function : module) {
    if (!function.isDeclaration() && isEntryPoint(function)) {
      if (selected != nullptr) {
        throw std::runtime_error("Multiple QIR entry points were found");
      }
      selected = &function;
    }
  }
  if (selected == nullptr) {
    throw std::runtime_error("No QIR entry point was found");
  }
  auto& entryPoint = *selected;
  const auto* type = entryPoint.getFunctionType();
  if (type->isVarArg() || type->getNumParams() != 0 ||
      !type->getReturnType()->isIntegerTy(64)) {
    std::string actual;
    llvm::raw_string_ostream stream(actual);
    type->print(stream);
    throw std::runtime_error("QIR entry point '" + entryPoint.getName().str() +
                             "' must have type i64 (), but has type " + actual);
  }
  return entryPoint;
}

static auto readOutputSchema(const llvm::Function& entryPoint)
    -> Runtime::OutputSchema {
  if (const auto attr = entryPoint.getFnAttribute(OUTPUT_LABELING_SCHEMA_ATTR);
      attr.isValid() && attr.getValueAsString().compare(ORDERED_SCHEMA) == 0) {
    return Runtime::OutputSchema::Ordered;
  }
  return Runtime::OutputSchema::Labeled;
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
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE ": " << errMsg << "\n");
  }
  return llvm::Error::success();
}

namespace {

enum class AbiType : uint8_t { Void, Pointer, I1, I32, I64, F64 };

struct RuntimeSymbol {
  AbiType result{};
  std::vector<AbiType> parameters;
  void* address{};
};

using RuntimeRegistry = std::unordered_map<std::string, RuntimeSymbol>;

} // namespace

static auto abiTypeName(const AbiType type) -> std::string_view {
  switch (type) {
  case AbiType::Void:
    return "void";
  case AbiType::Pointer:
    return "ptr";
  case AbiType::I1:
    return "i1";
  case AbiType::I32:
    return "i32";
  case AbiType::I64:
    return "i64";
  case AbiType::F64:
    return "double";
  }
  llvm_unreachable("unknown QIR ABI type");
}

static auto matches(const llvm::Type* actual, const AbiType expected) -> bool {
  switch (expected) {
  case AbiType::Void:
    return actual->isVoidTy();
  case AbiType::Pointer:
    return actual->isPointerTy();
  case AbiType::I1:
    return actual->isIntegerTy(1);
  case AbiType::I32:
    return actual->isIntegerTy(32);
  case AbiType::I64:
    return actual->isIntegerTy(64);
  case AbiType::F64:
    return actual->isDoubleTy();
  }
  llvm_unreachable("unknown QIR ABI type");
}

static auto matches(const llvm::FunctionType& actual,
                    const RuntimeSymbol& expected) -> bool {
  if (actual.isVarArg() || !matches(actual.getReturnType(), expected.result) ||
      actual.getNumParams() != expected.parameters.size()) {
    return false;
  }
  return std::ranges::equal(actual.params(), expected.parameters,
                            [](const llvm::Type* type, const AbiType abiType) {
                              return matches(type, abiType);
                            });
}

static auto describe(const RuntimeSymbol& symbol) -> std::string {
  std::ostringstream description;
  description << abiTypeName(symbol.result) << " (";
  for (std::size_t i = 0; i < symbol.parameters.size(); ++i) {
    if (i != 0) {
      description << ", ";
    }
    description << abiTypeName(symbol.parameters[i]);
  }
  description << ")";
  return description.str();
}

template <typename Function>
static auto addSymbol(RuntimeRegistry& registry, const std::string_view name,
                      const AbiType result,
                      std::initializer_list<AbiType> parameters,
                      Function* function) -> void {
  registry.insert_or_assign(std::string(name),
                            RuntimeSymbol{
                                .result = result,
                                .parameters = parameters,
                                .address = reinterpret_cast<void*>(function),
                            });
}

template <typename Function>
static auto addGate(RuntimeRegistry& registry, const std::string_view name,
                    const std::size_t targetCount,
                    const std::size_t parameterCount,
                    const std::size_t controlCount, Function* function)
    -> void {
  std::vector<AbiType> parameters(parameterCount, AbiType::F64);
  parameters.insert(parameters.end(), controlCount + targetCount,
                    AbiType::Pointer);
  registry.insert_or_assign(std::string(name),
                            RuntimeSymbol{
                                .result = AbiType::Void,
                                .parameters = std::move(parameters),
                                .address = reinterpret_cast<void*>(function),
                            });
}

static auto createRuntimeRegistry() -> RuntimeRegistry {
  RuntimeRegistry registry;

#define MQT_QIR_ADD_GATE(NAME, SUFFIX, CTL_SUFFIX, TARGETS, PARAMS)            \
  addGate(registry, "__quantum__qis__" #NAME "__" #SUFFIX, TARGETS, PARAMS, 0, \
          &__quantum__qis__##NAME##__##SUFFIX);                                \
  addGate(registry, "__quantum__qis__c" #NAME "__" #SUFFIX, TARGETS, PARAMS,   \
          1, &__quantum__qis__c##NAME##__##SUFFIX);                            \
  addGate(registry, "__quantum__qis__cc" #NAME "__" #SUFFIX, TARGETS, PARAMS,  \
          2, &__quantum__qis__cc##NAME##__##SUFFIX);                           \
  addSymbol(registry, "__quantum__qis__" #NAME "__" #CTL_SUFFIX,               \
            AbiType::Void, {AbiType::Pointer, AbiType::Pointer},               \
            &__quantum__qis__##NAME##__##CTL_SUFFIX);
#define MQT_GATE(KEY, NAME, GETTER, TARGETS, PARAMS, SUFFIX, CTL_SUFFIX)       \
  MQT_QIR_ADD_GATE(NAME, SUFFIX, CTL_SUFFIX, TARGETS, PARAMS)
#include "mlir/Conversion/GateTable.def"
#undef MQT_QIR_ADD_GATE

  addSymbol(registry, "__quantum__qis__gphase__body", AbiType::Void,
            {AbiType::F64}, &__quantum__qis__gphase__body);
  addSymbol(registry, "__quantum__qis__cnot__body", AbiType::Void,
            {AbiType::Pointer, AbiType::Pointer}, &__quantum__qis__cnot__body);
  addSymbol(registry, "__quantum__qis__mz__body", AbiType::Void,
            {AbiType::Pointer, AbiType::Pointer}, &__quantum__qis__mz__body);
  addSymbol(registry, "__quantum__qis__reset__body", AbiType::Void,
            {AbiType::Pointer}, &__quantum__qis__reset__body);
  addSymbol(registry, "__quantum__rt__array_create_1d", AbiType::Pointer,
            {AbiType::I32, AbiType::I64}, &__quantum__rt__array_create_1d);
  addSymbol(registry, "__quantum__rt__array_get_size_1d", AbiType::I64,
            {AbiType::Pointer}, &__quantum__rt__array_get_size_1d);
  addSymbol(registry, "__quantum__rt__array_get_element_ptr_1d",
            AbiType::Pointer, {AbiType::Pointer, AbiType::I64},
            &__quantum__rt__array_get_element_ptr_1d);
  addSymbol(registry, "__quantum__rt__array_update_reference_count",
            AbiType::Void, {AbiType::Pointer, AbiType::I32},
            &__quantum__rt__array_update_reference_count);
  addSymbol(registry, "__quantum__rt__tuple_create", AbiType::Pointer,
            {AbiType::I64}, &__quantum__rt__tuple_create);
  addSymbol(registry, "__quantum__rt__tuple_update_reference_count",
            AbiType::Void, {AbiType::Pointer, AbiType::I32},
            &__quantum__rt__tuple_update_reference_count);
  addSymbol(registry, "__quantum__rt__qubit_allocate", AbiType::Pointer,
            {AbiType::Pointer}, &__quantum__rt__qubit_allocate);
  addSymbol(registry, "__quantum__rt__qubit_release", AbiType::Void,
            {AbiType::Pointer}, &__quantum__rt__qubit_release);
  addSymbol(registry, "__quantum__rt__qubit_array_allocate", AbiType::Void,
            {AbiType::I64, AbiType::Pointer, AbiType::Pointer},
            &__quantum__rt__qubit_array_allocate);
  addSymbol(registry, "__quantum__rt__qubit_array_release", AbiType::Void,
            {AbiType::I64, AbiType::Pointer},
            &__quantum__rt__qubit_array_release);
  addSymbol(registry, "__quantum__rt__result_allocate", AbiType::Pointer,
            {AbiType::Pointer}, &__quantum__rt__result_allocate);
  addSymbol(registry, "__quantum__rt__result_release", AbiType::Void,
            {AbiType::Pointer}, &__quantum__rt__result_release);
  addSymbol(registry, "__quantum__rt__result_array_allocate", AbiType::Void,
            {AbiType::I64, AbiType::Pointer, AbiType::Pointer},
            &__quantum__rt__result_array_allocate);
  addSymbol(registry, "__quantum__rt__result_array_release", AbiType::Void,
            {AbiType::I64, AbiType::Pointer},
            &__quantum__rt__result_array_release);

  addSymbol(registry, "__quantum__rt__initialize", AbiType::Void,
            {AbiType::Pointer}, &__quantum__rt__initialize);
  addSymbol(registry, "__quantum__rt__read_result", AbiType::I1,
            {AbiType::Pointer}, &__quantum__rt__read_result);
  addSymbol(registry, "__quantum__rt__result_record_output", AbiType::Void,
            {AbiType::Pointer, AbiType::Pointer},
            &__quantum__rt__result_record_output);
  addSymbol(registry, "__quantum__rt__bool_record_output", AbiType::Void,
            {AbiType::I1, AbiType::Pointer},
            &__quantum__rt__bool_record_output);
  addSymbol(registry, "__quantum__rt__int_record_output", AbiType::Void,
            {AbiType::I64, AbiType::Pointer},
            &__quantum__rt__int_record_output);
  addSymbol(registry, "__quantum__rt__double_record_output", AbiType::Void,
            {AbiType::F64, AbiType::Pointer},
            &__quantum__rt__double_record_output);
  addSymbol(registry, "__quantum__rt__tuple_record_output", AbiType::Void,
            {AbiType::I64, AbiType::Pointer},
            &__quantum__rt__tuple_record_output);
  addSymbol(registry, "__quantum__rt__array_record_output", AbiType::Void,
            {AbiType::I64, AbiType::Pointer},
            &__quantum__rt__array_record_output);
  addSymbol(registry, "__quantum__rt__result_array_record_output",
            AbiType::Void, {AbiType::I64, AbiType::Pointer, AbiType::Pointer},
            &__quantum__rt__result_array_record_output);
  return registry;
}

static auto selectRuntimeSymbols(const llvm::Module& module)
    -> std::vector<std::pair<std::string, void*>> {
  static const auto REGISTRY = createRuntimeRegistry();
  std::vector<std::pair<std::string, void*>> selected;
  for (const auto& function : module) {
    if (!function.isDeclaration() || function.use_empty() ||
        !function.getName().starts_with("__quantum__")) {
      continue;
    }
    const auto it = REGISTRY.find(function.getName().str());
    if (it == REGISTRY.end()) {
      throw std::runtime_error("Unsupported QIR runtime declaration '" +
                               function.getName().str() + "'");
    }
    const auto& symbol = it->second;
    if (!matches(*function.getFunctionType(), symbol)) {
      std::string actual;
      llvm::raw_string_ostream stream(actual);
      function.getFunctionType()->print(stream);
      std::ostringstream message;
      message << "QIR declaration '" << function.getName().str()
              << "' has unsupported type " << actual << "; expected "
              << describe(symbol);
      throw std::runtime_error(message.str());
    }
    selected.emplace_back(function.getName().str(), symbol.address);
  }
  return selected;
}

llvm::Expected<llvm::orc::ThreadSafeModule>
JitSession::loadModuleFromMemory(const llvm::StringRef irBytes,
                                 const llvm::StringRef bufferName) {
  llvm::orc::ThreadSafeContext context{std::make_unique<llvm::LLVMContext>()};
  llvm::SMDiagnostic err;
  auto buffer = llvm::MemoryBuffer::getMemBuffer(
      irBytes, bufferName,
      /*RequiresNullTerminator=*/false); // bitcode isn't null-terminated
  auto m = context.withContextDo([&](llvm::LLVMContext* ctx) {
    return parseIR(buffer->getMemBufferRef(), err, *ctx);
  });
  if (!m) {
    std::string message;
    llvm::raw_string_ostream stream(message);
    err.print(DEBUG_TYPE, stream);
    return llvm::make_error<llvm::StringError>(std::move(message),
                                               llvm::inconvertibleErrorCode());
  }
  return llvm::orc::ThreadSafeModule(std::move(m), std::move(context));
}

JitSession::JitSession(const llvm::StringRef irBytes,
                       const llvm::StringRef bufferName,
                       const Execution execution,
                       std::optional<uint64_t> randomSeed)
    : runtime_(std::make_unique<Runtime>(
          randomSeed ? *randomSeed : Runtime::generateRandomSeed())),
      execution_(execution) {
  initialize(loadModuleFromMemory(irBytes, bufferName), execution);
}

JitSession::~JitSession() { deinitialize(); }

int64_t JitSession::run() {
  auto* previous = Runtime::bind(runtime_.get());
  const auto restoreRuntime =
      llvm::make_scope_exit([previous] { Runtime::bind(previous); });
  return entryPointFn_();
}

int64_t JitSession::sample(size_t shots, std::vector<std::string>& results) {
  if (execution_ != Execution::Sampling) {
    throw std::logic_error("Cannot sample a QIR state-extraction session");
  }
  results.clear();
  results.reserve(shots);
  runtime_->outputProgramHeader();
  const auto execute = [&] {
    if (!initializesRuntime_) {
      runtime_->reset();
    }
    runtime_->outputShotStart();
    const auto code = run();
    runtime_->outputShotEnd(code);
    return code;
  };
  if (samplingOutputs_ && !runtime_->hasOutput() && shots != 0) {
    runtime_->deferMeasurements_ = true;
    const auto restore =
        llvm::make_scope_exit([&] { runtime_->deferMeasurements_ = false; });
    if (const auto code = execute(); code != 0) {
      return code;
    }
    for (size_t i = 0; i < shots; ++i) {
      results.push_back(runtime_->sampleMeasurements(*samplingOutputs_));
    }
    return 0;
  }
  for (size_t i = 0; i < shots; ++i) {
    if (const auto code = execute(); code != 0) {
      return code;
    }
    results.push_back(runtime_->getMeasurements());
  }
  return 0;
}

auto JitSession::runtime() -> Runtime& { return *runtime_; }

void JitSession::initNativeTargets() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    static const llvm::codegen::RegisterCodeGenFlags CGF;

    // If we have a native target, initialize it to ensure it is linked in and
    // usable by the JIT.
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();
    llvm::InitializeNativeTargetAsmParser();
  });
}

static std::optional<size_t>
readStaticCapacity(const llvm::Module& llvmModule,
                   const llvm::Function& entryPoint, llvm::StringRef flagName,
                   llvm::StringRef attributeName) {
  if (auto* flag = llvmModule.getModuleFlag(flagName)) {
    const auto* dynamic = llvm::mdconst::dyn_extract<llvm::ConstantInt>(flag);
    if (dynamic == nullptr || dynamic->getValue().getLimitedValue() > 1) {
      throw std::invalid_argument("Invalid QIR resource flag '" +
                                  flagName.str() + "'");
    }
    if (!dynamic->isZero()) {
      return std::nullopt;
    }
  }
  const auto attribute = entryPoint.getFnAttribute(attributeName);
  if (!attribute.isValid()) {
    return std::nullopt;
  }
  uint64_t capacity = 0;
  if (attribute.getValueAsString().getAsInteger(10, capacity) ||
      capacity > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument("Invalid QIR resource capacity '" +
                                attributeName.str() + "'");
  }
  return static_cast<size_t>(capacity);
}

void JitSession::initialize(
    llvm::Expected<llvm::orc::ThreadSafeModule> llvmModule,
    const Execution execution) {
  if (!llvmModule) {
    throw std::runtime_error(llvm::toString(llvmModule.takeError()));
  }
  auto loadedModule = std::move(*llvmModule);

  std::string entryPointName;
  std::vector<std::pair<std::string, void*>> runtimeSymbols;
  loadedModule.withModuleDo([&](llvm::Module& module) {
    auto& entryPoint = selectEntryPoint(module);
    entryPointName = entryPoint.getName().str();
    runtime_->setOutputSchema(readOutputSchema(entryPoint));
    std::vector<std::pair<std::string, std::string>> metadata;
    for (const auto attribute : entryPoint.getAttributes().getFnAttrs()) {
      if (attribute.isStringAttribute()) {
        metadata.emplace_back(attribute.getKindAsString().str(),
                              attribute.getValueAsString().str());
      }
    }
    runtime_->setMetadata(std::move(metadata));
    if (execution == Execution::StateExtraction) {
      prepareForStateExtraction(entryPoint);
    }
    runtimeSymbols = selectRuntimeSymbols(module);
    runtime_->configureStaticResources(
        readStaticCapacity(module, entryPoint, "dynamic_qubit_management",
                           "required_num_qubits"),
        readStaticCapacity(module, entryPoint, "dynamic_result_management",
                           "required_num_results"));
    const auto* first =
        llvm::dyn_cast<llvm::CallInst>(&entryPoint.getEntryBlock().front());
    initializesRuntime_ =
        first != nullptr && first->getCalledFunction() != nullptr &&
        first->getCalledFunction()->isDeclaration() &&
        first->getCalledFunction()->getName() == "__quantum__rt__initialize";
    if (execution == Execution::Sampling) {
      samplingOutputs_ = getStaticSamplingOutputs(entryPoint);
    }
  });
  initNativeTargets();

  // Get TargetTriple and DataLayout from the main module if they're explicitly
  // set.
  std::optional<llvm::Triple> tt;
  std::optional<llvm::DataLayout> dl;
  loadedModule.withModuleDo([&](llvm::Module& m) {
    if (!m.getTargetTriple().empty()) {
      tt = m.getTargetTriple();
    }
    if (!m.getDataLayout().isDefault()) {
      dl = m.getDataLayout();
    }
  });

  /// Use the ordinary module JIT; no function-lazy modules are submitted.
  llvm::orc::LLJITBuilder builder;

  // Use the module's target triple if set, otherwise detect the host's.
  auto host = llvm::orc::JITTargetMachineBuilder::detectHost();
  if (!host) {
    throw std::runtime_error(llvm::toString(host.takeError()));
  }
  if (tt) {
    if (tt->getArch() != host->getTargetTriple().getArch() ||
        tt->getOS() != host->getTargetTriple().getOS()) {
      throw std::invalid_argument(
          "QIR target triple must match the execution host");
    }
    host->getTargetTriple() = *tt;
  }
  builder.setJITTargetMachineBuilder(*host);

  /// Apply the module's explicit data layout if present.
  if (dl) {
    builder.setDataLayout(dl);
  }

  // Optional architecture override from the -march codegen flag.
  if (!llvm::codegen::getMArch().empty()) {
    builder.getJITTargetMachineBuilder()->getTargetTriple().setArchName(
        llvm::codegen::getMArch());
  }

  // Apply CPU, features, relocation model, and code model from codegen flags.
  if (const auto cpu = llvm::codegen::getCPUStr(); !cpu.empty()) {
    builder.getJITTargetMachineBuilder()->setCPU(cpu);
  }
  builder.getJITTargetMachineBuilder()
      ->addFeatures(llvm::codegen::getFeatureList())
      .setRelocationModel(llvm::codegen::getExplicitRelocModel())
      .setCodeModel(llvm::codegen::getExplicitCodeModel());

  // Enable debugging of JIT'd code (only works on JITLink for ELF and MachO).
  builder.setPrePlatformSetup(tryEnableDebugSupport);

  // Build the JIT.
  auto expectedJit = builder.create();
  if (!expectedJit) {
    throw std::runtime_error(llvm::toString(expectedJit.takeError()));
  }
  jit_ = std::move(*expectedJit);

  // Register QIR runtime symbols.
  auto& jd = jit_->getMainJITDylib();
  llvm::orc::SymbolMap hostSymbols;
  for (const auto& [name, ptr] : runtimeSymbols) {
    hostSymbols[jit_->mangleAndIntern(name)] = {
        llvm::orc::ExecutorAddr::fromPtr(ptr), llvm::JITSymbolFlags::Exported};
  }
  if (auto err = jd.define(llvm::orc::absoluteSymbols(hostSymbols))) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // GDB listener (no error path)
  auto* objLayer = &jit_->getObjLinkingLayer();
  if (auto* rtDyldObjLayer =
          dyn_cast<llvm::orc::RTDyldObjectLinkingLayer>(objLayer)) {
    rtDyldObjLayer->registerJITEventListener(
        *llvm::JITEventListener::createGDBRegistrationListener());
  }

  // If this is a Mingw or Cygwin executor then we need to alias __main to
  // orc_rt_int_void_return_0.
  if (jit_->getTargetTriple().isOSCygMing()) {
    auto& workaroundJD = jit_->getProcessSymbolsJITDylib()
                             ? *jit_->getProcessSymbolsJITDylib()
                             : jit_->getMainJITDylib();
    if (auto err = workaroundJD.define(llvm::orc::absoluteSymbols({
            {
                jit_->mangleAndIntern("__main"),
                {
                    llvm::orc::ExecutorAddr::fromPtr(mingwNoopMain),
                    llvm::JITSymbolFlags::Exported,
                },
            },
        }))) {
      throw std::runtime_error(llvm::toString(std::move(err)));
    }
  }

  if (auto err = jit_->addIRModule(std::move(loadedModule))) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // Run any static constructors.
  if (auto err = jit_->initialize(jit_->getMainJITDylib())) {
    throw std::runtime_error(llvm::toString(std::move(err)));
  }

  // Resolve the selected QIR entry point.
  auto entryPointAddress = jit_->lookup(entryPointName);
  if (!entryPointAddress) {
    throw std::runtime_error(llvm::toString(entryPointAddress.takeError()));
  }
  entryPointFn_ = entryPointAddress->toPtr<EntryPointFn*>();
}

void JitSession::deinitialize() const {
  if (!jit_) {
    return;
  }
  if (auto err = jit_->deinitialize(jit_->getMainJITDylib())) {
    llvm::errs() << "JitSession deinitialize failed: "
                 << llvm::toString(std::move(err)) << "\n";
  }
}

} // namespace qir
