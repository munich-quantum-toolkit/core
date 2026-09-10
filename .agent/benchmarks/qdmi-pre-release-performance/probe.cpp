#include "mqt/Compiler/QDMIAdapter.h"
#include "mqt/Compiler/Target.h"
#include "qdmi/Client.hpp"
#include "qdmi/driver/Driver.hpp"

#include "llvm/Support/Error.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <new>
#include <string>
#include <vector>
static thread_local bool measuring = false;
static thread_local size_t calls = 0, bytes = 0;
void* operator new(size_t n) {
  if (measuring) {
    ++calls;
    bytes += n;
  }
  if (auto* p = std::malloc(n ? n : 1))
    return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, size_t) noexcept { std::free(p); }
static auto siteQuery =
    static_cast<decltype(QDMI_device_session_query_site_property)*>(nullptr);
static auto operationQuery =
    static_cast<decltype(QDMI_device_session_query_operation_property)*>(
        nullptr);
static size_t siteCalls = 0, operationCalls = 0;
template <class Fn> void measure(const std::string& label, Fn fn) {
  std::vector<double> times;
  for (int i = 0; i < 5; ++i) {
    calls = 0;
    bytes = 0;
    siteCalls = 0;
    operationCalls = 0;
    auto start = std::chrono::steady_clock::now();
    measuring = true;
    fn();
    measuring = false;
    times.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - start)
                        .count());
  }
  std::ranges::sort(times);
  std::cout << label << " median_ms=" << times[2] << " allocations=" << calls
            << " allocated_bytes=" << bytes << " site_queries=" << siteCalls
            << " min_ms=" << times.front() << " max_ms=" << times.back()
            << " samples_ms=";
  for (auto sample : times)
    std::cout << sample << ",";
  std::cout << " operation_queries=" << operationCalls << '\n';
}
int main() {
  for (const auto& name : {std::string("ddsim"), std::string("sc")}) {
    auto library = std::make_shared<qdmi::DynamicDeviceLibrary>(
        name == "ddsim" ? MQT_CORE_MLIR_DDSIM_DEVICE_LIBRARY
                        : MQT_CORE_MLIR_SC_DEVICE_LIBRARY,
        name == "ddsim" ? "MQT_DDSIM" : "MQT_SC");
    siteQuery = library->device_session_query_site_property;
    operationQuery = library->device_session_query_operation_property;
    library->device_session_query_site_property =
        [](QDMI_Device_Session s, QDMI_Site site, QDMI_Site_Property p,
           size_t n, void* value, size_t* returned) {
          ++siteCalls;
          return siteQuery(s, site, p, n, value, returned);
        };
    library->device_session_query_operation_property =
        [](QDMI_Device_Session s, QDMI_Operation op, size_t ns,
           const QDMI_Site* sites, size_t np, const double* params,
           QDMI_Operation_Property p, size_t n, void* value, size_t* returned) {
          ++operationCalls;
          return operationQuery(s, op, ns, sites, np, params, p, n, value,
                                returned);
        };
    QDMI_Device_impl_d raw(library);
    auto device = qdmi::Session::createSessionlessDevice(&raw);
    auto sites = device.getSites();
    auto ops = device.getOperations();
    size_t sum = 0;
    measure(name + "_index_100k", [&] {
      for (size_t i = 0; i < 100000; ++i)
        sum += sites[0].getIndex();
    });
    measure(name + "_snapshot", [&] {
      auto target = llvm::cantFail(mlir::compilerTargetFromDevice(device));
      sum += target.numSites();
    });
    std::cout << "checksum=" << sum << '\n';
  }
  const auto payload = llvm::cantFail(
      mlir::payloadSpecificationForProgramFormat(QDMI_PROGRAM_FORMAT_QASM3));
  for (size_t n : {1000, 10000, 100000}) {
    std::vector<mlir::CompilerTarget::SiteTuple> tuples;
    for (size_t i = 0; i < n; ++i)
      tuples.push_back(llvm::cantFail(
          mlir::CompilerTarget::SiteTuple::create({static_cast<int64_t>(i)})));
    std::vector<mlir::CompilerTarget::Operation> ops;
    ops.push_back(llvm::cantFail(mlir::CompilerTarget::Operation::create(
        "rx", 1, 1, std::move(tuples))));
    measure("operation_factory_" + std::to_string(n), [&] {
      auto copy = mlir::CompilerTarget::NativeOperations::fromOperations(ops);
      if (copy.operations().size() != 1)
        std::abort();
    });
  }
}
