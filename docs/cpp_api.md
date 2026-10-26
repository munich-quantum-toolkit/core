---
file_format: mystnb
kernelspec:
  name: python3
---

# C++ libraries and API reference

The <a href="cpp/index.html">native C++ API reference</a> documents the
installed public headers, including decision diagrams, benchmarks, and QDMI. It
is generated from the same source revision as this guide.

## Use the wheel's native runtime

The Python wheel supplies the QDMI driver, device bundles, and the `mqt-cc` and
`mqt-core-bench` executables. Its CMake package exposes the QDMI C interfaces;
C++ DD, QDMI client, and benchmark libraries require a source installation.

This application runs one deterministic shot through DDSIM's worker using the
builtin driver's C interface. Its bundle contains only the DDSIM device. Save it
as `main.cpp`:

```cpp
#include "qdmi/client.h"

#include "llvm/Support/raw_ostream.h"

#include <cstddef>
#include <stdexcept>
#include <string>

void check(int status) {
  if (status != QDMI_SUCCESS) {
    throw std::runtime_error("QDMI error: " + std::to_string(status));
  }
}

int main() {
  QDMI_Session session = nullptr;
  check(QDMI_session_alloc(&session));
  check(QDMI_session_init(session));
  QDMI_Device device = nullptr;
  check(QDMI_session_query_session_property(session,
                                            QDMI_SESSION_PROPERTY_DEVICES,
                                            sizeof(device), &device, nullptr));

  QDMI_Job job = nullptr;
  check(QDMI_device_create_job(device, &job));
  const char circuit[] = "OPENQASM 3.0; include \"stdgates.inc\"; "
                         "bit[1] c; x $0; c[0] = measure $0;";
  const void* program = circuit;
  const size_t length = sizeof(circuit);
  const size_t shots = 1;
  check(QDMI_job_set_programs(job, QDMI_PROGRAM_FORMAT_QASM3, 1, &length,
                              &program));
  check(QDMI_job_set_parameter(job, QDMI_JOB_PARAMETER_SHOTSNUM, sizeof(shots),
                               &shots));
  check(QDMI_job_submit(job));
  check(QDMI_job_wait(job, 30000));

  char result[2] = {};
  check(QDMI_job_get_results(job, 0, QDMI_JOB_RESULT_SHOTS, sizeof(result),
                             result, nullptr));
  QDMI_job_free(job);
  QDMI_session_free(session);
  if (std::string(result) != "1") {
    return 1;
  }
  llvm::outs() << "DDSIM result: " << result << '\n';
}
```

Save the following as `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.28)
project(qdmi-example LANGUAGES CXX)
find_package(mqt-core CONFIG REQUIRED COMPONENTS Runtime)
add_executable(qdmi-example main.cpp)
target_compile_features(qdmi-example PRIVATE cxx_std_20)
target_link_libraries(qdmi-example PRIVATE MQT::CoreQDMIDriver LLVMSupport)
mqt_copy_qdmi_runtime(qdmi-example MQT::CoreQDMI_DDSIM_Device)

enable_testing()
add_test(NAME qdmi-runtime COMMAND qdmi-example)
```

With the wheel installed in the active environment and an LLVM SDK matching the
major and minor version used to build MQT Core, run:

```console
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$(mqt-core-cli --cmake_dir)"
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

The application prints `DDSIM result: 1`.

CMake also provides `MQT::mqt-cc` and `MQT::mqt-core-bench` imported executable
targets. Use them in custom commands, or run `mqt-cc` and `mqt-core-bench` from
the environment's command line.

```{code-cell} ipython3
:tags: [remove-cell]
import subprocess
import sys

subprocess.run([sys.executable, "../scripts/check_wheel_runtime.py"], check=True)
```

## Use the C++ development libraries

Build and install MQT Core from source, then request the `Development`
component. It provides `MQT::CoreDD`, `MQT::CoreQDMI`, and `MQT::CoreBench`. The
DD and QDMI C++ libraries are static; compile consumers with a compatible C++
toolchain. Development consumers require an LLVM SDK matching the major and
minor version used to build MQT Core. CMake finds LLVM and propagates its
headers and libraries through the imported targets.

```cmake
find_package(mqt-core CONFIG REQUIRED COMPONENTS Development)
target_link_libraries(my-application PRIVATE MQT::CoreDD)
```

Point `CMAKE_PREFIX_PATH` at the source installation's prefix. See
{doc}`installation` for source builds and other CMake integration options.

## Handle native errors

DD construction and numerical operations return `llvm::Expected<T>` or
`llvm::Error` on fallible paths. Check an `Expected` before accessing its value;
consume or propagate its `takeError()` on failure. An `Error` converts to true
on failure. Infallible operations return ordinary values or `void`.

Use `create(...)` for fallible construction:

```cpp
#include "dd/Package.hpp"

#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

auto package = dd::Package::create(2);
if (!package) {
  llvm::logAllUnhandledErrors(package.takeError(), llvm::errs());
  return 1;
}
```

Errors own their messages and categories. Bindings translate them to Python
exceptions. Internal memory exhaustion is outside the recoverable-error
contract.

## Extend the compiler or QIR runtime

The MLIR compiler and QIR runtime use source-tree C++ interfaces. They are not
part of the installed public-header reference above. See
[device compilation from C++](mlir/target_compilation.md#c-source-tree-api), the
{doc}`QIR runtime guide <qir/index>`, and the
{doc}`MLIR dialect and pass references <mlir/index>`.

## DD input and output

DD serializers, DOT exporters, statistics and number printers write to
`llvm::raw_ostream`. The caller owns the stream and checks any sink errors;
file-writing overloads report failures with exceptions. Use
`llvm::raw_string_ostream` to collect output in a `std::string`.

`Package::deserialize` reads borrowed `llvm::StringRef` bytes, including
embedded NUL bytes. Use `Package::deserializeFile` for a filename. Binary
serialization retains its native representation and is not portable between
architectures. Serialized decimal weights are locale-independent; finite
subnormal values are preserved.
