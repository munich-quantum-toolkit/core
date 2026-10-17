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

This application allocates a session through the builtin driver's C interface.
Save it as `main.cpp`:

```cpp
#include <qdmi/client.h>

#include <iostream>

int main() {
  QDMI_Session session = nullptr;
  if (QDMI_session_alloc(&session) != QDMI_SUCCESS) {
    return 1;
  }
  QDMI_session_free(session);
  std::cout << "QDMI driver available\n";
}
```

Save the following as `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.28)
project(qdmi-example LANGUAGES CXX)
find_package(mqt-core CONFIG REQUIRED COMPONENTS Runtime)
add_executable(qdmi-example main.cpp)
target_link_libraries(qdmi-example PRIVATE MQT::CoreQDMIDriver)
mqt_copy_qdmi_runtime(qdmi-example MQT::CoreQDMI_DDSIM_Device)
```

With the wheel installed in the active environment, run:

```console
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH="$(mqt-core-cli --cmake_dir)"
cmake --build build
./build/qdmi-example
```

On Windows, run `build\qdmi-example.exe`. The application prints
`QDMI driver available`. No LLVM/MLIR installation is required.

CMake also provides `MQT::mqt-cc` and `MQT::mqt-core-bench` imported executable
targets. Use them in custom commands, or run `mqt-cc` and `mqt-core-bench` from
the environment's command line.

```{code-cell} ipython3
:tags: [remove-cell]
from pathlib import Path
from tempfile import TemporaryDirectory
import os
import subprocess

source = Path("cpp_api.md").read_text(encoding="utf-8")
cpp = source.split("```cpp\n")[1].split("\n```", 1)[0]
cmake = source.split("```cmake\n")[1].split("\n```", 1)[0]
prefix = subprocess.run(
    ["mqt-core-cli", "--cmake_dir"], check=True, capture_output=True, text=True
).stdout.strip()
with TemporaryDirectory() as directory:
    root = Path(directory)
    (root / "main.cpp").write_text(cpp, encoding="utf-8")
    (root / "CMakeLists.txt").write_text(cmake, encoding="utf-8")
    subprocess.run(
        ["cmake", "-S", directory, "-B", str(root / "build"), "-G", "Ninja",
         f"-DCMAKE_PREFIX_PATH={prefix}"], check=True, stderr=subprocess.STDOUT
    )
    subprocess.run(
        ["cmake", "--build", str(root / "build")], check=True, stderr=subprocess.STDOUT
    )
    executable = root / "build" / ("qdmi-example.exe" if os.name == "nt" else "qdmi-example")
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "QDMI driver available"
```

## Use the C++ development libraries

Build and install MQT Core from source, then request the `Development`
component. It provides `MQT::CoreDD`, `MQT::CoreQDMI`, and `MQT::CoreBench`. The
DD library is static; compile consumers with a compatible C++ toolchain.
Development consumers require an LLVM SDK matching the major and minor version
used to build MQT Core. CMake finds LLVM and propagates its headers and
libraries through the imported targets.

```cmake
find_package(mqt-core CONFIG REQUIRED COMPONENTS Development)
target_link_libraries(my-application PRIVATE MQT::CoreDD)
```

Point `CMAKE_PREFIX_PATH` at the source installation's prefix. See
{doc}`installation` for source builds and other CMake integration options.

## Handle native errors

Fallible operations return `llvm::FailureOr<T>` or `llvm::LogicalResult` for
status alone. Check `llvm::failed(result)` before dereferencing a value.
Infallible operations return ordinary values or `void`. An `optional<T>` can
represent successful absence, such as an unsupported optional QDMI property.
Borrowed results use pointers; keep their owner alive.

Use `create(...)` for fallible construction. Python reports invalid arguments as
`ValueError`, range errors as `IndexError`, and unsupported direct QDMI
operations as `RuntimeError`. Compiler target queries and submission failures
raise `ValueError`; device lookup and source parsing use their respective
exception categories.

Diagnostics carry the message, severity, error category, and original QDMI
status when applicable. Install a handler **before** calling the operation:

```cpp
#include "support/Diagnostics.hpp"

mqt::ScopedDiagnosticHandler handler([](const mqt::Diagnostic& diagnostic) {
  std::cerr << diagnostic.message << '\n';
  return llvm::success();
});
auto package = dd::Package::create(2);
if (llvm::failed(package)) {
  return 1;
}
```

Handlers run synchronously on their installing thread, newest first, and must
not throw. Success consumes a diagnostic; failure forwards it to the previous
handler. Unhandled diagnostics go to stderr. Install handlers on each worker
thread. Diagnostics emitted inside a handler start at the previous handler.

Compiler contexts created by Core forward MLIR diagnostics, including attached
notes and locations, to these handlers. A later MLIR handler can consume a
diagnostic before forwarding. Caller-owned contexts retain their handler policy.

Standalone QDMI driver and device libraries embed their own diagnostic runtime.
A caller's handler cannot capture messages from another runtime. QDMI C
interfaces transfer status codes; provider messages remain in local logging.
Client-side wrappers translate statuses to diagnostics and Python exceptions.
Builds that enable shared Core libraries retain their shared dependencies.

Native algorithms do not recover from allocation exhaustion or unexpected
dependency exceptions. QDMI session allocation reports memory exhaustion through
its C status. See [QIR runtime failures](qir/index.md#runtime-failures) for the
direct-execution contract.

## Extend the compiler or QIR runtime

The MLIR compiler and QIR runtime use source-tree C++ interfaces. They are not
part of the installed public-header reference above. See
[device compilation from C++](mlir/target_compilation.md#c-source-tree-api), the
{doc}`QIR runtime guide <qir/index>`, and the
{doc}`MLIR dialect and pass references <mlir/index>`.
