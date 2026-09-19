---
file_format: mystnb
kernelspec:
  name: python3
---

# C++ libraries and API reference

The <a href="cpp/index.html">native C++ API reference</a> documents the
installed public headers, including decision diagrams, benchmarks, and QDMI. It
is generated from the same source revision as this guide.

## Use the DD library

Create a two-qubit GHZ state, which is a Bell state, and print its amplitudes.
Save this as `main.cpp`:

```cpp
#include "dd/Package.hpp"
#include "dd/StateGeneration.hpp"

#include <iostream>
#include <memory>
#include <variant>

int main() {
  auto created = dd::Package::create(2);
  if (const auto* error = std::get_if<dd::Error>(&created)) {
    std::cerr << error->message << '\n';
    return 1;
  }
  auto& package = *std::get<std::unique_ptr<dd::Package>>(created);
  auto generated = dd::makeGHZState(2, package);
  if (const auto* error = std::get_if<dd::Error>(&generated)) {
    std::cerr << error->message << '\n';
    return 1;
  }
  const auto state = std::get<dd::VectorDD>(generated);
  for (const auto amplitude : state.getVector()) {
    std::cout << amplitude << '\n';
  }
  if (const auto error = package.decRef(state)) {
    std::cerr << error->message << '\n';
    return 1;
  }
}
```

The DD package owns its nodes. The state returned by `makeGHZState` holds a
reference until `decRef` releases it. Keep the package alive while using the
state. Converting a DD to a dense vector takes space exponential in the number
of qubits; this example has only four amplitudes.

Save the following as `CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.28)
project(dd-example LANGUAGES CXX)
find_package(mqt-core CONFIG REQUIRED)
add_executable(dd-example main.cpp)
target_link_libraries(dd-example PRIVATE MQT::CoreDD)
```

With the Python wheel installed in the active environment, CMake and Ninja
available, run:

```console
cmake -S . -B build -G Ninja -DCMAKE_PREFIX_PATH="$(mqt-core-cli --cmake_dir)"
cmake --build build
./build/dd-example
```

The last command is for a Unix shell; on Windows, run `build\dd-example.exe`.
The output has amplitude $1/\sqrt{2}$ at `00` and `11`, and zero at `01` and
`10`. For a separate C++ installation, point `CMAKE_PREFIX_PATH` at its install
prefix instead. See {doc}`installation` for source builds and other CMake
integration options.

```{code-cell} ipython3
:tags: [remove-cell]
from pathlib import Path
from tempfile import TemporaryDirectory
import math
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
    executable = root / "build" / ("dd-example.exe" if os.name == "nt" else "dd-example")
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    amplitudes = [complex(*map(float, line.strip("()").split(","))) for line in result.stdout.splitlines()]
    expected = [1 / math.sqrt(2), 0, 0, 1 / math.sqrt(2)]
    assert len(amplitudes) == len(expected)
    assert all(abs(actual - ideal) < 1e-6 for actual, ideal in zip(amplitudes, expected))
```

## Handle native errors

The C++20 DD, benchmark and QDMI libraries return `Result<T>`, an alias for
`std::variant<T, Error>` in each library's namespace. Operations with no value
return `std::optional<Error>`; an empty optional means success. Check errors
before taking the value. An optional property can therefore return a value, an
empty optional for an unsupported property, or an error for a failed query.

Fallible construction uses `create(...)`. For example, `dd::Package::create`
returns an owned package, `mqt::bench::Grover::create` returns a validated
benchmark, and `qdmi::Session::create` returns a session. QDMI errors retain the
provider's status code. DD and benchmark errors carry a category and message.
Python constructors and methods translate these errors into Python exceptions.
Invalid DD arguments, including insufficient package capacity and conflicting
controls, raise `ValueError`. QDMI provider failures retain their category
through the compiler adapter: an unsupported operation raises `RuntimeError`
from both `submit_job` and `submit_program`.

LLVM-facing services use `llvm::Expected<T>` and `llvm::Error`; MLIR passes and
interpreters use `LogicalResult` or `FailureOr<T>` with diagnostics. Callers
must consume each LLVM error. The native result contracts cover invalid inputs
and reported operational failures. They do not provide recovery from allocation
exhaustion or unexpected exceptions in system libraries or dependencies.

## Extend the compiler or QIR runtime

The MLIR compiler and QIR runtime use source-tree C++ interfaces. They are not
part of the installed public-header reference above. See
[device compilation from C++](mlir/target_compilation.md#c-source-tree-api), the
{doc}`QIR runtime guide <qir/index>`, and the
{doc}`MLIR dialect and pass references <mlir/index>`.
