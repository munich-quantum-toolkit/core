# C++ include order

Status: complete; existing C++ lint findings are recorded below.

## Goal and scope

Apply issue #2420 to non-vendored C++ sources. The root `.clang-format` owns
include grouping: matching header, MQT Core, private or third-party headers,
upstream MLIR, LLVM, then system and standard-library headers. Library includes
use quotes unless the library requires angle brackets.

## Decisions

Move MQT's MLIR headers and TableGen definitions from `mlir/include/mlir/` to
`mlir/include/mqt/`. The generated include tree follows the source tree through
the CMake subdirectory layout. Update includes, TableGen references, public
header file sets, and documentation together. The `mqt/` prefix distinguishes
MQT headers without enumerating its dialects in formatter rules. C++ namespaces
and CMake target names stay the same.

Generated `.inc` includes must remain after their required macros and
declarations. Vendored sources and historical experiment results are outside the
change. MLIR is unreleased v4 functionality, so this migration does not add an
upgrade notice for a released API.

Keep `<qiskit.h>` angle-bracketed because quotes resolve to the local `Qiskit.h`
on case-insensitive file systems. Keep `<qiskit/funcs_py.h>` in the same form so
the formatter preserves its required position after the umbrella header. Both
headers belong to the third-party include group.

Remove eight redundant include-cleaner suppressions. Four JSON header includes
need no suppression; two JSON aliases need a direct `json_fwd.hpp` include. The
two Qiskit caster suppressions are redundant because the MLIR binding policy
disables include-cleaner. Keep the nanobind caster suppressions outside MLIR:
removing them still produces unused-header warnings with clang-tidy 23.1.1.

## Validation

The release build uses LLVM/MLIR 23.1.0. Source and TableGen include paths
resolve under `mqt/`, including all 98 generated-header references. The C/C++
code bodies are preserved; the suppression cleanup adds two direct JSON
declaration includes. Formatter assertions cover the matching header, include
groups, QDMI header ownership, and macro-dependent includes. Clang-format 23.1.0
reports no changes across all 513 C/C++ and TableGen files.

`ctest --preset release-no-mlir --parallel 4` passed 570 tests and skipped one.
`ctest --preset release --parallel 4` passed 3,452 tests and skipped one after
the final rebuild. `uvx nox -s lint` passed. `uvx nox -s stubs` passed and left
the generated type stubs unchanged.

`uvx nox -s cpp-lint -- --all` checked all 363 eligible files with clang-tidy
23.1.1. It reported nine findings: six unused includes, two exception-escape
warnings, and one widening-cast warning. All nine reproduce on the original
sources from `6a928c868`, using the same compilation settings and normalizing
only the MLIR header prefix to use the current headers. These findings are
outside the include-order change. The default session compares commits and can
check no files before a commit exists, so use `--all` for uncommitted changes.

Direct whole-file clang-tidy rechecks of the three dialect sources whose include
comments changed also completed. They reported generated-header warnings and
existing missing namespace comments in public headers, which the repository lint
session excludes.

For the suppression cleanup, `uvx nox -s cpp-lint -- fee059e85` checked every
line of all five changed C++ files and passed with zero findings.
Include-cleaner trials covered 21 sources to distinguish redundant suppressions
from required nanobind caster suppressions.
